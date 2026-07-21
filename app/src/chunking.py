import os
import fitz
import re
from config import *
from utils import print_object_properties
from typing import Any, Dict, List, Optional, Tuple
from joblib import dump, load
from langchain_text_splitters import RecursiveCharacterTextSplitter


_BOILERPLATE_PATTERNS = (
    r"^archival copy\.",
    r"^for current information, see the osu extension catalog:.*$",
    r"^performance excellence$",
    r"^in the wood products industry$",
    r"^em\s*8718$",
    r"^october 1998$",
    r"^\$2\.50$",
)

_MONOSPACE_FONTS = (
    "courier", "consolas", "monaco", "monospace", "fixed",
    "source code", "fira code", "jetbrains", "hack", "inconsolata",
    "liberation mono", "dejavu sans mono", "ubuntu mono", "ubuntumono",
    "menlo", "lucida console", "cascadia",
)

_CODE_LANGUAGE_HINTS = {
    "python": (r"\bdef\s+\w+\s*\(|^\s*(import|from)\s+\w+|class\s+\w+.*:",),
    "javascript": (r"\b(function|const|let|var)\s+\w+|=>\s*\{",),
    "java": (r"\b(public|private|protected)\s+(static\s+)?(void|int|String|class)\s",),
    "cpp": (r"^\s*#include\s*[<\"]|std::\w+|int\s+main\s*\(",),
    "go": (r"\bfunc\s+\w+\s*\(|package\s+main\b",),
    "rust": (r"\bfn\s+\w+\s*\(|let\s+mut\s+\w+",),
    "sql": (r"\b(SELECT|INSERT|UPDATE|DELETE|CREATE TABLE)\s",),
}


def _protect_code_runs(text: str) -> str:
    """Replace newlines inside consecutive [CODE]...[/CODE] lines with spaces.

    Prevents RecursiveCharacterTextSplitter from splitting code snippets across
    multiple chunks. Since code lines are individually wrapped in
    [CODE]...[/CODE] markers, a run of N code lines looks like:

        [CODE]line1[/CODE]\\n[CODE]line2[/CODE]\\n...

    The splitter's "\\n" separator would split between lines. By replacing
    those internal newlines with spaces, the splitter treats the entire run
    as one "word" and won't split inside it. After splitting, restore_lines()
    reverses this transformation.

    A run is defined as two or more consecutive [CODE]...[/CODE] lines.
    """
    # Match two or more consecutive [CODE]...[/CODE] lines (with optional \n between)
    pattern = re.compile(
        r'(\[CODE\].*?\[/CODE\])\n(?=\[CODE\])',
        re.DOTALL,
    )
    return pattern.sub(r'\1 ', text)


def _restore_code_lines(text: str) -> str:
    """Restore newlines that were replaced by _protect_code_runs.

    Reverse of _protect_code_runs: changes space-separator between
    consecutive [/CODE] and [CODE] back to newlines.
    """
    return re.sub(
        r'(\[/CODE\]) (\[CODE\])',
        r'\1\n\2',
        text,
    )


def _merge_incomplete_code_chunks(
    chunks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Post-processing: merge chunks that contain incomplete code markers.

    After the text splitter runs (even with protection), a few edge cases
    can still produce chunks with unbalanced [CODE]/[/CODE] counts:
    - The splitter splits at ". " or " " inside a code line
    - Very long code runs that exceed chunk_size and get split by character

    This pass walks all chunks, merges any consecutive chunks whose
    [CODE]/[/CODE] counts aren't balanced, and returns the deduplicated list.

    Merged chunks may exceed CHUNK_SIZE — this is intentional, as code
    fidelity is more important than chunk size for programming-book RAG.
    """
    if not chunks:
        return chunks

    merged: List[Dict[str, Any]] = []
    buffer: Optional[Dict[str, Any]] = None

    for chunk in chunks:
        text = chunk.get("chunk", "")

        if buffer is not None:
            # Merge with buffered chunk
            buf_text = buffer.get("chunk", "")
            buffer["chunk"] = buf_text + "\n" + text

            # Check if code is now balanced
            open_count = buffer["chunk"].count("[CODE]")
            close_count = buffer["chunk"].count("[/CODE]")

            if open_count == close_count:
                merged.append(buffer)
                buffer = None
            # else: keep buffering — code is still incomplete
        else:
            open_count = text.count("[CODE]")
            close_count = text.count("[/CODE]")

            if open_count != close_count:
                # Code snippet was cut — start buffering
                buffer = dict(chunk)
            else:
                merged.append(dict(chunk))

    # Append any remaining buffered chunk (unlikely but safe)
    if buffer is not None:
        merged.append(buffer)

    if len(merged) != len(chunks):
        print(
            f"[merge_code] {len(chunks)} chunks → {len(merged)} "
            f"after merging incomplete code fragments "
            f"({len(chunks) - len(merged)} merged away)"
        )

    return merged


# ==================================================================================== #
# NEW: strategy-specific patterns and constants
# ==================================================================================== #

# Matches the book's "File: cmd/api/movies.go" captions, which precede almost every
# code listing and are the single strongest signal in the corpus for tracking which
# file a piece of code belongs to and how it evolves across chapters.
_FILE_PATH_PATTERN = re.compile(r"File:\s*([A-Za-z0-9_\-./$]+)")

# Matches the book's chapter/subchapter headings, e.g.:
#   "Chapter 2.1.\nProject Setup and Skeleton Structure"
#   "Chapter 2.\nGetting Started"
# Subchapters (2.1, 2.2, ...) are the real topical unit in this book, so we capture
# the minor number separately and expose it as a sortable/filterable chapter_number.
_CHAPTER_PATTERN = re.compile(
    r"Chapter\s+(\d{1,2})(?:\.(\d{1,2}))?\.\s*\n?\s*([A-Z][A-Za-z0-9 ,\-'’&]{2,90})"
)

# Go symbol extraction — used to build the `code_identifiers` field so that BM25 can
# match exact function/type/package names even when the embedding model blurs them.
_GO_FUNC_PATTERN = re.compile(r"func\s+(?:\([^)]*\)\s+)?(\w+)\s*\(")
_GO_TYPE_PATTERN = re.compile(r"\btype\s+(\w+)\s+(?:struct|interface)\b")
_GO_PACKAGE_LINE_PATTERN = re.compile(r"^\s*package\s+(\w+)", re.MULTILINE)

# HTTP method + path detector. Reused for both code blocks (route registration lines
# like `router.HandlerFunc(http.MethodGet, "/v1/movies/:id", ...)` won't match this,
# but flattened endpoint tables and curl examples will) and for table detection.
_HTTP_ROUTE_PATTERN = re.compile(r"\b(GET|POST|PUT|PATCH|DELETE|OPTIONS)\s+(/[\w\-/:]*)")

# First-line keywords that reliably identify the book's reference tables (Method/URL
# tables, Go-type<->JSON-type mapping tables, settings tables, etc.).
_TABLE_HEADER_KEYWORDS = (
    "Method URL Pattern",
    "Go type",
    "JSON type",
    "PostgreSQL type",
    "Signal Description",
    "Key Description",
    "Level Description",
)

# The ellipsis-omission marker the book uses to indicate a code listing is a diff /
# partial snippet rather than a complete file, e.g.:
#   package main
#   ... // Indicates that some existing code has been omitted.
_ELLIPSIS_OMISSION_PATTERN = re.compile(r"^\s*\.\.\.\s*(//.*)?$", re.MULTILINE)

# Ceiling above which a single code listing is split on top-level func/type
# boundaries rather than left as one oversized chunk. Chosen to comfortably fit a
# full handler + a bit of surrounding context without splitting mid-function.
MAX_CODE_CHUNK_CHARS = 2200


def _is_monospace_font(font_name: str) -> bool:
    if not font_name:
        return False
    lower = font_name.lower()
    return any(mf in lower for mf in _MONOSPACE_FONTS)


def _detect_code_language(text: str) -> str:
    for lang, patterns in _CODE_LANGUAGE_HINTS.items():
        for pat in patterns:
            if re.search(pat, text, re.MULTILINE):
                return lang
    return ""


def _line_looks_like_code(line: str) -> bool:
    """Check if a single line of text looks like it belongs to a code snippet.

    Uses the same language-specific heuristics as *_detect_code_language*,
    plus common structural patterns (indentation, braces, etc.).
    """
    stripped = line.strip()
    if not stripped:
        return False

    # Language-specific patterns (matches keyword at start of line)
    for lang, patterns in _CODE_LANGUAGE_HINTS.items():
        for pat in patterns:
            if re.search(pat, stripped):
                return True

    # Structural code patterns (works for indented AND unindented code)
    if re.match(
        r"^\s*(package|import|func|return|if|for|while|switch|case|break|continue"
        r"|const|type|struct|interface|map|chan|go\s+func|defer|select"
        r"|public|private|protected|static|void|int|float|double"
        r"|String|bool|byte|char|long|short|unsigned|signed"
        r"|var|let|const|class|def|defn|fn|fun|lambda"
        r"|from|import|export|module|require|use|include"
        r"|try|catch|throw|throws|finally|raise|except"
        r"|print|println|printf|fmt\.|log\.|Sprintf|Errorf"
        r"|err\s*!=\s*nil|err\s*!=\s*null"
        r"|new|make|append|delete|close|len|cap|copy"
        r"|true|false|nil|null|undefined|NaN"
        r"</?\w+|SELECT|INSERT|UPDATE|DELETE|CREATE|ALTER|DROP|INTO|VALUES|SET"
        r"|http\.|json\.|ioutil\.|os\.|io\.|strings\.|strconv\.|bytes\.|time\.)"
        .replace(r"|SELECT", r"|SELECT"),  # force raw upper-case
        stripped, re.IGNORECASE,
    ):
        return True

    # Import paths (e.g. "golang.org/x/time/rate", "greenlight.alexedwards.net/...")
    if re.match(r"^\s*[\"<][a-zA-Z0-9_.\-/]+[>\"]", stripped):
        return True

    # Indented + keyword combo (e.g. "    return ...", "    err := ...")
    if re.match(r"^\s{2,}", stripped) and re.search(
        r"\b(return|import|from|if|for|while|try|catch|throw|class|def|function"
        r"|const|let|var|switch|case|break|continue|package|pub\s+fn|fmt\.|err\b)",
        stripped,
    ):
        return True

    # Braces or semicolons — common in C-family languages
    if stripped.endswith("{") or stripped == "}" or stripped.startswith("}"):
        return True
    if stripped.endswith(";") and len(stripped) > 2:
        return True

    # Comment-like lines — common in code
    if re.match(r"^\s*[/#].*", stripped) and len(stripped) > 5:
        return True

    return False


def _extract_page_with_code_markers(page) -> tuple:
    """Extract page text, marking code blocks with [CODE]…[/CODE].

    Uses two detection strategies:
    1. **Font-based** – lines using monospace fonts are tagged as code.
    2. **Pattern-based** – if font-based detection finds nothing, falls
       back to heuristic pattern matching on the raw text.

    Returns ``(text, code_open)`` where *code_open* is True when a code
    block was still open at the end of the page (artificially closed by
    this function because it continues on the next page).
    """
    blocks = page.get_text("dict")
    lines_out = []
    in_code_block = False
    font_found_code = False  # did any line use a monospace font?

    for block in blocks.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            line_parts = []
            code_line = False
            for span in line.get("spans", []):
                font = span.get("font", "")
                text = span.get("text", "")
                if _is_monospace_font(font):
                    code_line = True
                    font_found_code = True
                line_parts.append(text)

            full_line = "".join(line_parts)

            # Use font detection first; fall back to pattern detection later
            if code_line and full_line.strip():
                if not in_code_block:
                    lines_out.append("[CODE]")
                    in_code_block = True
                lines_out.append(full_line)
            else:
                if in_code_block:
                    lines_out.append("[/CODE]")
                    in_code_block = False
                lines_out.append(full_line)

    if in_code_block:
        lines_out.append("[/CODE]")

    text = "\n".join(lines_out)

    # ---- Pattern-based fallback ----
    # If font detection never found code, try pattern matching on the raw
    # text from PyMuPDF (before any [CODE] markers).
    if not font_found_code:
        raw_text = page.get_text("text")
        text, code_open = _insert_code_markers_by_pattern(raw_text)
        return text, code_open

    code_open = in_code_block
    return text, code_open


def _insert_code_markers_by_pattern(raw_text: str) -> tuple:
    """Fallback: insert [CODE]…[/CODE] around lines that look like code.

    This handles PDFs where code uses proportional fonts so font-based
    detection misses it entirely.

    Returns ``(text, code_open)`` where *code_open* is True when a code
    block was still open at the end of the page.
    """
    lines = raw_text.split("\n")
    result = []
    in_code = False

    for line in lines:
        is_code = _line_looks_like_code(line)

        if is_code and not in_code:
            result.append("[CODE]")
            result.append(line)
            in_code = True
        elif not is_code and in_code:
            result.append("[/CODE]")
            result.append(line)
            in_code = False
        else:
            result.append(line)

    code_open = in_code
    if in_code:
        result.append("[/CODE]")

    return "\n".join(result), code_open


def _clean_pdf_text(text: str) -> str:
    cleaned_lines = []
    in_code_block = False
    for line in text.splitlines():
        stripped = line.strip()

        # Track [CODE] ... [/CODE] boundaries.
        # Support both standalone-line markers (normal case from
        # _extract_page_with_code_markers) and inline markers as a fallback.
        if not in_code_block and "[CODE]" in stripped:
            in_code_block = True
            cleaned_lines.append("[CODE]")
            remainder = stripped[stripped.index("[CODE]") + len("[CODE]"):]
            if "[/CODE]" in remainder:
                inner = remainder[:remainder.index("[/CODE]")]
                if inner.strip():
                    cleaned_lines.append(inner)
                in_code_block = False
                cleaned_lines.append("[/CODE]")
            elif remainder.strip():
                cleaned_lines.append(remainder)
            continue
        if in_code_block and "[/CODE]" in stripped:
            before = stripped[:stripped.index("[/CODE]")]
            if before.strip():
                cleaned_lines.append(before)
            in_code_block = False
            cleaned_lines.append("[/CODE]")
            continue

        if in_code_block:
            # Preserve code lines as-is (keep indentation)
            cleaned_lines.append(line)
            continue

        if not stripped:
            continue
        normalized = re.sub(r"\s+", " ", stripped).lower()
        if any(re.match(pattern, normalized) for pattern in _BOILERPLATE_PATTERNS):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def _merge_code_across_pages(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge code blocks that span consecutive pages.

    Works at the *line level* by detecting when the last code line of page N
    would naturally continue onto page N+1. It removes the artificial
    ``[/CODE]`` / ``[CODE]`` boundary and concatenates the raw code lines,
    keeping any trailing non-code text on page N+1 separate.

    This is more robust than text-boundary matching because a single page can
    contain multiple ``[CODE]…[/CODE]`` blocks, and only the *last* one may
    continue to the next page.
    """
    if not pages:
        return pages

    merged = []
    i = 0
    while i < len(pages):
        page = dict(pages[i])
        text = page["text"]

        if i + 1 < len(pages):
            next_text = pages[i + 1]["text"]

            # Split both pages into lines
            page_lines = text.split("\n")
            next_lines = next_text.split("\n")

            # Find the last [/CODE] boundary on this page:
            #   We look for the last occurrence of [/CODE] on page N.
            #   If the NEXT page starts with [CODE], it's a continuation.
            last_close_idx = -1
            for idx in range(len(page_lines) - 1, -1, -1):
                if page_lines[idx].strip() == "[/CODE]":
                    last_close_idx = idx
                    break

            first_open_idx = -1
            for idx in range(len(next_lines)):
                if next_lines[idx].strip() == "[CODE]":
                    first_open_idx = idx
                    break

            if last_close_idx >= 0 and first_open_idx >= 0:
                # Extract the content of the last code block on page N
                # (lines between its [CODE] and [/CODE])
                last_code_start = -1
                for idx in range(last_close_idx - 1, -1, -1):
                    if page_lines[idx].strip() == "[CODE]":
                        last_code_start = idx
                        break
                code_lines_n = (
                    [l for l in page_lines[last_code_start + 1:last_close_idx] if l.strip()]
                    if last_code_start >= 0 else []
                )

                # Extract the content of the first code block on page N+1
                first_code_end = -1
                for idx in range(first_open_idx + 1, len(next_lines)):
                    if next_lines[idx].strip() == "[/CODE]":
                        first_code_end = idx
                        break
                code_lines_n1 = (
                    [l for l in next_lines[first_open_idx + 1:first_code_end] if l.strip()]
                    if first_code_end >= 0 else [l for l in next_lines[first_open_idx + 1:] if l.strip()]
                )

                # Heuristic: only merge if at least one side has ≥ 3 non-empty
                # code lines (rules out single-line items like `/v1/healthcheck`
                # that happen to sit at a page boundary but aren't continuations).
                if len(code_lines_n) >= 3 or len(code_lines_n1) >= 3:
                    # We found a genuine continuation — merge
                    before_close = page_lines[:last_close_idx]
                    after_open = next_lines[first_open_idx + 1:]

                    if first_code_end >= 0:
                        code_lines = before_close + after_open[:first_code_end + 1]
                        remaining = after_open[first_code_end + 1:]
                    else:
                        code_lines = before_close + after_open
                        remaining = []

                    while code_lines and not code_lines[-1].strip():
                        code_lines.pop()

                    new_text = "\n".join(code_lines)
                    if remaining:
                        new_text += "\n" + "\n".join(remaining)

                    page["text"] = new_text
                    # Keep page as the original starting page (int) — Weaviate
                    # schema requires DataType.INT, not a range string.

                    # Skip the absorbed next page
                    i += 1

        merged.append(page)
        i += 1

    return merged


def extract_text_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    pages = []
    try:
        doc = fitz.open(pdf_path)
        for i, page in enumerate(doc, start=1):
            raw_text, code_open = _extract_page_with_code_markers(page)
            cleaned = _clean_pdf_text(raw_text)
            pages.append({
                "page": i,
                "text": cleaned,
                "_code_open": code_open,       # True if code continued to next page
            })
        return pages
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return []


def get_all_pdfs() -> List[Dict[str, Any]]:
    pdf_files = []
    for root, _, files in os.walk(PDF_DATA_PATH):
        for file in files:
            if file.lower().endswith('.pdf'):
                pages = extract_text_from_pdf(os.path.join(root, file))
                pages = _merge_code_across_pages(pages)
                pdf_files.append({"source": file, "pages": pages})
    return pdf_files


# ==================================================================================== #
# NEW: strategy-specific helper functions
# ==================================================================================== #

def _format_chapter(match: "re.Match") -> Tuple[str, float]:
    """Turn a _CHAPTER_PATTERN match into (full chapter string, chapter_number).

    "Chapter 2.1." + "Project Setup..." -> ("Chapter 2.1. Project Setup...", 2.1)
    "Chapter 2."   + "Getting Started"  -> ("Chapter 2. Getting Started", 2.0)
    """
    major, minor, title = match.group(1), match.group(2), match.group(3).strip()
    title = re.sub(r"\s+", " ", title)
    if minor:
        return f"Chapter {major}.{minor}. {title}", float(f"{major}.{minor}")
    return f"Chapter {major}. {title}", float(major)


def _is_terminal_block(content: str) -> bool:
    """Distinguish a terminal/command-line example from real source code.

    Both are typeset in a monospace font in the PDF, so the font-based [CODE]
    detection upstream conflates them. We split them back apart here using the
    book's consistent "$ " command-prompt convention plus raw HTTP response
    lines, both of which never appear in actual Go/SQL/JSON code listings.
    """
    lines = [l for l in content.splitlines() if l.strip()]
    if not lines:
        return False
    dollar_lines = sum(1 for l in lines if l.strip().startswith("$"))
    http_resp_lines = sum(1 for l in lines if re.match(r"^HTTP/1\.\d\s+\d{3}", l.strip()))
    first_is_command = lines[0].strip().startswith("$")
    density = (dollar_lines + http_resp_lines) / len(lines)
    return first_is_command or density >= 0.25


def _is_table_like(text: str) -> bool:
    """Detect the book's reference tables (endpoint tables, type-mapping tables, …).

    PDF text extraction collapses column spacing to single spaces, so we can't
    rely on whitespace-run column detection. Instead we look for either (a) a
    run of multiple HTTP method/route lines — the highest-value table type for
    retrieval — or (b) a first line matching one of the book's recurring table
    headers. Tables that match neither heuristic fall back to normal prose
    chunking rather than being lost; see the accompanying write-up for why a
    fully general table parser isn't worth the brittleness here.
    """
    stripped = text.strip()
    if not stripped:
        return False
    lines = [l for l in stripped.splitlines() if l.strip()]
    if len(lines) < 2:
        return False

    route_lines = sum(1 for l in lines if _HTTP_ROUTE_PATTERN.search(l))
    if route_lines >= 2:
        return True

    first_line = lines[0].strip()
    if any(first_line.startswith(kw) for kw in _TABLE_HEADER_KEYWORDS):
        return True

    return False


def _is_full_file_snapshot(code_content: str) -> bool:
    """Heuristic: is this code listing a *complete* file, or a diff/snippet?

    A listing is treated as a full snapshot only if it contains no ellipsis
    omission marker ("... // Indicates that some existing code has been
    omitted."). For Go files specifically we also require the listing to
    start with a `package` declaration, since diff snippets sometimes restate
    `package main` at the top immediately followed by an omission marker.
    """
    stripped = code_content.strip()
    if not stripped:
        return False
    first_line = stripped.splitlines()[0].strip()
    has_ellipsis = bool(_ELLIPSIS_OMISSION_PATTERN.search(code_content))
    if first_line.startswith("package "):
        return not has_ellipsis
    return not has_ellipsis


def _extract_go_import_paths(code: str) -> List[str]:
    imports: List[str] = []
    block = re.search(r"import\s*\(([^)]*)\)", code, re.DOTALL)
    if block:
        for line in block.group(1).splitlines():
            m = re.search(r'"([^"]+)"', line)
            if m:
                imports.append(m.group(1))
    else:
        for m in re.finditer(r'import\s+"([^"]+)"', code):
            imports.append(m.group(1))
    return imports


def _extract_code_identifiers(code: str) -> str:
    """Extract Go func/type/package/import names as a space-joined BM25-searchable
    string. Deliberately kept as one flat text field rather than several
    per-symbol-kind schema properties — a single book doesn't have enough volume
    to justify the extra schema surface, and BM25 doesn't care which "kind" of
    identifier it's matching against.
    """
    identifiers = set()
    identifiers.update(_GO_FUNC_PATTERN.findall(code))
    identifiers.update(_GO_TYPE_PATTERN.findall(code))
    identifiers.update(_GO_PACKAGE_LINE_PATTERN.findall(code))
    for path in _extract_go_import_paths(code):
        identifiers.add(path)
        identifiers.add(path.rsplit("/", 1)[-1])
    identifiers.discard("")
    return " ".join(sorted(identifiers))


def _extract_http_routes(text: str) -> str:
    routes = set()
    for method, path in _HTTP_ROUTE_PATTERN.findall(text):
        routes.add(f"{method} {path}")
    return " | ".join(sorted(routes))


def _last_sentence(text: str, max_chars: int = 220) -> str:
    """Grab the tail of a prose chunk so it can be prepended to the code block
    that immediately follows it. Code embeds poorly in isolation (identifier
    soup, thin semantic signal); code + the sentence that introduced it
    embeds noticeably better and gives BM25 a natural-language anchor too.
    """
    text = text.strip()
    if not text:
        return ""
    sentences = re.split(r"(?<=[.!?:])\s+", text)
    tail = sentences[-1].strip() if sentences else text
    if len(tail) < 15 and len(sentences) > 1:
        tail = " ".join(sentences[-2:]).strip()
    return tail[-max_chars:]


def _split_large_code_block(code_content: str, file_path: Optional[str]) -> List[str]:
    """Split an oversized code listing on top-level func/type boundaries only.

    Never splits mid-function or mid-statement. If no safe boundary is found
    the block is returned intact (occasionally producing one large chunk is
    preferable to corrupting the code).
    """
    if len(code_content) <= MAX_CODE_CHUNK_CHARS:
        return [code_content]

    boundary_pattern = re.compile(r"\n(?=\s*(?:func\s|type\s+\w+\s+(?:struct|interface)\b))")
    pieces = [p for p in boundary_pattern.split(code_content) if p.strip()]

    if len(pieces) <= 1:
        return [code_content]

    header = f"// File: {file_path}\n" if file_path else ""
    parts = [pieces[0]] + [f"{header}{piece}" for piece in pieces[1:]]
    return parts


def _mark_latest_file_versions(chunks: List[Dict[str, Any]]) -> None:
    """Post-pass: flag which chunks belong to the most recent showing of each
    file, preferring the most recent *complete* (is_full_file_snapshot) showing
    over the most recent diff/snippet showing. This is what lets a "give me the
    full X.go" query bypass vector search entirely and fetch by (book,
    file_path, is_latest_file_version) instead.
    """
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for c in chunks:
        fp = c.get("file_path")
        if not fp:
            continue
        groups.setdefault((c["book"], fp), []).append(c)

    for group in groups.values():
        full_versions = [c for c in group if c.get("is_full_file_snapshot")]
        pool = full_versions if full_versions else group
        latest_version_index = max((c.get("file_version_index") or 0) for c in pool)
        for c in group:
            if c.get("file_version_index") == latest_version_index:
                c["is_latest_file_version"] = True


def _new_chunk(
    *, source: str, book: str, chapter: Optional[str], page, chunk: str,
    chunk_index: int, has_code: bool, code_language: str, content_type: str,
    file_path: str = "", chapter_number: Optional[float] = None,
    code_identifiers: str = "", http_route: str = "",
    is_full_file_snapshot: bool = False, file_version_index: Optional[int] = None,
    part_index: int = 1, part_total: int = 1,
) -> Dict[str, Any]:
    """Single place that assembles a chunk dict, so every branch of chunk_text()
    populates exactly the same set of fields (avoids silently-missing keys)."""
    return {
        "source": source,

        "book": book,

        "chapter": chapter,

        "page": _parse_page(page),

        "chunk": chunk,

        "chunk_index": chunk_index,

        "has_code": has_code,

        "code_language": code_language,

        "content_type": content_type,
        "file_path": file_path,
        "chapter_number": chapter_number,
        "code_identifiers": code_identifiers,
        "http_route": http_route,
        "is_full_file_snapshot": is_full_file_snapshot,
        "file_version_index": file_version_index,
        "part_index": part_index,
        "part_total": part_total,
        "is_latest_file_version": False,  # filled in by _mark_latest_file_versions()
    }


def chunk_text() -> List[Dict[str, Any]]:
    """Chunk every PDF in PDF_DATA_PATH following the book-aware strategy:

    - subchapter (e.g. "Chapter 2.1. ...") is the topical scope; chapter_number
      is captured separately as a sortable float.
    - `File: <path>` captions are tracked as running state so every code chunk
      knows which file it belongs to, including across page boundaries.
    - code, terminal examples, reference tables, "Additional Information"
      asides and regular prose are each classified via `content_type` and
      chunked differently — code/terminal/tables are never split by the
      character-based splitter.
    - the sentence introducing a code block is prepended to it, since code
      embeds poorly in isolation.
    - oversized code listings are split on func/type boundaries, never
      mid-statement.
    - a final pass marks which showing of each file is the most recent
      (preferring full-file snapshots), enabling direct lookup for
      "give me the full X.go" style queries without going through vector
      search at all.
    """
    documents = get_all_pdfs()
    all_chunks: List[Dict[str, Any]] = []

    # Keyed by (book, file_path) rather than just file_path, so this stays
    # correct if the knowledge base ever grows beyond a single book.
    file_version_counts: Dict[Tuple[str, str], int] = {}

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=min(CHUNK_SIZE, 350),
        chunk_overlap=max(CHUNK_OVERLAP, 80),
        separators=CHUNK_SEPARATORS,
    )

    for doc in documents:
        source = doc["source"]
        book = os.path.splitext(source)[0]
        pages = doc.get("pages", [])

        current_chapter: Optional[str] = None
        current_chapter_number: Optional[float] = None
        current_file_path: Optional[str] = None
        fresh_caption_seen = False
        in_additional_info = False
        pending_explanation_tail = ""

        global_chunk_idx = 0

        for page_info in pages:
            page_num = page_info.get("page")
            page_text = page_info.get("text", "")

            chap_match = _CHAPTER_PATTERN.search(page_text)
            if chap_match:
                current_chapter, current_chapter_number = _format_chapter(chap_match)
                in_additional_info = False  # a new (sub)chapter closes the previous aside

            segments = re.split(r"(\[CODE\].*?\[/CODE\])", page_text, flags=re.DOTALL)

            for segment in segments:
                if not segment.strip():
                    continue

                is_code_segment = segment.startswith("[CODE]") and segment.endswith("[/CODE]")

                # ------------------------------------------------------------------ #
                # CODE / TERMINAL segment — never split by the character splitter
                # ------------------------------------------------------------------ #
                if is_code_segment:
                    code_content = segment[len("[CODE]"):-len("[/CODE]")].strip()
                    if not code_content:
                        continue

                    if _is_terminal_block(code_content):
                        content_type = "terminal"
                        code_lang = ""
                        has_code = False
                        file_path_for_chunk = None
                        file_version_index = None
                        is_full_snapshot = False
                        identifiers = ""
                    else:
                        content_type = "code"
                        code_lang = _detect_code_language(code_content)
                        has_code = True
                        file_path_for_chunk = current_file_path or ""

                        file_version_index = None
                        is_full_snapshot = False
                        if file_path_for_chunk:
                            key = (book, file_path_for_chunk)
                            if fresh_caption_seen:
                                file_version_counts[key] = file_version_counts.get(key, 0) + 1
                                fresh_caption_seen = False
                            file_version_index = file_version_counts.get(key, 1)
                            is_full_snapshot = _is_full_file_snapshot(code_content)

                        identifiers = _extract_code_identifiers(code_content)

                    http_route = _extract_http_routes(code_content)

                    prefix = ""
                    if content_type == "code" and pending_explanation_tail:
                        prefix = f"{pending_explanation_tail}\n\n"

                    parts = _split_large_code_block(
                        code_content,
                        file_path_for_chunk if content_type == "code" else None,
                    )
                    for part_idx, part_text in enumerate(parts, start=1):
                        body = f"[CODE]\n{part_text.strip()}\n[/CODE]"
                        full_text = f"{prefix}{body}" if part_idx == 1 else body

                        all_chunks.append(_new_chunk(
                            source=source, book=book, chapter=current_chapter,
                            page=page_num, chunk=full_text, chunk_index=global_chunk_idx,
                            has_code=has_code, code_language=code_lang,
                            content_type=content_type,
                            file_path=file_path_for_chunk or "",
                            chapter_number=current_chapter_number,
                            code_identifiers=identifiers, http_route=http_route,
                            is_full_file_snapshot=is_full_snapshot,
                            file_version_index=file_version_index,
                            part_index=part_idx, part_total=len(parts),
                        ))
                        global_chunk_idx += 1

                    pending_explanation_tail = ""
                    continue

                # ------------------------------------------------------------------ #
                # NON-CODE segment — scan for File: captions / "Additional
                # Information" headings, then split into tables vs. prose
                # ------------------------------------------------------------------ #
                caption_matches = list(_FILE_PATH_PATTERN.finditer(segment))
                if caption_matches:
                    current_file_path = caption_matches[-1].group(1).rstrip(".,;:")
                    fresh_caption_seen = True

                if re.search(r"\bAdditional Information\b", segment):
                    in_additional_info = True

                for para in re.split(r"\n\s*\n", segment):
                    if not para.strip():
                        continue

                    if _is_table_like(para):
                        all_chunks.append(_new_chunk(
                            source=source, book=book, chapter=current_chapter,
                            page=page_num, chunk=para.strip(), chunk_index=global_chunk_idx,
                            has_code=False, code_language="", content_type="table",
                            chapter_number=current_chapter_number,
                            http_route=_extract_http_routes(para),
                        ))
                        global_chunk_idx += 1
                        pending_explanation_tail = ""
                        continue

                    content_type = "additional_info" if in_additional_info else "explanation"

                    # Protect code blocks before splitting so code snippets
                    # stay together in one chunk instead of being fragmented
                    protected_para = _protect_code_runs(para)

                    sub_chunks = text_splitter.create_documents([protected_para])
                    for sc in sub_chunks:
                        # Restore newlines that were replaced during protection
                        content = _restore_code_lines(sc.page_content)
                        if not content.strip():
                            continue
                        has_code_inline = "[CODE]" in content
                        code_lang_inline = _detect_code_language(content)
                        if code_lang_inline and not has_code_inline:
                            has_code_inline = True

                        all_chunks.append(_new_chunk(
                            source=source, book=book, chapter=current_chapter,
                            page=page_num, chunk=content, chunk_index=global_chunk_idx,
                            has_code=has_code_inline, code_language=code_lang_inline,
                            content_type=content_type,
                            chapter_number=current_chapter_number,
                            http_route=_extract_http_routes(content),
                        ))
                        global_chunk_idx += 1

                    if sub_chunks:
                        pending_explanation_tail = _last_sentence(sub_chunks[-1].page_content)

    _mark_latest_file_versions(all_chunks)

    # Final pass: merge any chunks that still have incomplete code markers
    all_chunks = _merge_incomplete_code_chunks(all_chunks)

    return all_chunks


def _parse_page(page_val):
    """Normalize the page field: could be int, str (page range "1-2"), or None."""
    if page_val is None:
        return None
    if isinstance(page_val, (int, str)):
        return page_val
    return int(page_val)


def save_chunks(chunks: List[Dict[str, Any]], path: str = CHUNKS_CACHE_PATH) -> None:
    dump(chunks, path, compress=3)


def load_chunks(path: str = CHUNKS_CACHE_PATH) -> List[Dict[str, Any]]:
    if os.path.exists(path):
        return load(path)
    else:
        return []


if __name__ == "__main__":
    if os.path.exists(CHUNKS_CACHE_PATH):
        print(f"Loading chunks from cache at {CHUNKS_CACHE_PATH}...")
        chunks = load_chunks()
        print(f"Loaded {len(chunks)} chunks from cache.")
        print_object_properties(chunks[:2])
    else:
        chunks = chunk_text()
        print(f"Total chunks created: {len(chunks)}")
        save_chunks(chunks)
        print(f"Chunks saved to {CHUNKS_CACHE_PATH}")