import os
import fitz
import re
from config import *
from utils import print_object_properties
from typing import List, Dict, Any
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
def chunk_text() -> List[Dict[str, Any]]:
    documents = get_all_pdfs()
    all_chunks = []
    for doc in documents:
        source = doc["source"]
        book = os.path.splitext(source)[0]
        pages = doc.get("pages", [])
        current_chapter = None

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=min(CHUNK_SIZE, 350),
            chunk_overlap=max(CHUNK_OVERLAP, 80),
            separators=CHUNK_SEPARATORS,
        )

        # Overall chunk_index across all segments on this page
        global_chunk_idx = 0

        for page_info in pages:
            page_num = page_info.get("page")
            page_text = page_info.get("text", "")

            chap_match = re.search(r"(?i)chapter\s+([0-9]+|[IVXLC]+)", page_text)
            if chap_match:
                current_chapter = chap_match.group(0).strip()

            # ---- Split page into code / non-code segments ----
            segments = re.split(
                r"(\[CODE\].*?\[/CODE\])",
                page_text,
                flags=re.DOTALL,
            )

            for segment in segments:
                if not segment.strip():
                    continue

                # --- Code segment: ALWAYS preserve intact (never split code) ---
                if segment.startswith("[CODE]") and segment.endswith("[/CODE]"):
                    code_content = segment[len("[CODE]"):-len("[/CODE]")]
                    code_lang = _detect_code_language(code_content)

                    code_text = f"[CODE]\n{code_content.strip()}\n[/CODE]"

                    all_chunks.append({
                        "source": source,
                        "book": book,
                        "chapter": current_chapter,
                        "page": _parse_page(page_num),
                        "chunk": code_text,
                        "chunk_index": global_chunk_idx,
                        "has_code": True,
                        "code_language": code_lang,
                    })
                    global_chunk_idx += 1

                # --- Non-code segment: split normally ---
                else:
                    sub_chunks = text_splitter.create_documents([segment])
                    for sc in sub_chunks:
                        content = sc.page_content
                        has_code = "[CODE]" in content
                        code_lang = _detect_code_language(content)
                        if code_lang and not has_code:
                            has_code = True
                        all_chunks.append({
                            "source": source,
                            "book": book,
                            "chapter": current_chapter,
                            "page": _parse_page(page_num),
                            "chunk": content,
                            "chunk_index": global_chunk_idx,
                            "has_code": has_code,
                            "code_language": code_lang,
                        })
                        global_chunk_idx += 1

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