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
    "liberation mono", "dejavu sans mono", "ubuntu mono",
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


def _extract_page_with_code_markers(page) -> str:


    blocks = page.get_text("dict")
    lines_out = []
    in_code_block = False

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
                line_parts.append(text)

            full_line = "".join(line_parts)

            if code_line and full_line.strip():
                # Open a new code block if we aren't already in one
                if not in_code_block:
                    lines_out.append("[CODE]")
                    in_code_block = True
                lines_out.append(full_line)
            else:
                # Close an open code block before emitting normal text
                if in_code_block:
                    lines_out.append("[/CODE]")
                    in_code_block = False
                lines_out.append(full_line)

    # Make sure every opened block is closed
    if in_code_block:
        lines_out.append("[/CODE]")

    return "\n".join(lines_out)


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


def extract_text_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    pages = []
    try:
        doc = fitz.open(pdf_path)
        for i, page in enumerate(doc, start=1):
            text = _extract_page_with_code_markers(page)
            cleaned = _clean_pdf_text(text)
            pages.append({"page": i, "text": cleaned})
        return pages
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return []

def get_all_pdfs() -> List[str]:
    pdf_files = []
    for root, _, files in os.walk(PDF_DATA_PATH):
        for file in files:
            if file.lower().endswith('.pdf'):
                pages = extract_text_from_pdf(os.path.join(root, file))
                pdf_files.append({"source": file, "pages": pages})
    return pdf_files
def chunk_text() -> List[Dict[str, Any]]:
    ducments = get_all_pdfs()
    all_chunks = []
    for doc in ducments:
        source = doc["source"]
        book = os.path.splitext(source)[0]
        pages = doc.get("pages", [])
        current_chapter = None
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=min(CHUNK_SIZE, 350),
            chunk_overlap=max(CHUNK_OVERLAP, 80),
            separators=CHUNK_SEPARATORS
        )
        for page_info in pages:
            page_num = page_info.get("page")
            page_text = page_info.get("text", "")
            chap_match = re.search(r"(?i)chapter\s+([0-9]+|[IVXLC]+)", page_text)
            if chap_match:
                current_chapter = chap_match.group(0).strip()
            chunks = text_splitter.create_documents([page_text])
            for i, chunk in enumerate(chunks):
                content = chunk.page_content
                has_code = "[CODE]" in content
                # Also detect code by pattern even without font-based markers
                # (covers PDFs that use regular fonts for code blocks)
                code_lang = _detect_code_language(content)
                if code_lang and not has_code:
                    has_code = True
                all_chunks.append({
                    "source": source,
                    "book": book,
                    "chapter": current_chapter,
                    "page": int(page_num) if page_num is not None else None,
                    "chunk": content,
                    "chunk_index": i,
                    "has_code": has_code,
                    "code_language": code_lang,
                })
    return all_chunks
def save_chunks(chunks: List[Dict[str, Any]], path: str = CHUNKS_CACHE_PATH) -> None:
    dump(chunks, path)
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