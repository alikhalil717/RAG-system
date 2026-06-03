
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


def _clean_pdf_text(text: str) -> str:
    cleaned_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        normalized = re.sub(r"\s+", " ", stripped).lower()
        if any(re.match(pattern, normalized) for pattern in _BOILERPLATE_PATTERNS):
            continue
        cleaned_lines.append(stripped)
    return "\n".join(cleaned_lines)


def extract_text_from_pdf(pdf_path: str) -> str:
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return _clean_pdf_text(text)
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return ""

def get_all_pdfs() -> List[str]:
    pdf_files = []
    for root, _, files in os.walk(PDF_DATA_PATH):
        for file in files:
            if file.lower().endswith('.pdf'):
                text = extract_text_from_pdf(os.path.join(root, file))
                pdf_files.append({"source": file, "text": text})
    return pdf_files
def chunk_text() -> List[Dict[str, Any]]:
    ducments = get_all_pdfs()
    all_chunks = []
    for doc in ducments:
        text = doc["text"]
        source = doc["source"]
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=min(CHUNK_SIZE, 350),
            chunk_overlap=max(CHUNK_OVERLAP, 80),
            separators=CHUNK_SEPARATORS
        )
        chunks = text_splitter.create_documents([text])
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "source": source,
                "chunk": chunk.page_content,
                "chunk_index": i
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