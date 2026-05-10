# 1. Imports
# os, fitz (PyMuPDF) – for PDF reading.
import os
import fitz
from config import *
from utils import print_object_properties
from typing import List, Dict, Any
from joblib import dump, load
from langchain_text_splitters import RecursiveCharacterTextSplitter
def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file."""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return ""
# Function: get_all_pdfs() -> List[Dict[str, str]]
# Use os.listdir(PDF_DATA_PATH).

# Filter for files ending with '.pdf'.

# For each PDF, call extract_text_from_pdf() to get the text.

# Return a list of dictionaries: [{"source": "filename.pdf", "text": "the full document text"}, ...].

def get_all_pdfs() -> List[str]:
    """Get a list of all PDF file paths in the PDF_DATA_PATH directory."""
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
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
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
    """Save the list of chunks to a file using joblib."""
    dump(chunks, path)
def load_chunks(path: str = CHUNKS_CACHE_PATH) -> List[Dict[str, Any]]:
    """Load the list of chunks from a file using joblib."""
    if os.path.exists(path):
        return load(path)
    else:
        return []
if __name__ == "__main__":
    if os.path.exists(CHUNKS_CACHE_PATH):
        print(f"Loading chunks from cache at {CHUNKS_CACHE_PATH}...")
        chunks = load_chunks()
        print(f"Loaded {len(chunks)} chunks from cache.")
        print_object_properties(chunks[:2])  # Print properties of the first 2 chunks
    else:
        print("Extracting and chunking PDF documents...")
        chunks = chunk_text()
        print(f"Total chunks created: {len(chunks)}")
        save_chunks(chunks)
        print(f"Chunks saved to {CHUNKS_CACHE_PATH}")