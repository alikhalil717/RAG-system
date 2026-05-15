# Architecture

This project is a local-first RAG pipeline with  LLM call for final generation

## Components

- Document ingestion reads PDFs from `data/pdfs` and extracts text with PyMuPDF.
- Chunking uses `RecursiveCharacterTextSplitter` with a small chunk size and overlap.
- Indexing stores each chunk in an embedded Weaviate collection with vector and keyword search enabled.
- Retrieval uses Weaviate hybrid search and an optional rerank step for better ordering.
- Generation sends the retrieved chunks and chat history to DeepSeek through the OpenAI-compatible `ChatOpenAI` client.

## Runtime

- `app/main.py` starts the reranker Flask server and prepares the Weaviate collection.
- `app/src/reranker_server.py` serves two local endpoints: `/vectors` and `/rerank`.
- `app/src/connect_to_llm.py` formats retrieved chunks into the final prompt and manages per-session chat history.
