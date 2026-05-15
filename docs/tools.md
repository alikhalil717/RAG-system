# Tools and libraries
The system uses the following libraries and services.

 Tool                    |  Purpose

 Python            | Project language and runtime
 PyMuPDF (`fitz`)    | PDF text extraction
 LangChain      | Prompting, chat history, and text splitters
 Weaviate          | Vector database and hybrid retrieval
 Weaviate Python client | Embedded collection management and queries
 sentence-transformers  | Local embedding and rerank models
 Flask               | Local reranker API
 requests              | HTTP calls to the local reranker service
 DeepSeek API          | Final answer generation
 joblib           | Chunk cache persistence
 tqdm                | Indexing progress display

## What each one does

- PyMuPDF extracts offline document text before chunking.
- LangChain provides the prompt and message-history primitives used by the chat flow.
- Weaviate stores chunks, performs hybrid search, and supports reranking.
- sentence-transformers provides the local embedding model and the cross-encoder reranker.
- Flask exposes the local inference endpoints used by indexing and retrieval.
- DeepSeek provides the final language model used to answer questions from retrieved context.
