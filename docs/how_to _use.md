# How to Use

## Setup

1. **Install dependencies:**

   ```bash
   python -m pip install -r requirements.txt
   ```

2. **Configure API keys:**
   Copy `.env.example` to `.env` in the project root and set your DeepSeek API key:

   ```bash
   cp .env.example .env
   # then edit .env with your editor
   ```

3. **Place PDFs** in `data/pdfs/`.

---

## Run

### Option A — Web UI (recommended)

```bash
python app/web.py
```

Opens a browser UI at **http://localhost:8080** with:

- Multi-session chat sidebar
- Q&A conversation with collapsible source chunks
- Metadata display (book, chapter, page, code language)

### Option B — Python API (programmatic)

```python
from app import main

# Start the runtime
main.run_app(rebuild_index=True, kill_existing_reranker=True)

# Ask a question
answer = main.get_answer("What is this document about?", top_k=5, session_id="default")
print(answer)

# Retrieve raw chunks (no LLM call)
chunks = main.get_chunks("give me code from movies.go", top_k=10)

# Close when done
main.close_app()
```

### Option C — Jupyter Notebook

Open `app/src/test.ipynb` for an interactive step-by-step experience.

---

## Indexing

| First run | `rebuild_index=True` | Creates collection + indexes all PDFs |
| Subsequent runs | `rebuild_index=False` | Reuses existing index (faster) |
| Force refresh | `rebuild_index=True` | Drops and re-creates collection |

When you add new PDFs to `data/pdfs/`, run with `rebuild_index=True` to re-index everything.

---

## Sessions

The system supports **multiple named chat sessions**, each with independent conversation history:

```python
main.get_answer("What is OR?", session_id="session1")
main.get_answer("How to code in Go?", session_id="session2")  # separate history
```

Each session maintains its own `ChatMessageHistory`. Sessions persist in memory until `close_app()`.

---

## What each file does

| File                         | Role                                                                   |
| ---------------------------- | ---------------------------------------------------------------------- |
| `app/main.py`                | Public API: `run_app()`, `get_answer()`, `get_chunks()`, `close_app()` |
| `app/web.py`                 | Flask web server on port 8080                                          |
| `app/templates/index.html`   | Web UI frontend (single page)                                          |
| `app/src/config.py`          | Paths, model names, env var loading                                    |
| `app/src/chunking.py`        | PDF text extraction + code-aware chunking                              |
| `app/src/indexer.py`         | Weaviate collection creation + indexing                                |
| `app/src/search.py`          | Hybrid search (vector + BM25) + rerank                                 |
| `app/src/reranker_server.py` | Flask inference server (embeddings + reranker)                         |
| `app/src/retriever.py`       | Runtime orchestration                                                  |
| `app/src/connect_to_llm.py`  | RAG prompt + DeepSeek LLM call + session history                       |
| `app/src/mock_runner.py`     | Lightweight demo (no Weaviate/LLM needed)                              |
