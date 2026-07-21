# Phase 1 — Existing Pipeline Analysis & Extension Points

Generated: 2026-07-13

---

## 1. Current Pipeline (end-to-end)

```
User types question in web UI (Flask :8080)
    │
    ▼
app/web.py  →  api_query()
    │
    ▼
app/main.py  →  get_answer(query, top_k=60, session_id, expand_window=8)
    │
    ▼
app/src/retriever.py  →  answer_query(AppRuntime, query, top_k, session_id, expand_window)
    │
    ├─ retrieve_chunks(runtime, query, top_k=60, agentic=True, expand_window=8)
    │    │
    │    ├─ rewrite_query_for_retrieval(query)   ← skipped for code queries
    │    ├─ decompose_query(query)                ← LLM: 3-4 sub-queries
    │    ├─ multi_query_search(queries, collection, ...)  ← RRF fusion
    │    └─ expand_chunks(collection, fused, window=8)    ← context expansion
    │
    └─ answer_with_rag(query, chunks, session_id)
         │
         ├─ format_chunks_into_context(chunks)
         ├─ Build SystemMessage + history + HumanMessage
         └─ llm.invoke(messages) → DeepSeek
```

## 2. Weaviate Schema (18 properties)

| Property               | Type   | Vectorized | Purpose                                         |
| ---------------------- | ------ | ---------- | ----------------------------------------------- |
| source                 | TEXT   | No         | PDF filename                                    |
| book                   | TEXT   | No         | Book name (derived from filename)               |
| chapter                | TEXT   | No         | Chapter heading text                            |
| page                   | INT    | No         | Page number                                     |
| chunk                  | TEXT   | **Yes**    | Main chunk content (only vectorized field)      |
| chunk_index            | INT    | No         | Order within book                               |
| has_code               | BOOL   | No         | Does chunk contain code?                        |
| code_language          | TEXT   | No         | Detected language (go, python, etc.)            |
| content_type           | TEXT   | No         | code/explanation/table/terminal/additional_info |
| file_path              | TEXT   | No         | "cmd/api/movies.go" from File: captions         |
| chapter_number         | NUMBER | No         | Sortable float (2.0, 2.1, 2.2, ...)             |
| code_identifiers       | TEXT   | No         | "recoverPanic rateLimit enableCORS"             |
| http_route             | TEXT   | No         | "GET /v1/movies"                                |
| is_full_file_snapshot  | BOOL   | No         | Complete file vs diff/snippet                   |
| file_version_index     | INT    | No         | Version # of file showing (1st, 2nd, ...)       |
| part_index             | INT    | No         | Part N of split code block                      |
| part_total             | INT    | No         | Total parts of split code block                 |
| is_latest_file_version | BOOL   | No         | Post-pass flag for latest showing               |

## 3. Extension Points (Where to Inject the Agentic Pipeline)

### EXT-1: `retriever.py:retrieve_chunks()` — PRIMARY INJECTION POINT

- **Current**: Calls `decompose_query()` → `multi_query_search()` → `expand_chunks()`
- **New**: Replace with `AgenticRetriever.retrieve()` call
- **Lines**: 96-145
- **Risk**: Low — the function already has an `agentic=True` flag

### EXT-2: `connect_to_llm.py:96` — `rewrite_query_for_retrieval()`

- **Current**: Simple query rewriting (extract keywords)
- **New**: Replaced by `query_analyzer.analyze_query()` which does structured analysis
- **Lines**: 96-110

### EXT-3: `connect_to_llm.py:140` — `decompose_query()`

- **Current**: LLM-generated sub-queries (3-4 per query)
- **New**: Replaced by `retrieval_planner.generate_retrieval_plan()` which decides strategies, not just queries
- **Lines**: 140-170

### EXT-4: `search.py:120` — `multi_query_search()`

- **Current**: Hybrid + BM25 on each sub-query, RRF fusion
- **New**: Replaced by `search_strategies.execute_plan()` which dispatches to per-strategy functions
- **Lines**: 120-200

### EXT-5: `search.py:65` — `expand_chunks()`

- **Current**: ±8 window around anchor chunks
- **New**: Kept as-is but moved into `search_strategies` as a post-processing step
- **Lines**: 65-108

### EXT-6: `retriever.py:148` — `answer_query()`

- **Current**: Calls `retrieve_chunks()` then `answer_with_rag()`
- **New**: Wraps in iteration loop with confidence checking
- **Lines**: 148-158

## 4. New Module Map

```
app/src/planning/                    ← NEW PACKAGE
├── __init__.py                      ← Public API surface
├── query_analyzer.py                ← Intent + entity + concept extraction
├── retrieval_planner.py             ← LLM-driven plan generation
├── search_strategies.py             ← Per-strategy Weaviate queries
├── result_fusion.py                 ← Merge + deduplicate + normalize
├── reranker.py                      ← Multi-signal intelligent reranking
├── confidence.py                    ← Confidence scoring + retry logic
└── agentic_retriever.py             ← Top-level orchestrator
```

## 5. Data Flow in New Pipeline

```
User Query: "how to make middleware in Go"
    │
    ▼
query_analyzer.analyze_query()
    ├─ Intent: implementation
    ├─ Entities: {go_functions: [], go_files: [], api_routes: [], ...}
    ├─ Concepts: [recoverPanic, rateLimit, enableCORS, handler chain, ...]
    ├─ Keywords: [middleware, Go]
    └─ Sub-queries: ["how to make middleware in Go", "recoverPanic rateLimit enableCORS", ...]
    │
    ▼
retrieval_planner.generate_retrieval_plan()
    ├─ strategy=code_identifier  query="recoverPanic rateLimit" weight=2.0
    ├─ strategy=hybrid           query="how to make middleware in Go" weight=1.0
    ├─ strategy=bm25             query="middleware Go" weight=0.7
    └─ strategy=hybrid           query="recoverPanic rateLimit enableCORS" weight=0.5
    │
    ▼
search_strategies.execute_plan()
    ├─ search_code_identifier() → 15 chunks (α=0.2, rerank on code_identifiers)
    ├─ search_hybrid()          → 20 chunks (α=0.5, cross-encoder rerank)
    ├─ search_bm25()            → 10 chunks
    └─ search_hybrid()          → 10 chunks
    │
    ▼
result_fusion.fuse_results()
    ├─ Weighted RRF (K=60)
    ├─ Deduplicate by content hash
    └─ → 45 unique chunks
    │
    ▼
reranker.rerank_intelligently()
    ├─ semantic_weight=0.35
    ├─ metadata_weight=0.20
    ├─ entity_weight=0.15
    ├─ intent_weight=0.15
    ├─ topic_weight=0.10
    └─ code_density_weight=0.05
    │
    ▼
confidence.compute_confidence()
    ├─ result_count=0.75, score_distribution=0.60, entity_coverage=0.80
    ├─ metadata_match=0.70, content_diversity=0.65
    └─ overall=0.72 → HIGH CONFIDENCE → no retry
    │
    ▼
connect_to_llm.answer_with_rag()
    └─ DeepSeek generates answer with cited sources
```

## 6. What Phase 1 Delivers vs What's Left for Later Phases

| Delivered Now                       | Phase 2+                                 |
| ----------------------------------- | ---------------------------------------- |
| Full module stubs with docstrings   | LLM-driven plan generation               |
| Heuristic intent classification     | Cross-encoder integration                |
| Rule-based entity extraction        | Weights auto-tuning                      |
| 8 search strategy implementations   | Progressive search implementation        |
| Weighted RRF + min-max fusion       | Confidence-weighted context construction |
| Multi-signal reranking scaffold     | A/B testing framework                    |
| Confidence scoring with retry logic | Metrics collection                       |
| AgenticRetriever orchestrator       | `web.py` integration                     |
| Architecture documentation          | `main.py` public API update              |
