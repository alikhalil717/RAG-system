# Phase 2 — Agentic Retrieval Architecture Design

Generated: 2026-07-13

---

## 1. Integration Strategy: Non-Breaking, Dual-Path

The core principle is **backward compatibility via mode toggles**:

```
retrieve_chunks(agentic="full")    → NEW agentic pipeline
retrieve_chunks(agentic="legacy")  → ORIGINAL decompose+multi_query+expand
retrieve_chunks(agentic="off")     → Single hybrid search only
```

No existing code is deleted. The new pipeline is an _additional path_, not a replacement.

---

## 2. Full Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         app/web.py  (Flask :8080)                    │
│  POST /api/query  {query, top_k, session_id, agentic: "full"}       │
└────────────────────────────┬────────────────────────────────────────┘
                             │
              ┌──────────────┴──────────────┐
              │  agentic="full"             │  agentic="legacy"/"off"
              ▼                             ▼
┌──────────────────────────┐   ┌──────────────────────────┐
│     AGENTIC PATH (NEW)    │   │     LEGACY PATH (ORIG)    │
│                           │   │                           │
│  planning/query_analyzer  │   │  decompose_query()        │
│         │                 │   │         │                 │
│         ▼                 │   │         ▼                 │
│  planning/retrieval_      │   │  multi_query_search()     │
│  planner                  │   │         │                 │
│         │                 │   │         ▼                 │
│         ▼                 │   │  expand_chunks()          │
│  planning/search_strategies│  │                           │
│  (execute_plan)           │   │                           │
│         │                 │   │                           │
│         ▼                 │   │                           │
│  planning/result_fusion   │   │                           │
│         │                 │   │                           │
│         ▼                 │   │                           │
│  planning/reranker        │   │                           │
│         │                 │   │                           │
│         ▼                 │   │                           │
│  planning/confidence      │   │                           │
│    (loop if low)           │   │                           │
│         │                 │   │                           │
│         ▼                 │   │                           │
│  expand_chunks()          │   │                           │
│         │                 │   │                           │
└─────────┼─────────────────┘   └───────────┬───────────────┘
          │                                 │
          └────────────┬────────────────────┘
                       ▼
          ┌──────────────────────────┐
          │  answer_with_rag()       │
          │  connect_to_llm.py       │
          │         │                │
          │  DeepSeek API            │
          └──────────────────────────┘
```

## 3. Mode Selection Logic

```python
# In web.py: the client chooses the mode
POST /api/query {"query": "...", "agentic": "full"}     # new pipeline
POST /api/query {"query": "...", "agentic": "legacy"}   # original
POST /api/query {"query": "...", "agentic": "off"}      # simple

# Default: "full" (new agentic pipeline)
```

## 4. Agentic Pipeline Detail (per iteration)

```
Iteration 1:
  analyze_query("how to make middleware")
  → intent=implementation, entities=[], concepts=[recoverPanic, rateLimit, ...]

  generate_retrieval_plan(analysis)
  → [
      SearchStep(hybrid, "how to make middleware", weight=1.0),
      SearchStep(bm25, "middleware", weight=0.7),
      SearchStep(hybrid, "recoverPanic rateLimit enableCORS", weight=0.5),
    ]

  execute_plan(collection, steps)
  → [SearchResult(hybrid, 20 chunks), SearchResult(bm25, 10 chunks), ...]

  fuse_results(results, method="weighted_rrf")
  → 45 unique chunks (deduplicated by content hash)

  rerank_intelligently(chunks, query, analysis)
  → 45 chunks reordered by 6-signal scoring

  compute_confidence(chunks, analysis, iteration=1)
  → ConfidenceReport(overall=0.72, should_retry=False)
  → STOP (confidence > 0.70)

Iteration 2 (if needed):
  _adjust_plan_for_retry(plan, confidence)
  → Adds code_identifier search step, increases top_k on hybrid
  → Same pipeline runs again, results merged
```

## 5. Component Responsibilities

| Component           | Input                     | Output                                       | Calls Weaviate? |
| ------------------- | ------------------------- | -------------------------------------------- | --------------- |
| `query_analyzer`    | Raw query string          | `QueryAnalysis` (intent, entities, concepts) | No              |
| `retrieval_planner` | `QueryAnalysis`           | `RetrievalPlan` (ordered SearchSteps)        | No              |
| `search_strategies` | `RetrievalPlan`           | `List[SearchResult]`                         | **Yes**         |
| `result_fusion`     | `List[SearchResult]`      | `List[Dict]` (merged chunks)                 | No              |
| `reranker`          | chunks + query + analysis | `List[Dict]` (reranked chunks)               | No              |
| `confidence`        | chunks + analysis         | `ConfidenceReport` (retry decision)          | No              |

## 6. Weaviate Calls Per Strategy

| Strategy          | Weaviate Query                                                                      | Properties Used                  |
| ----------------- | ----------------------------------------------------------------------------------- | -------------------------------- |
| `hybrid`          | `collection.query.hybrid(alpha=0.5)`                                                | `chunk` (vectorized)             |
| `bm25`            | `collection.query.bm25()`                                                           | all text properties              |
| `vector`          | `collection.query.hybrid(alpha=1.0)`                                                | `chunk` (vectorized)             |
| `metadata_filter` | `collection.query.hybrid(filters=...)`                                              | `content_type`, `chapter_number` |
| `chapter`         | `collection.query.hybrid(filters=Filter.by_property("chapter_number").equal(...))`  | `chapter_number`                 |
| `code_identifier` | `collection.query.hybrid(alpha=0.2, rerank on code_identifiers)`                    | `code_identifiers`               |
| `filename`        | `collection.query.fetch_objects(filters=Filter.by_property("file_path").like(...))` | `file_path`                      |
| `symbol`          | `collection.query.hybrid(filters=Filter.by_property("content_type").equal("code"))` | `content_type`                   |

## 7. Key Design Decisions

### 7.1 Why heuristic analysis first, not LLM?

- Phase 1 uses keyword heuristics for instant intent classification
- Phase 2 can add LLM-driven analysis for complex queries
- Heuristic path has zero latency and zero cost
- LLM path is opt-in per query complexity

### 7.2 Why 6 reranking signals?

- Single cross-encoder score is prone to embedding bias
- Metadata signals capture structural relevance (is this the right file? the right chapter?)
- Entity overlap captures precision (does this chunk actually contain what was asked?)
- Intent matching captures type relevance (code for implementation, explanation for architecture)
- Topic matching captures concept coverage
- Each signal is independently verifiable and weighted

### 7.3 Why weighted RRF for fusion?

- RRF handles heterogeneous score distributions naturally
- Different strategies produce scores in different ranges (BM25 vs vector vs metadata)
- Ranks are always comparable
- Weights encode strategy importance (code_identifier > hybrid > concept)

### 7.4 Why max 3 iterations?

- Each iteration costs multiple Weaviate calls
- Diminishing returns after 2 iterations in practice
- 3rd iteration only triggers when confidence < 0.40 (very rare)
- Cap prevents infinite loops on impossible queries

## 8. Files Modified (Phase 2)

| File           | Change                                                                     | Lines Affected |
| -------------- | -------------------------------------------------------------------------- | -------------- |
| `retriever.py` | Added agentic imports + `_agentic_retrieve()` + `_adjust_plan_for_retry()` | +120 lines     |
| `retriever.py` | Updated `retrieve_chunks()` signature to accept `agentic` mode             | ~100 lines     |
| `retriever.py` | Updated `answer_query()` to pass `agentic` through                         | ~5 lines       |
| `main.py`      | Updated `get_answer()` and `get_chunks()` signatures                       | ~20 lines      |
| `web.py`       | Updated `/api/query` to read and pass `agentic` field                      | ~4 lines       |

## 9. Files NOT Modified

- `search.py` — unchanged (legacy path still uses it)
- `connect_to_llm.py` — unchanged (always used for final generation)
- `indexer.py` — unchanged
- `chunking.py` — unchanged
- `config.py` — unchanged
- `reranker_server.py` — unchanged

## 10. Backward Compatibility

All existing tests and scripts continue to work:

```python
# These all still work exactly as before:
main.get_answer(query, top_k=60, session_id="default")       # now uses agentic="full" by default
main.get_answer(query, top_k=60, session_id="default", agentic="legacy")  # explicit legacy
app/web.py  →  POST /api/query {"query": "..."}               # defaults to "full"
app/web.py  →  POST /api/query {"query": "...", "agentic": "legacy"}     # explicit legacy
```
