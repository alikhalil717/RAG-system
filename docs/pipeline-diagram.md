# RAG Pipeline — Full Architecture Diagram

````mermaid
flowchart TB
    subgraph INGEST["📥 INGESTION"]
        direction TB
        A1[("📄 PDFs\n(data/pdfs/)")] --> A2["extract_text_from_pdf()\n+ pattern fallback"]
        A2 --> A3["_clean_pdf_text()\nstrip boilerplate"]
        A3 --> A4["get_all_pdfs()\n→ List of page dicts"]
        A4 --> A5["chunk_text()\nRecursiveCharacterTextSplitter\nCHUNK_SIZE=500, OVERLAP=50"]
        A5 --> A6["_merge_incomplete_code_chunks()\nmerge split [CODE] blocks"]
        A6 --> A7["_mark_latest_file_versions()\ntrack file evolution"]
        A7 --> A8[("🗂️ Chunks\n+ metadata dicts")]
    end

    subgraph INDEX["🔍 INDEXING"]
        direction TB
        B1["create_collection()"] --> B2["create_index()"]
        B2 --> B3["generate_embedding()\nBAAI/bge-base-en-v1.5\nvia reranker_server:5000"]
        B3 --> B4[("Weaviate Embedded\nport 8079 / gRPC 50050\n16 properties")]
    end

    subgraph RUNTIME["⚙️ RUNTIME SETUP"]
        direction TB
        C1["run_runtime()"] --> C2["_start_reranker_server()\nFlask on port 5000\ncross-encoder/ms-marco-MiniLM-L-6-v2"]
        C2 --> C3["_prepare_collection()\nconnect to Weaviate"]
        C3 --> C4["AppRuntime\n{client, collection, reranker_process, ready}"]
    end

    subgraph RETRIEVAL["🎯 RETRIEVAL PIPELINE"]
        direction TB

        subgraph LEGACY["mode: legacy"]
            direction TB
            L1["decompose_query()\nLLM → 3-4 sub-queries"] --> L2["multi_query_search()"]
            L2 --> L3["RRF fusion"]
            L3 --> L4["expand_chunks()\n±N window"]
        end

        subgraph SIMPLE["mode: off"]
            direction TB
            M1["hybrid_search()\nvector + BM25\nα=0.5 + cross-encoder rerank"]
        end

        subgraph AGENTIC["🧠 mode: full (agentic)"]
            direction TB

            D1["🔍 PHASE 1\nQuery Understanding"] --> D2
            D2["analyze_query()\nLLM → QueryAnalysis\n{intent, entities, concepts}"]

            D2 --> D3["📋 PHASE 2\nRetrieval Planning"]
            D3 --> D4["generate_retrieval_plan()\nLLM → RetrievalPlan\nordered SearchSteps × N"]

            D4 --> D5["🔄 PHASE 3\nIterative Search Loop\n(up to 3 iterations)"]
            D5 --> D6

            subgraph ITER["One Iteration"]
                direction TB
                D6["execute_plan()\n11 search strategies"] --> D7
                D7["fuse_results()\nweighted RRF + dedup"] --> D8
                D8["rerank_intelligently()\nper-intent 6-signal weights"] --> D9
                D9["compute_confidence()\n5-factor scoring"]
            end

            D9 --> D10{"should_continue?\nconf ≥ 0.7?"}
            D10 -->|"yes — continue"| D11["_adjust_plan_for_retry()\nboost missing coverage"]
            D11 --> D6
            D10 -->|"no — done"| D12["PHASE 4\nexpand_chunks()\ncontext window"]
        end

        D12 --> RET["→ List of chunks"]
        L4 --> RET
        M1 --> RET
    end

    subgraph STRATEGIES["🗺️ 11 SEARCH STRATEGIES"]
        direction LR
        S1["hybrid\nvector+BM25"]
        S2["bm25\nkeyword"]
        S3["vector\nsemantic"]
        S4["metadata_filter\nproperty match"]
        S5["chapter\nchapter_number"]
        S6["code_identifier\nBM25 on symbols"]
        S7["filename\nfile_path"]
        S8["symbol\nGo func/type names"]
        S9["multi_query\nRRF fusion"]
        S10["progressive\nnarrow→wide"]
        S11["hierarchical\nchapter→chunk"]
    end

    subgraph GENERATION["💬 ANSWER GENERATION"]
        direction TB
        E1["answer_query()"] --> E2["retrieve_chunks()\n→ contexts"]
        E2 --> E3["answer_with_rag()\nformat_chunks_into_context()\n[CODE] → ```lang fences"]
        E3 --> E4["llm.invoke()\nSystemMessage + history + HumanMessage"]
        E4 --> E5["store in session history\nChatMessageHistory"]
        E5 --> E6["→ Final Answer"]
    end

    subgraph RERANKER["🎚️ RERANKER SIGNALS"]
        direction LR
        R1["semantic\ncross-encoder score\nweight: 0.35"]
        R2["metadata\nfield overlap\nweight: 0.20"]
        R3["entity\nidentifier match\nweight: 0.15"]
        R4["intent\ncontent_type match\nweight: 0.15"]
        R5["topic\nconcept overlap\nweight: 0.10"]
        R6["code_density\ncode-heavy chunks\nweight: 0.05"]
    end

    subgraph INTERFACE["🖥️ USER INTERFACES"]
        direction LR
        U1["web.py\nFlask :8080\nREST API + HTML UI"]
        U2["main.py\nPython API\nget_answer() / get_chunks()"]
        U3["test.ipynb\nJupyter notebook\ninteractive testing"]
    end

    subgraph EXTERNAL["☁️ EXTERNAL SERVICES"]
        direction LR
        X1["DeepSeek API\napi.deepseek.com\ndeepseek-coder\nT=0.4, max_tokens=3000"]
        X2["Embedding Model\nBAAI/bge-base-en-v1.5\nlocal CPU inference"]
        X3["Reranker Model\ncross-encoder/ms-marco-MiniLM-L-6-v2\nlocal CPU inference"]
    end

    %% ── FLOW CONNECTIONS ──────────────────────────────────────────
    A8 --> B1
    B4 --> C3
    C4 --> RET

    RET --> E2
    RET -.-> S1 & S2 & S3 & S4 & S5 & S6 & S7 & S8 & S9 & S10 & S11
    D6 -.-> S1 & S2 & S3 & S4 & S5 & S6 & S7 & S8 & S9 & S10 & S11
    D8 -.-> R1 & R2 & R3 & R4 & R5 & R6

    U1 --> C1
    U2 --> C1
    U3 --> C1

    E4 --> X1
    B3 --> X2
    C2 --> X3
    D2 --> X1
    D4 --> X1
    L1 --> X1

    %% ── STYLING ───────────────────────────────────────────────────
    classDef ingest fill:#e1f5fe,stroke:#0288d1,stroke-width:2px
    classDef index fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    classDef runtime fill:#e8f5e9,stroke:#388e3c,stroke-width:2px
    classDef retrieval fill:#fce4ec,stroke:#c62828,stroke-width:2px
    classDef strategy fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef generation fill:#e0f2f1,stroke:#00695c,stroke-width:2px
    classDef reranker fill:#fff8e1,stroke:#f9a825,stroke-width:2px
    classDef interface fill:#e8eaf6,stroke:#283593,stroke-width:2px
    classDef external fill:#efebe9,stroke:#4e342e,stroke-width:2px

    class INGEST,A1,A2,A3,A4,A5,A6,A7,A8 ingest
    class INDEX,B1,B2,B3,B4 index
    class RUNTIME,C1,C2,C3,C4 runtime
    class RETRIEVAL,D1,D2,D3,D4,D5,D6,D7,D8,D9,D10,D11,D12,RET,L1,L2,L3,L4,M1 retrieval
    class STRATEGIES,S1,S2,S3,S4,S5,S6,S7,S8,S9,S10,S11 strategy
    class GENERATION,E1,E2,E3,E4,E5,E6 generation
    class RERANKER,R1,R2,R3,R4,R5,R6 reranker
    class INTERFACE,U1,U2,U3 interface
    class EXTERNAL,X1,X2,X3 external
````

---

## Pipeline Summary

| Phase              | Module                 | Function                                                                                 |
| ------------------ | ---------------------- | ---------------------------------------------------------------------------------------- |
| **Ingestion**      | `chunking.py`          | PDF → font+pattern code detection → `[CODE]` markers → text split → metadata-rich chunks |
| **Indexing**       | `indexer.py`           | Chunks → BGE embeddings → Weaviate collection (16 properties)                            |
| **Runtime**        | `retriever.py`         | Spawn reranker Flask server → connect to Weaviate embedded                               |
| **Query Analysis** | `query_analyzer.py`    | LLM classifies intent (10 categories) + extracts entities/concepts                       |
| **Planning**       | `retrieval_planner.py` | LLM generates ordered `SearchSteps` per strategy                                         |
| **Search**         | `search_strategies.py` | 11 strategy implementations (hybrid, BM25, vector, code_identifier, etc.)                |
| **Fusion**         | `result_fusion.py`     | Weighted RRF (K=60) + content-hash dedup                                                 |
| **Reranking**      | `reranker.py`          | 6-signal per-intent reranking (semantic, metadata, entity, intent, topic, code_density)  |
| **Confidence**     | `confidence.py`        | 5-factor scoring + iterative retry (max 3 rounds)                                        |
| **Generation**     | `connect_to_llm.py`    | Format chunks (code → markdown fences) → LLM via `SystemMessage`/`HumanMessage`          |
| **Interface**      | `web.py` / `main.py`   | Flask REST API (:8080) + Python API + Jupyter notebook                                   |
