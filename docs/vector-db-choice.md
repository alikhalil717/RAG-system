# Vector Database Choice

This project uses Weaviate in embedded mode instead of PostgreSQL with pgvector.

## Why Weaviate

- It supports hybrid retrieval out of the box, combining vector search with BM25 keyword search.
- It integrates cleanly with local embedding and reranker modules.
- Embedded mode keeps the demo self-contained and removes the need to manage a separate database service.
- The Python client matches the current codebase well and keeps indexing and querying simple.

## Trade-offs vs PostgreSQL + pgvector

- Weaviate is less standard than PostgreSQL and pgvector for teams that already run Postgres.
- PostgreSQL would fit better if the system needed strong relational storage or existing SQL workflows.
- Weaviate is the better fit here because the project prioritizes a fast local demo with hybrid search and vector-native APIs.

## Current setup

- The collection is created in embedded Weaviate with local persistence under `.collections`.
- Chunk text is stored as the document payload and the collection is queried with hybrid search plus reranking.
