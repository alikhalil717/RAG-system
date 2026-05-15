# Limitations

## 1. Current constraints and design rationale

1. The current system indexes only PDF documents stored in `data/pdfs`, so its scope is intentionally narrow.
2. The implementation currently uses Weaviate in local embedded mode through the Python client.
3. That choice was made to simplify local development, reduce setup overhead, and keep the demo self-contained.
4. It also allowed the retrieval pipeline to be tested without depending on an externally managed database service.
5. The system currently uses CPU execution for the embedding and reranking components.
6. This decision was not made for performance reasons, but because the available GPU on my machine cannot be reliably used with the current PyTorch setup.
7. Using CPU therefore provided a stable and reproducible execution path during development.
8. The retrieval quality is also bounded by the quality of the extracted text and the size of the chunk windows.
9. As a result, the system is better suited to short factual questions than to long-form analytical reasoning.
10. In addition, the fallback hash-based embedding mechanism is a robustness measure rather than a high-quality semantic representation.
11. It ensures that the pipeline can continue to operate when the local embedding model is unavailable, but it is not equivalent to a learned embedding model.

## 2. Planned improvements

1. In the next development phase, I plan to add a configuration option that allows the user to choose between Weaviate and pgvector.
- This change will make the vector store a selectable architectural decision rather than a fixed implementation choice.

2. I also plan to formalize the local Weaviate integration so that the client operates explicitly with the embedded Weaviate instance.
- That refinement will make the deployment model clearer and reduce ambiguity between local and external database usage.

3. A further planned improvement is to expose the execution device in the configuration file so the user can choose CPU or GPU when the environment supports it.
- This will make the system more portable, while preserving the current CPU-based default for machines where GPU execution is not available or not stable.

4. I also plan to add support for more types of data sources, such as databases and datasets.
5. I plan to enhance the system's ability to handle and process different data formats and structures.
6. Additionally, I aim to improve the user interface and provide better documentation and examples for easier adoption.

7. Finally, I plan to add more comprehensive error handling and logging throughout the pipeline to improve robustness and debuggability.
- This will include better handling of edge cases in document ingestion, embedding, retrieval, and generation, as well as more informative logging of the system's internal state and decisions.
