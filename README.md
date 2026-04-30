# Retrieval-Augmented Generation (RAG)

In this project, you will build practical knowledge around RAG frameworks and implement a working demo system.

## Prerequisites

- Conduct your own research and decide which tools and technologies you want to use.
- If you require API keys (especially for paid services), let me know in advance.
- If you need learning materials (e.g., paid courses, books), feel free to request them.

## Expected Outcomes

- A working example of a RAG system that can operate on offline documents.
- An architecture diagram describing the system components and how they interact.
- A list of all tools used in the system (relevant to RAG), including references to their official websites.
- Documentation of the vector database used, including justification for the choice.
- Documentation of the LLM(s) used, including justification for the choice.
- Documentation of limitations.

## Guidelines

### Technology Choices

- The preferred language is **Python**, as many RAG-related libraries are optimized for it.  
  If you choose a different language, document the reasoning clearly.

- The preferred database is **PostgreSQL with a vector extension (e.g., pgvector)**.  
  Reason: PostgreSQL is a mature, widely adopted system with a strong ecosystem.

- If you choose a different vector database (e.g., for easier local development), that is acceptable — but you must document:
  - Why you chose it
  - Trade-offs compared to PostgreSQL

### Documentation

- Add all documentation as Markdown files under the `docs/` directory.
- Documentation should be:
  - Brief
  - Precise
  - Technically focused

The goal is not to write long explanations, but to create a clear technical reference of:

- What was built
- How it works
- Why specific decisions were made

Start with minimal documentation — during review, I will point out any missing details.
