# Agentic RAG Documentation Assistant

An agentic Retrieval-Augmented Generation (RAG) system that doesn't just retrieve and answer in a single pass — it grades its own retrieved context, reformulates its query and retries when results fall short, and traces its own decision-making end-to-end.

Built to answer questions about RAG and Agentic RAG itself, using a knowledge base of 4 research survey papers — a deliberately self-referential choice.

## Why this project

I work in Application Observability at TCS, where the core skill is tracing *why* a complex system behaves the way it does, not just whether it produces the right output. This project applies that same instinct to an AI system: instead of treating a RAG pipeline as a black box, every stage — retrieval, relevance grading, query reformulation, generation — is observable, debuggable, and was iteratively improved based on real, measured failures.

## Architecture

    User question
         |
         v
     [Retrieve] -- ChromaDB vector search (cosine similarity) over embedded child chunks
         |
         v
     [Grade] -- LLM judges each retrieved chunk for genuine relevance (not just topic overlap)
         |
         +-- Enough relevant chunks? --Yes--> [Generate] --> Answer
         |
         No
         |
         v
     [Reformulate query] --> back to [Retrieve] (up to 2 retries)

Retrieval uses **Parent-Document Retrieval**: small chunks are embedded for accurate search, but the full parent passage is passed to grading/generation — this avoids losing information cut off at chunk boundaries (see *Key Debugging Findings* below).

## Tech Stack

- **Orchestration:** LangGraph (agent state machine, conditional retry loop)
- **Retrieval pipeline:** LangChain (text splitting)
- **Vector store:** ChromaDB (cosine distance, matched to the embedding model's training)
- **Embeddings:** sentence-transformers (`all-MiniLM-L6-v2`), run on CPU
- **LLM:** Llama 3.1 8B, run locally via Ollama (no external API calls or costs)
- **Observability:** Langfuse (full pipeline tracing — every node's inputs, outputs, and latency)
- **Frontend:** Streamlit

## Key Debugging Findings

Real issues found and fixed during development, using Langfuse tracing and direct output inspection:

- **Grading was the actual latency bottleneck** (not generation) — tracing showed relevance grading, which makes one LLM call per retrieved chunk, accounted for most of the pipeline's runtime.
- **Query reformulation hallucinated an acronym expansion** — the LLM invented an incorrect expansion of "RAG" while rewriting a failed query. Fixed by explicitly constraining the reformulation prompt.
- **Generation hallucinated a fake section number and quote** when given weak context, instead of admitting insufficient information — fixed with a stricter, more explicit grounding instruction.
- **Chunk-boundary information loss** — a retrieved chunk stated "there are three components" but was cut off before listing them, since the full explanation spanned a chunk boundary. Fixed by implementing Parent-Document Retrieval.

## Setup

    python -m venv venv
    venv\Scripts\activate
    pip install -r requirements.txt

    ollama pull llama3.1:8b

    python src/ingest.py        # load, chunk, and clean the source PDFs
    python src/vectorstore.py   # embed chunks and build the ChromaDB store
    streamlit run src/app.py    # launch the chat interface

## Possible Next Steps

- HyDE (Hypothetical Document Embeddings) for improved retrieval matching
- Dedicated reranking step (e.g. cross-encoder) as an alternative to LLM-based grading
- A second agent tool (e.g. live web search) for genuine multi-tool routing decisions