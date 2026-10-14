# RAG for OpenAI API Documentation

A Retrieval-Augmented Generation (RAG) system built in Python using the OpenAI API documentation as its knowledge base.

The project is designed as an end-to-end RAG pipeline, with emphasis on retrieval quality, source grounding, reproducible evaluation, and the engineering trade-offs behind production-oriented RAG systems.

The system crawls the OpenAI API documentation, converts it into structured chunks, indexes both dense and sparse representations in Qdrant, retrieves relevant context using hybrid search, and generates answers with explicit source metadata.

The project is still under active development.

## Architecture

```text
OpenAI API Docs
        ↓
Full-site Crawling
        ↓
HTML Parsing / Cleanup
        ↓
Document Deduplication
        ↓
Structure-Aware Chunking
        ↓
Chunk Deduplication
        ↓
Dense Embeddings + Sparse BM25 Vectors
        ↓
Qdrant
        ↓
Hybrid Retrieval (Dense + Sparse)
        ↓
Reciprocal Rank Fusion (RRF)
        ↓
Language-Aware Context Selection
        ↓
LLM Generation
        ↓
Structured Answer + Sources
```

The crawler is restricted to:

```text
https://developers.openai.com/api/docs
```

and continues until there are no more reachable documentation URLs rather than stopping at a fixed page limit.

The current indexed corpus contains:

```text
6,349 unique chunks
├── 2,733 code chunks
└── 3,616 prose chunks
```

## What is implemented

- Full crawling of the OpenAI `/api/docs` documentation
- URL filtering and crawl boundary enforcement
- Redirect handling
- HTML parsing and cleanup with BeautifulSoup + lxml
- Document-level duplicate detection using content fingerprints
- Structure-aware chunking
- Exact chunk deduplication
- Separate handling of prose and code blocks
- Metadata for source URL, section and programming language
- Batched OpenAI dense embeddings
- Sparse BM25 vectors with FastEmbed
- Qdrant storage with named dense and sparse vectors
- Dense semantic retrieval
- Sparse lexical retrieval
- Hybrid dense + sparse retrieval
- Reciprocal Rank Fusion (RRF)
- Weighted hybrid retrieval experiments
- Cross-encoder reranking experiments
- Language-aware context selection
- Structured RAG responses with source metadata
- Source deduplication
- Retrieval evaluation using Recall@K and MRR
- Automated tests with pytest

## Retrieval

The production retrieval strategy is currently **Hybrid 1:1 using Reciprocal Rank Fusion**.

Dense retrieval captures semantic similarity, while sparse BM25 retrieval is useful for exact API terminology, parameter names, and documentation-specific vocabulary.

The two result sets are combined using RRF rather than directly combining their raw scores, since dense cosine similarity and sparse retrieval scores are not directly comparable.

```text
Query
 ├── Dense retrieval
 └── Sparse BM25 retrieval
          ↓
      RRF fusion
          ↓
       Top-K results
```

### Why hybrid retrieval?

The project originally used dense retrieval as the default.

After expanding the corpus to the full OpenAI API documentation, I evaluated several retrieval strategies on a manually defined ground-truth evaluation set.

Current results:

```text
Retriever           Recall@5   MRR
----------------------------------
Dense                 0.75     0.57
Sparse                0.75     0.57
Hybrid 1:1            0.92     0.61
Hybrid 2:1            0.92     0.60
Dense + Reranker      0.67     0.40
```

Hybrid 1:1 was selected because it achieved the highest Recall@5 and slightly better MRR without introducing an additional manually tuned weighting parameter.

The evaluation set is still small, so these results are treated as engineering evidence for the current corpus rather than as a claim of general retrieval accuracy.

## Reranking experiment

I also evaluated a local cross-encoder reranker using FastEmbed.

The reranker did not improve retrieval quality on the current evaluation set and introduced additional latency and memory cost.

For that reason, reranking is currently kept as an experiment rather than part of the default retrieval pipeline.

This was an intentional engineering decision: additional RAG components are only kept when they provide measurable value.

## Language-aware context selection

Documentation often contains the same example in several programming languages.

If the user explicitly asks for a language such as JavaScript or Python, the system prefers chunks written in that language.

For example:

```text
How do I stream responses from the OpenAI API in JavaScript?
```

The pipeline detects the requested language and prioritizes JavaScript code while still preserving relevant prose.

If no language is explicitly requested, Python is currently used as a soft preference. If Python is unavailable for a relevant section, the system falls back to another available language rather than removing the information entirely.

## Structured responses

The final result is represented separately from the retrieved sources:

```text
RAGResponse
├── answer
└── sources
    └── RetrievedSource
        ├── url
        ├── section
        └── language
```

Source metadata comes directly from retrieved documents rather than asking the LLM to generate or reconstruct citations.

This keeps retrieval provenance separate from generated text.

## Retrieval evaluation

Retrieval changes are evaluated programmatically rather than only through manual inspection.

The evaluation supports multiple acceptable targets for each query.

A target can constrain:

```text
URL
section
programming language
```

For example, a broad conceptual query may accept any relevant chunk from a specific guide, while a Python-specific query can require both the correct section and `language="python"`.

The current metrics are:

- `Recall@5`
- Mean Reciprocal Rank (`MRR`)

The evaluation set currently contains 12 queries covering:

- semantic paraphrases
- exact API terminology
- code and programming-language requests
- API parameters
- tool usage
- conceptual documentation questions
- deliberately difficult retrieval cases

## Stack

- Python 3.12
- OpenAI API
- `text-embedding-3-small`
- Qdrant
- Docker
- FastEmbed
- BM25 sparse retrieval
- httpx
- BeautifulSoup
- lxml
- tiktoken
- pytest
- uv

## Project structure

```text
src/rag/
├── chunking/
├── context/
├── embeddings/
├── evaluation/
├── generation/
├── ingestion/
├── retrieval/
├── vector_store/
├── pipeline.py
├── rag.py
└── response.py
```

## Running locally

Install dependencies:

```bash
uv sync
```

Set the OpenAI API key in PowerShell:

```powershell
$env:OPENAI_API_KEY="your_api_key"
```

Start Qdrant:

```bash
docker start qdrant
```

### Build or rebuild the documentation index

```bash
uv run python -m rag.pipeline
```

This performs the full ingestion pipeline and rebuilds the Qdrant collection.

### Run the RAG system

```bash
uv run python -m rag.rag
```

### Run retrieval evaluation

```bash
uv run python -m rag.evaluation.retrieval_eval
```

### Run tests

```bash
uv run pytest
```

Current test suite:

```text
23 passed
```

## Current limitations

The retrieval evaluation set is intentionally small and should be expanded before drawing broader conclusions about retrieval quality.

The current system also assumes a locally available Qdrant instance and requires the documentation index to be built separately from query-time execution.

## Next steps

Planned work includes:

- FastAPI interface
- API request and response validation
- configuration management
- structured logging
- improved error handling
- larger retrieval evaluation set
- latency and cost measurements
- application containerization
- deployment

After the implementation is complete, I also plan to review the system end-to-end and document the major architecture decisions, failure modes, and retrieval quality / latency trade-offs.