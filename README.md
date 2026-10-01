# RAG for OpenAI Docs

I'm building a Retrieval-Augmented Generation (RAG) system in Python using the OpenAI API documentation as its knowledge base.

The goal is to build the system end-to-end and gradually improve the retrieval pipeline while learning the engineering decisions behind a real RAG application.

The project is still under development.

## How it works

```text
OpenAI Docs
    ↓
Crawling / Ingestion
    ↓
HTML Parsing
    ↓
Structure-Aware Chunking
    ↓
Embeddings
    ↓
Qdrant
    ↓
Dense Retrieval
    ↓
Context Selection
    ↓
LLM Generation
    ↓
Answer + Sources
```

The current corpus is based on the OpenAI API Quickstart documentation.

## What I've implemented so far

- Web crawling with `httpx`
- HTML parsing and cleanup with BeautifulSoup + lxml
- Structure-aware chunking
- Separate handling of prose and code blocks
- Metadata for source URL, section and programming language
- OpenAI embeddings
- Qdrant for vector storage and search
- Dense semantic retrieval
- Language-aware context selection
- Structured responses with source metadata
- Source deduplication
- Retrieval evaluation with Recall@K and MRR
- Automated tests with pytest

One issue I wanted to handle was documentation pages that contain the same code example in several programming languages.

If the user explicitly asks for JavaScript, for example, the system prefers JavaScript chunks. If no language is specified, Python is currently the default preference. If Python is not available for that section, the system falls back to another relevant language instead of dropping the information completely.

## Example

Query:

```text
How do I stream responses from the OpenAI API in JavaScript?
```

The system retrieves the relevant documentation, detects the requested language, selects the appropriate context and sends that context to the LLM.

The final result is represented as:

```text
RAGResponse
├── answer
└── sources
    └── RetrievedSource
        ├── url
        ├── section
        └── language
```

This way the generated answer and the retrieved sources remain separate instead of relying on the LLM to generate source information itself.

## Retrieval evaluation

I added a small evaluation set so changes to the retriever can be measured instead of only checking results manually.

At the moment I use:

- `Recall@K`
- Mean Reciprocal Rank (`MRR`)

The first evaluation set contains five questions from the OpenAI Quickstart corpus.

Current baseline:

```text
Recall@5: 1.00
MRR:      1.00
```

The dataset is still very small, so these numbers are only a development baseline and not a claim about general retrieval accuracy.

## Stack

- Python 3.12
- OpenAI API
- Qdrant
- Docker
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

Start the existing Qdrant container:

```bash
docker start qdrant
```

Run the RAG pipeline:

```bash
uv run python -m rag.rag
```

Run retrieval evaluation:

```bash
uv run python -m rag.evaluation.retrieval_eval
```

Run the tests:

```bash
uv run pytest
```

## Next steps

I'm currently planning to add:

- hybrid dense + lexical retrieval
- retrieval fusion
- reranking
- a larger evaluation dataset
- FastAPI
- configuration and logging
- better error handling
- full application containerization
- deployment