from rag.ingestion.crawl import crawl
from rag.chunking.chunk import Chunk, chunk_document


def build_chunk_corpus(start_url: str, max_pages: int = 10) -> list[Chunk]:
    documents = crawl(start_url, max_pages)
    all_chunks = []

    for doc in documents:
        chunks = chunk_document(doc)
        all_chunks.extend(chunks)

    return all_chunks


if __name__ == "__main__":
    chunks = build_chunk_corpus(
        "https://developers.openai.com/api/docs",
        max_pages=10,
    )

    code_chunks = [c for c in chunks if c.language is not None]
    prose_chunks = [c for c in chunks if c.language is None]

    print(
        f"Total chunks: {len(chunks)} "
        f"(code={len(code_chunks)}, prose={len(prose_chunks)})"
    )