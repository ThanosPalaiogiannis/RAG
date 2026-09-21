from rag.ingestion.crawl import crawl
from rag.chunking.chunk import Chunk, chunk_document
from rag.embeddings.embed import embed_chunks
from rag.vector_store.qdrant_store import (
    get_qdrant_client,
    embedded_chunks_to_points,
    upsert_points,
    reset_collection,
)


def build_chunk_corpus(start_url: str, max_pages: int = 10) -> list[Chunk]:
    documents = crawl(start_url, max_pages)
    all_chunks = []

    for doc in documents:
        chunks = chunk_document(doc)
        all_chunks.extend(chunks)

    return all_chunks


if __name__ == "__main__":
    chunks = build_chunk_corpus(
    "https://developers.openai.com/api/docs/quickstart",
    max_pages=1,
    )
    
    code_chunks = [c for c in chunks if c.language is not None]
    prose_chunks = [c for c in chunks if c.language is None]

    print(
        f"Total chunks: {len(chunks)} "
        f"(code={len(code_chunks)}, prose={len(prose_chunks)})"
    )
    embedded_chunks = embed_chunks(chunks)

    client = get_qdrant_client()
    
    reset_collection(client)
    
    points = embedded_chunks_to_points(embedded_chunks)

    upsert_points(client, points)

    count = client.count(
        collection_name="openai_docs",
        exact=True,
    )

    print(f"Qdrant points: {count.count}")
    
