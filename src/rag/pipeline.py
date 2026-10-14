from rag.ingestion.crawl import crawl
from rag.chunking.chunk import Chunk, chunk_document
from rag.embeddings.embed import embed_chunks
from rag.vector_store.qdrant_store import (
    get_qdrant_client,
    embedded_chunks_to_points,
    upsert_points,
    reset_collection,
)


def build_chunk_corpus(start_url: str, max_pages: int | None = None,) -> list[Chunk]:
    documents = crawl(start_url, max_pages)
    all_chunks = []
    seen_chunk_keys = set()
    for doc in documents:
        chunks = chunk_document(doc)
        for chunk in chunks:
            chunk_key = (
            chunk.section,
            chunk.language,
            chunk.text,
            )
            if chunk_key not in seen_chunk_keys:
                seen_chunk_keys.add(chunk_key)
                all_chunks.append(chunk)
        print(f"{doc.url} -> {len(chunks)} chunks")
    return all_chunks


if __name__ == "__main__":
    chunks = build_chunk_corpus("https://developers.openai.com/api/docs")
    
    code_chunks = [c for c in chunks if c.language is not None]
    prose_chunks = [c for c in chunks if c.language is None]

    print(
        f"Total chunks: {len(chunks)} "
        f"(code={len(code_chunks)}, prose={len(prose_chunks)})"
    )

    embedded_chunks = embed_chunks(chunks)
    points = embedded_chunks_to_points(embedded_chunks)

    client = get_qdrant_client()
    reset_collection(client)

    upsert_points(client, points)

    count = client.count(
        collection_name="openai_docs",
        exact=True,
    )
    print(f"Qdrant points: {count.count}")
    
