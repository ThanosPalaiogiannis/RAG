from qdrant_client.models import ScoredPoint
from rag.embeddings.embed import embed_texts
from rag.vector_store.qdrant_store import (
    get_qdrant_client,
    search_points,
)
from rag.context.builder import build_context

def retrieve(query: str, limit: int = 5) -> list[ScoredPoint]:
    query_embedding = embed_texts([query])[0]
    client = get_qdrant_client()
    results = search_points(
        client=client,
        query_vector=query_embedding,
        limit=limit,
    )
    return results

if __name__ == "__main__":
    results = retrieve("How do I stream responses from the OpenAI API?")
    context = build_context(results)
    print(context)