from qdrant_client.models import ScoredPoint
from rag.embeddings.embed import embed_texts
from rag.vector_store.qdrant_store import (
    get_qdrant_client,
    search_points,
    search_sparse_points,
    search_hybrid_points,
    search_weighted_hybrid_points,
)
from rag.context.builder import build_context
from rag.embeddings.sparse import embed_sparse_query
from rag.reranking.reranker import rerank_results

def retrieve(query: str, limit: int = 5) -> list[ScoredPoint]:
    dense_vector = embed_texts([query])[0]
    client = get_qdrant_client()
    results = search_points(
        client=client,
        query_vector=dense_vector,
        limit=limit,
    )
    return results

def sparse_retrieve(query: str, limit: int = 5) -> list[ScoredPoint]:
    sparse_vector = embed_sparse_query(query)
    client = get_qdrant_client()
    results = search_sparse_points(
        client=client,
        query_vector=sparse_vector,
        limit=limit,
    )
    return results

def hybrid_retrieve(query: str, limit: int = 5) -> list[ScoredPoint]:
    dense_vector = embed_texts([query])[0]
    sparse_vector = embed_sparse_query(query)
    client = get_qdrant_client()
    results = search_hybrid_points(
        client=client,
        dense_vector=dense_vector,
        sparse_vector=sparse_vector,
        limit=limit,
    )
    return results

def weighted_hybrid_retrieve(query: str, limit: int = 5,) -> list[ScoredPoint]:
    dense_vector = embed_texts([query])[0]
    sparse_vector = embed_sparse_query(query)
    client = get_qdrant_client()
    results = search_weighted_hybrid_points(
        client=client,
        dense_vector=dense_vector,
        sparse_vector=sparse_vector,
        limit=limit,
    )
    return results

def reranked_retrieve( query: str, limit: int = 5,)-> list[ScoredPoint]:
    candidates = retrieve(query, limit=15)
    return rerank_results(query,candidates,top_k=limit)

if __name__ == "__main__":
    query = "What does max_output_tokens control in the Responses API?"

    results = retrieve(query, limit=20)

    for rank, result in enumerate(results, start=1):
        payload = result.payload or {}

        print(f"\n--- Rank {rank} ---")
        print(f"Score: {result.score}")
        print(f"URL: {payload.get('source_url')}")
        print(f"Section: {payload.get('section')}")
        print(f"Language: {payload.get('language')}")

        text = payload.get("text", "").replace("\n", " ")
        print(f"Text: {text[:300]}")