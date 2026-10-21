from qdrant_client import QdrantClient
from qdrant_client import models
from rag.embeddings.embed import EmbeddedChunk
from rag.embeddings.sparse import embed_sparse_texts
from rag.config import QDRANT_URL, QDRANT_COLLECTION

def get_qdrant_client() -> QdrantClient:
    client = QdrantClient(url=QDRANT_URL)
    return client

def create_collection(client: QdrantClient) -> None:
    client.create_collection(
        collection_name = QDRANT_COLLECTION,
        vectors_config={
            "dense": models.VectorParams(
                size = 1536,
                distance = models.Distance.COSINE,
            )
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(
                modifier = models.Modifier.IDF,
            )
        },
    )

def embedded_chunk_to_point(embedded_chunk: EmbeddedChunk, sparse_vector: models.SparseVector,point_id: int,) -> models.PointStruct:
    return models.PointStruct(
        id=point_id,
        vector={
            "dense": embedded_chunk.embedding,
            "sparse": sparse_vector,
        },
        payload={
            "text": embedded_chunk.chunk.text,
            "source_url": embedded_chunk.chunk.source_url,
            "section": embedded_chunk.chunk.section,
            "language": embedded_chunk.chunk.language,
        },
    )

def embedded_chunks_to_points(embedded_chunks: list[EmbeddedChunk]) -> list[models.PointStruct]:
    texts = [embedded_chunk.chunk.text for embedded_chunk in embedded_chunks]
    sparse_vectors = embed_sparse_texts(texts)
    points = []
    for point_id, (embedded_chunk, sparse_vector) in enumerate(zip(embedded_chunks, sparse_vectors)):
        point = embedded_chunk_to_point(
            embedded_chunk=embedded_chunk,
            sparse_vector=sparse_vector,
            point_id=point_id,
        )
        points.append(point)

    return points

def upsert_points(client: QdrantClient, points: list[models.PointStruct], batch_size: int = 500,) -> None:
    for start in range(0, len(points), batch_size):
        batch = points[start:start + batch_size]
        client.upsert(
        collection_name=QDRANT_COLLECTION,
        points=batch,
        wait=True,
        )

def search_points(client: QdrantClient,query_vector: list[float],limit: int = 5) -> list[models.ScoredPoint]:
    result = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_vector,
        limit=limit,
        using="dense",
        with_payload=True,
        with_vectors=False,
    )

    return result.points

def search_sparse_points(client: QdrantClient,query_vector: models.SparseVector,limit: int = 5) -> list[models.ScoredPoint]:
    result = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query = query_vector,
        using = "sparse",
        limit = limit,
        with_payload=True,
        with_vectors=False,
    )
    return result.points

def search_hybrid_points(client: QdrantClient,dense_vector: list[float],sparse_vector: models.SparseVector,limit: int = 5,prefetch_limit: int = 15,) -> list[models.ScoredPoint]:
    results = client.query_points(
        collection_name=QDRANT_COLLECTION,
        prefetch=[
            models.Prefetch(
                query=dense_vector,
                using="dense",
                limit=prefetch_limit,
            ),

            models.Prefetch(
                query=sparse_vector,
                using="sparse",
                limit=prefetch_limit,
            ),
        ],
        query=models.RrfQuery(
            rrf=models.Rrf()
        ),
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )

    return results.points

def search_weighted_hybrid_points(client: QdrantClient,dense_vector: list[float],sparse_vector: models.SparseVector,limit: int = 5,prefetch_limit: int = 15,) -> list[models.ScoredPoint]:
    results = client.query_points(
        collection_name=QDRANT_COLLECTION,
        prefetch=[
            models.Prefetch(
                query=dense_vector,
                using="dense",
                limit=prefetch_limit,
            ),

            models.Prefetch(
                query=sparse_vector,
                using="sparse",
                limit=prefetch_limit,
            ),
        ],
        query=models.RrfQuery(
            rrf=models.Rrf(
                weights=[2.0, 1.0],
            )
        ),
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )

    return results.points

def delete_points(client: QdrantClient,point_ids: list[int]) -> None:
    client.delete(
        collection_name=QDRANT_COLLECTION,
        points_selector=models.PointIdsList(
            points=point_ids,
        ),
        wait=True,
    )

def reset_collection(client: QdrantClient) -> None:
    if client.collection_exists(QDRANT_COLLECTION):
        client.delete_collection(QDRANT_COLLECTION)

    create_collection(client)