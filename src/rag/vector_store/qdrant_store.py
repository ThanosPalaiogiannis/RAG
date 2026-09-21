from qdrant_client import QdrantClient
from qdrant_client import models
from rag.embeddings.embed import EmbeddedChunk

def get_qdrant_client() -> QdrantClient:
    client = QdrantClient(url="http://localhost:6333")
    return client

def create_collection(client: QdrantClient) -> None:
    client.create_collection(
        collection_name = "openai_docs",
        vectors_config = models.VectorParams(
            size = 1536,
            distance = models.Distance.COSINE,
        ),
    )
    
def embedded_chunk_to_point(embedded_chunk: EmbeddedChunk,point_id: int)-> models.PointStruct:
    return models.PointStruct(
        id=point_id,
        vector=embedded_chunk.embedding,
        payload={
            "text": embedded_chunk.chunk.text,
            "source_url": embedded_chunk.chunk.source_url,
            "section": embedded_chunk.chunk.section,
            "language": embedded_chunk.chunk.language,
        },
    )
    
def embedded_chunks_to_points(embedded_chunks: list[EmbeddedChunk]) -> list[models.PointStruct]:
    points = []
    for point_id, embedded_chunk in enumerate(embedded_chunks):
       point=embedded_chunk_to_point(embedded_chunk=embedded_chunk,point_id=point_id) 
       points.append(point)
    return points

def upsert_points(client: QdrantClient, points: list[models.PointStruct]) -> None:
    client.upsert(
        collection_name="openai_docs",
        points=points,
        wait=True,
    )

def search_points(client: QdrantClient,query_vector: list[float],limit: int = 5) -> list[models.ScoredPoint]:
    result = client.query_points(
        collection_name="openai_docs",
        query=query_vector,
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )

    return result.points

def delete_points(client: QdrantClient,point_ids: list[int]) -> None:
    client.delete(
        collection_name="openai_docs",
        points_selector=models.PointIdsList(
            points=point_ids,
        ),
        wait=True,
    )

def reset_collection(client: QdrantClient) -> None:
    if client.collection_exists("openai_docs"):
        client.delete_collection("openai_docs")

    create_collection(client)