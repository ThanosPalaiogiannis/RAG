from fastembed import SparseTextEmbedding
from qdrant_client import models


SPARSE_MODEL = "Qdrant/bm25"

sparse_model = SparseTextEmbedding(
    model_name=SPARSE_MODEL
)


def embed_sparse_text(text: str) -> models.SparseVector:
    embedding = next(sparse_model.embed([text]))

    return models.SparseVector(
        indices=embedding.indices.tolist(),
        values=embedding.values.tolist(),
    )

def embed_sparse_query(query: str) -> models.SparseVector:
    embedding = next(sparse_model.embed([query]))
    return models.SparseVector(
        indices=embedding.indices.tolist(),
        values=embedding.values.tolist(),
    )

def embed_sparse_texts(texts: list[str]) -> list[models.SparseVector]:
    embeddings = sparse_model.embed(texts)

    sparse_vectors = []

    for embedding in embeddings:

        sparse_vector = models.SparseVector(
        indices=embedding.indices.tolist(),
        values=embedding.values.tolist(),
        )

        sparse_vectors.append(sparse_vector)

    return sparse_vectors
