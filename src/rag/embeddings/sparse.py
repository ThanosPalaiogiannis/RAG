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
