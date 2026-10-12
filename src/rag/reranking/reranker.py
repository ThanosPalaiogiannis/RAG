from fastembed.rerank.cross_encoder import TextCrossEncoder
from qdrant_client.models import ScoredPoint
RERANKER_MODEL = "BAAI/bge-reranker-base"

reranker = TextCrossEncoder(
    model_name=RERANKER_MODEL
)

def rerank_results(query: str, results:list[ScoredPoint], top_k: int=5,)->list[ScoredPoint]:
    documents = []
    for result in results:
        section = result.payload.get("section") or ""
        text = result.payload.get("text", "")
        document = section + "\n\n" + text
        documents.append(document)
    scores = list(reranker.rerank(query,documents))
    result_scores = list(zip(results,scores))
    ranked_results = sorted(
    result_scores,
    key=lambda pair: pair[1],
    reverse=True,
    )
    return [ result for result, _ in ranked_results[:top_k]]

   