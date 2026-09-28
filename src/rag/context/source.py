from dataclasses import dataclass
from qdrant_client.models import ScoredPoint


@dataclass
class RetrievedSource:
    url: str
    section: str | None
    language: str | None
    
def result_to_source(result: ScoredPoint) -> RetrievedSource:
    payload = result.payload

    return RetrievedSource(
        url=payload["source_url"],
        section=payload.get("section"),
        language=payload.get("language"),
    )
    
def results_to_sources(results: list[ScoredPoint]) -> list[RetrievedSource]:
    sources = []
    seen = set()
    for result in results:
        source = result_to_source(result)
        source_key = (
        source.url,
        source.section,
        source.language,
        )

        if source_key in seen:
            continue

        seen.add(source_key)
        sources.append(source)

    return sources