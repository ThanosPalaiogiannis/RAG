from qdrant_client.models import ScoredPoint


def build_context(results: list[ScoredPoint]) -> str:
    context_parts = []
    for i, result in enumerate(results, start=1):
        payload = result.payload
        source_part = (
        f"[Source {i}]\n"
        f"Section: {payload.get('section')}\n"
        f"Language: {payload.get('language')}\n"
        f"URL: {payload.get('source_url')}\n"
        f"Content:\n{payload.get('text')}"
        )
        context_parts.append(source_part)
    return "\n\n".join(context_parts)