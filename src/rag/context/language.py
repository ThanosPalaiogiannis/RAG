def detect_requested_language(query: str) -> str | None:
    query_lower = query.lower()

    if "python" in query_lower:
        return "python"

    if "javascript" in query_lower or "js" in query_lower:
        return "javascript"

    if "bash" in query_lower or "shell" in query_lower:
        return "bash"

    if "java" in query_lower:
        return "java"

    if "c#" in query_lower or "csharp" in query_lower:
        return "csharp"

    if "ruby" in query_lower:
        return "ruby"

    return None