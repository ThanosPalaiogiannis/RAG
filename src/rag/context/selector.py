from qdrant_client.models import ScoredPoint


def select_context_results(results: list[ScoredPoint],explicit_language: str | None,default_language: str = "python",) -> list[ScoredPoint]:
    results_by_section = {}

    for result in results:
        source_url = result.payload.get("source_url")
        section = result.payload.get("section")

        group_key = (source_url, section)

        if group_key not in results_by_section:
            results_by_section[group_key] = []

        results_by_section[group_key].append(result)

    selected_results = []
    preferred_language = explicit_language or default_language

    selected_language_by_section = {}
    for group_key, group_results in results_by_section.items():
            chosen_language = None
            for result in group_results:
                language = result.payload.get("language")
                if language == preferred_language:
                    chosen_language = language
                    break
            if chosen_language is None:
                for result in group_results:
                    language = result.payload.get("language")
                    if language is not None:
                        chosen_language = language
                        break
            selected_language_by_section[group_key]=chosen_language       
    for result in results:
        source_url = result.payload.get("source_url")
        section = result.payload.get("section")
        language = result.payload.get("language")
        group_key = (source_url, section)  
        if language is None:
            selected_results.append(result)
        elif language == selected_language_by_section[group_key] :
            selected_results.append(result)
    
    return selected_results
                       
                    
        