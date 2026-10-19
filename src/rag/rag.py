from rag.retrieval.retriever import hybrid_retrieve
from rag.context.builder import build_context
from rag.generation.generator import generate_answer
from rag.context.source import results_to_sources
from .response import RAGResponse
from rag.context.language import detect_requested_language
from rag.context.selector import select_context_results
from openai import OpenAIError
from rag.exceptions import RAGServiceError,RAGConfigurationError
from qdrant_client.http.exceptions import (
    UnexpectedResponse,
    ResponseHandlingException,
)

def answer_query(query: str) -> RAGResponse :
    try:
        results = hybrid_retrieve(query)
        explicit_language = detect_requested_language(query)
        selected_results = select_context_results(results,explicit_language)
        context = build_context(selected_results)
        answer = generate_answer(query=query,context=context)
        sources = results_to_sources(selected_results)
        
        return RAGResponse(
            answer=answer,
            sources=sources
        )
    except AuthedicationError as err:
        raise RAGConfigurationError("OpenAI authentication failed") from err

    except OpenAIError as err:
        raise RAGServiceError("OpenAI service failed") from err
        
    except (UnexpectedResponse, ResponseHandlingException) as err:
        raise RAGServiceError("Qdrant service failed") from err

if __name__ == "__main__":
    answer = answer_query("How should I set my OpenAI API key in the terminal?")
    print(answer)