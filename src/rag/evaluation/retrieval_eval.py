from rag.retrieval.retriever import (
    retrieve,
    sparse_retrieve,
    hybrid_retrieve,
    weighted_hybrid_retrieve,
    reranked_retrieve,
)
from qdrant_client.models import ScoredPoint

DEV_CASES = [
    {
        "query": "How do I stream responses from the OpenAI API?",
        "expected_section": "Stream responses and build real-time apps",
    },
    {
        "query": "How do I send an image to the model?",
        "expected_section": "Analyze images and files",
    },
    {
        "query": "How can I give the model access to web search?",
        "expected_section": "Extend the model with tools",
    },
    {
        "query": "How do I make my first API request in Python?",
        "expected_section": "Install the OpenAI SDK and Run an API Call",
    },
    {
        "query": "How should I set my OpenAI API key on my computer?",
        "expected_section": "Create and export an API key",
    },
    {
        "query": "How should I set OPENAI_API_KEY on my computer?",
        "expected_section": "Create and export an API key",
    },
    {
        "query": "How can I receive the response gradually instead of waiting for the full answer?",
        "expected_section": "Stream responses and build real-time apps",
    },
    {
        "query": "Where do I export my API key as an environment variable?",
        "expected_section": "Create and export an API key",
    },
    {
    "query": "Can I see the answer while it is still being generated?",
    "expected_section": "Stream responses and build real-time apps",
    },
    {
        "query": "Can the model look up current information from the internet?",
        "expected_section": "Extend the model with tools",
    },
    {
        "query": "How can I ask the model questions about a picture?",
        "expected_section": "Analyze images and files",
    },
    {
        "query": "I don't want to wait until the entire response is finished. What can I do?",
        "expected_section": "Stream responses and build real-time apps",
    },
    {
    "query": "setx OPENAI_API_KEY",
    "expected_section": "Create and export an API key",
    },
    {
        "query": "stream=True",
        "expected_section": "Stream responses and build real-time apps",
    },
    {
        "query": "web_search",
        "expected_section": "Extend the model with tools",
    },
    {
        "query": "input_image",
        "expected_section": "Analyze images and files",
    },
]

VALIDATION_CASES = [
    {
        "query": "What do I use if I want tokens to arrive as they are generated?",
        "expected_section": "Stream responses and build real-time apps",
    },
    {
        "query": "Can OpenAI search the web for information that changed recently?",
        "expected_section": "Extend the model with tools",
    },
    {
        "query": "How do I include a photo in a request?",
        "expected_section": "Analyze images and files",
    },
    {
        "query": "Where should my API credential be stored before running the SDK?",
        "expected_section": "Create and export an API key",
    },
    {
        "query": "Authorization: Bearer $OPENAI_API_KEY",
        "expected_section": "Create and export an API key",
    },
    {
        "query": "How do I start using the Python SDK for my first request?",
        "expected_section": "Install the OpenAI SDK and Run an API Call",
    },
    {
        "query": "I need live output instead of receiving everything at the end.",
        "expected_section": "Stream responses and build real-time apps",
    },
    {
        "query": "How can the model use an internet search tool?",
        "expected_section": "Extend the model with tools",
    },
]

LARGE_CORPUS_TEST_CASES = [
    {
        "query": "How do I use web search with the Responses API?",
        "expected_targets": [
            {
                "url_contains": "/guides/tools-web-search",
                "section": "Choose an integration",
            },
            {
                "url_contains": "/guides/tools-web-search",
                "section": "Responses API",
            },
            {
                "url_contains": "/guides/tools-web-search",
                "section": "Web search",
            },
        ],
    },
    {
        "query": "How do I create embeddings in Python?",
        "expected_targets": [
            {
                "url_contains": "/guides/embeddings",
                "section": "How to get embeddings",
                "language": "python",
            },
        ],
    },
    {
        "query": "What does previous_response_id do in the Responses API?",
        "expected_targets": [
            {
                "url_contains": "/guides/conversation-state",
                "section": "Passing context from the previous response",
            },
        ],
    },
    {
        "query": "How can I receive tokens while the response is still being generated?",
        "expected_targets": [
            {
                "url_contains": "/guides/streaming-responses",
            },
        ],
    },
    {
        "query": "How do I send an image to the model for analysis?",
        "expected_targets": [
            {
                "url_contains": "/guides/images",
            },
        ],
    },
    {
        "query": "How can I let the model call one of my own functions?",
        "expected_targets": [
            {
                "url_contains": "/guides/function-calling",
            },
        ],
    },
    {
        "query": "How do I keep conversation state between multiple Responses API calls?",
        "expected_targets": [
            {
                "url_contains": "/guides/conversation-state",
            },
        ],
    },
    {
        "query": "What does max_output_tokens control in the Responses API?",
        "expected_targets": [
            {
                "url_contains": "/guides/reasoning",
                "section": "Allocating space for reasoning",
            },
            {
                "url_contains": "/guides/reasoning",
                "section": "Controlling costs",
            },
        ],
    },
    {
        "query": "What should I do when my application hits API rate limits?",
        "expected_targets": [
            {
                "url_contains": "/guides/rate-limits",
            },
        ],
    },
    {
        "query": "How do I search files that I uploaded to a vector store?",
        "expected_targets": [
            {
                "url_contains": "/guides/tools-file-search",
            },
        ],
    },
    {
        "query": "How can I make the model return JSON that follows a specific schema?",
        "expected_targets": [
            {
                "url_contains": "/guides/structured-outputs",
            },
        ],
    },
    {
        "query": "How does prompt caching reduce latency and cost for repeated prompts?",
        "expected_targets": [
            {
                "url_contains": "/guides/prompt-caching",
            },
        ],
    },
]
def evaluate_case(case: dict,retriever, k: int = 5):
    query = case["query"]
    expected_targets = case["expected_targets"]
    results = retriever(query, limit=k)
    rank = None
    for i, result in enumerate(results, start=1):
        if any(matches_target(result, target) for target in expected_targets):
            rank = i
            break
    return rank

def evaluate_retrieval(cases: list[dict],retriever, k: int = 5):
    hits = 0
    reciprocal_rank_sum = 0.0
    ranks = []
    for case in cases:
        rank = evaluate_case(case,retriever=retriever, k=k)
        ranks.append(rank)
        if rank is not None:
            hits += 1
            reciprocal_rank_sum += 1 / rank
            
    recall_at_k = hits / len(cases)
    mrr = reciprocal_rank_sum / len(cases)

    return ranks,recall_at_k, mrr

def run_evaluation(cases: list[dict], retrievers: dict, k: int=5,):
    evaluation_results = {}
    for name, retriever in retrievers.items():
        ranks, recall_at_k, mrr = evaluate_retrieval(
            cases,
            retriever=retriever,
            k=k,
        )
        evaluation_results[name] = {
            "ranks": ranks,
            "recall": recall_at_k,
            "mrr": mrr,
        }
    return evaluation_results

def print_evaluation_results(title: str,cases: list[dict],results: dict,retrievers: dict,k: int = 5,)-> None:
    print(f"\n{title}")
    print("Per-query ranks:")

    for i, case in enumerate(cases):
        print(f"\nQuery: {case['query']}")
        print("Expected targets:")

        for target in case["expected_targets"]:
            print(f"  {target}")

        for name in retrievers:
            rank = results[name]["ranks"][i]
            print(f"{name}: rank {rank}")

    print("\nOverall metrics:")

    for name, result in results.items():
        print(
            f"{name}: "
            f"Recall@{k}={result['recall']:.2f}, "
            f"MRR={result['mrr']:.2f}"
        )

def matches_target(result: ScoredPoint, target: dict) -> bool:
    payload = result.payload or {}
    if "url_contains" in target:
        source_url = payload.get("source_url", "")

        if target["url_contains"] not in source_url:
            return False
    if "section" in target:
        section = payload.get("section")

        if section != target["section"]:
            return False

    if "language" in target:
        language = payload.get("language")

        if language != target["language"]:
            return False
    return True


if __name__ == "__main__":
    k = 5

    retrievers = {
        "Dense": retrieve,
        "Sparse": sparse_retrieve,
        "Hybrid 1:1": hybrid_retrieve,
        "Hybrid 2:1": weighted_hybrid_retrieve,
        "Dense + Reranker": reranked_retrieve,
    }

    results = run_evaluation(
        LARGE_CORPUS_TEST_CASES,
        retrievers,
        k=k,
    )

    print_evaluation_results(
        title="LARGE CORPUS TEST SET",
        cases=LARGE_CORPUS_TEST_CASES,
        results=results,
        retrievers=retrievers,
        k=k,
    )
