from rag.retrieval.retriever import (
    retrieve,
    sparse_retrieve,
    hybrid_retrieve,
    weighted_hybrid_retrieve,
    reranked_retrieve,
)

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

def evaluate_case(case: dict,retriever, k: int = 5):
    query = case["query"]
    expected_section = case["expected_section"]
    results = retriever(query, limit=k)
    rank = None
    for i, result in enumerate(results, start=1):
        section = result.payload.get("section")
        if section == expected_section:
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
        print(f"Expected: {case['expected_section']}")

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

if __name__ == "__main__":
    k = 5

    retrievers = {
        "Dense": retrieve,
        "Sparse": sparse_retrieve,
        "Hybrid 1:1": hybrid_retrieve,
        "Hybrid 2:1": weighted_hybrid_retrieve,
        "Dense + Reranker" : reranked_retrieve,
    }

    dev_results = run_evaluation(
        DEV_CASES,
        retrievers,
        k=k,
    )

    validation_results = run_evaluation(
        VALIDATION_CASES,
        retrievers,
        k=k,
    )
    print_evaluation_results(
    title="DEV SET",
    cases=DEV_CASES,
    results=dev_results,
    retrievers=retrievers,
    k=k,
    )

    print_evaluation_results(
        title="VALIDATION SET",
        cases=VALIDATION_CASES,
        results=validation_results,
        retrievers=retrievers,
        k=k,
    )