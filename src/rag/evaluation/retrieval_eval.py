from rag.retrieval.retriever import retrieve



EVAL_CASES = [
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
]

def evaluate_case(case: dict, k: int = 5):
    query = case["query"]
    expected_section = case["expected_section"]
    results = retrieve(query, limit=k)
    rank = None
    for i, result in enumerate(results, start=1):
        section = result.payload.get("section")
        if section == expected_section:
            rank = i
            break
    return rank

def evaluate_retrieval(cases: list[dict], k: int = 5):
    hits = 0
    reciprocal_rank_sum = 0.0

    for case in cases:
        rank = evaluate_case(case, k=k)

        if rank is not None:
            hits += 1
            reciprocal_rank_sum += 1 / rank
            
    recall_at_k = hits / len(cases)
    mrr = reciprocal_rank_sum / len(cases)

    return recall_at_k, mrr

if __name__ == "__main__":
    k=5
    recall_at_k, mrr = evaluate_retrieval(EVAL_CASES, k=k)

    print(f"Recall@{k}: {recall_at_k:.2f}")
    print(f"MRR: {mrr:.2f}")
    