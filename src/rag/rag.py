from rag.retrieval.retriever import retrieve
from rag.context.builder import build_context
from rag.generation.generator import generate_answer

def answer_query(query: str) -> str:
    results = retrieve(query)
    context = build_context(results)
    answer = generate_answer(query=query,context=context)
    return answer

if __name__ == "__main__":
    answer = answer_query(
        "How do I stream responses from the OpenAI API?"
    )

    print(answer)