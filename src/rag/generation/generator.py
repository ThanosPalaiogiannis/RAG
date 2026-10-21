from openai import OpenAI
from rag.config import GENERATION_MODEL

def generate_answer(query: str, context: str) -> str:
    instructions = (
    "Answer the user's question using only the provided context. "
    "If the context does not contain enough information, say so. "
    "Do not mention internal source labels such as [Source N] in the final answer. "
    "Use only the most relevant sources needed to answer the question. "
    "Do not include multiple programming-language examples unless the user asks for them. "
    "Write the answer in a clear, concise, and user-friendly way."
    )  
    input_text = (
        f"Question:\n{query}\n\n"
        f"Context:\n{context}"
    )
    client = OpenAI()
    response = client.responses.create(
        model=GENERATION_MODEL,
        reasoning={"effort": "low"},
        instructions=instructions,
        input=input_text,
    )
    return response.output_text