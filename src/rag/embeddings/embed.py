from openai import OpenAI
from dataclasses import dataclass
from rag.chunking.chunk import Chunk

embedding_model = "text-embedding-3-small"

@dataclass
class EmbeddedChunk:
    chunk: Chunk
    embedding: list[float]

def embed_texts(texts: list[str]) -> list[list[float]]:
    client = OpenAI()
    response = client.embeddings.create(
        model=embedding_model,
        input=texts,
    )

    return [item.embedding for item in response.data]

def build_embedding_text(chunk: Chunk) -> str:
    segments = []
    if chunk.section:
        segments.append(f"Section: {chunk.section}")
    if chunk.language:
        segments.append(f"Language: {chunk.language}")

    segments.append(f"Text: {chunk.text}")
    return "\n\n".join(segments) 

def embed_chunks(chunks: list[Chunk]) -> list[EmbeddedChunk]:
    texts = [build_embedding_text(chunk) for chunk in chunks]
    embeddings = embed_texts(texts)
    if len(chunks) != len(embeddings):
        raise ValueError("Number of embeddings does not match number of chunks")
    embedded_chunks = []
    for chunk, embedding in zip(chunks, embeddings):
        embedded_chunk = EmbeddedChunk(
            chunk=chunk,
            embedding=embedding
        )
        embedded_chunks.append(embedded_chunk)
    return embedded_chunks     

if __name__ == "__main__":
    chunk_1 = Chunk(
        text="client.responses.create(...)",
        source_url="https://example.com/docs",
        section="Create a response",
        language="python",
    )
    chunk_2 = Chunk(
        text="The Responses API can generate model outputs from user input.",
        source_url="https://example.com/docs",
        section="Responses API",
        language=None,
    )
    chunks  = [chunk_1,chunk_2]
    embedded_chunks = embed_chunks(chunks)
    print("Embedded chunks:", len(embedded_chunks))
    print("Dimensions:", len(embedded_chunks[0].embedding))
    print("First chunk matches:", embedded_chunks[0].chunk.text == chunk_1.text)