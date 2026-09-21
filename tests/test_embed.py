import pytest
from rag.chunking.chunk import Chunk
from rag.embeddings.embed import build_embedding_text, embed_chunks
import rag.embeddings.embed as embed_module

def test_build_embedding_text_includes_metadata():
    chunk = Chunk(
        text="client.responses.create(...)",
        source_url="https://example.com/docs",
        section="Create a response",
        language="python",
    )
    result = build_embedding_text(chunk)
    assert "Section: Create a response" in result
    assert "Language: python" in result
    assert "client.responses.create(...)" in result
    
def test_build_embedding_text_without_language():
    chunk = Chunk(
        text="The Responses API generates model responses.",
        source_url="https://example.com/docs",
        section="Responses API",
        language=None,
    )

    result = build_embedding_text(chunk)

    assert "Section: Responses API" in result
    assert "The Responses API generates model responses." in result
    assert "Language:" not in result
    
def test_embed_chunks_pairs_chunks_with_embeddings(monkeypatch):
    def fake_embed_texts(texts):
        return [
            [1.0, 2.0],
            [3.0, 4.0],
        ]

    monkeypatch.setattr(
        embed_module,
        "embed_texts",
        fake_embed_texts,
    )
    chunk_1 = Chunk(
        text="first chunk",
        source_url="https://example.com/1",
        section="Section 1",
        language="python",
    )

    chunk_2 = Chunk(
        text="second chunk",
        source_url="https://example.com/2",
        section="Section 2",
        language=None,
    )
    chunks = [chunk_1, chunk_2]
    embedded_chunks = embed_chunks(chunks)
    assert len(embedded_chunks) == 2
    assert embedded_chunks[0].chunk == chunk_1
    assert embedded_chunks[0].embedding == [1.0, 2.0]
    assert embedded_chunks[1].chunk == chunk_2
    assert embedded_chunks[1].embedding == [3.0, 4.0]

def test_embed_chunks_raises_on_count_mismatch(monkeypatch):
    def fake_embed_texts(texts):
        return [[1.0, 2.0]]

    monkeypatch.setattr(
        embed_module,
        "embed_texts",
        fake_embed_texts,
    )

    chunk_1 = Chunk(
        text="first chunk",
        source_url="https://example.com/1",
        section="Section 1",
        language="python",
    )

    chunk_2 = Chunk(
        text="second chunk",
        source_url="https://example.com/2",
        section="Section 2",
        language=None,
    )

    chunks = [chunk_1, chunk_2]

    with pytest.raises(ValueError):
        embed_chunks(chunks)