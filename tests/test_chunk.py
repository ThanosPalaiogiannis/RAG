import pytest
from rag.ingestion.parse import parse_html, CodeBlock
from rag.chunking.chunk import chunk_document

@pytest.fixture
def doc_and_chunks():
    with open("tests/fixtures/openai_docs.html", encoding="utf-8") as f:
        html = f.read()
    doc = parse_html(html, "https://developers.openai.com/api/docs")
    chunks = chunk_document(doc)
    return doc, chunks

def test_code_chunk_count_matches_source(doc_and_chunks):
    doc, chunks = doc_and_chunks
    code_blocks =[block for block in doc.blocks if isinstance(block, CodeBlock)]
    code_chunks = [chunk for chunk in chunks if chunk.language is not None]
    assert len(code_blocks) == len(code_chunks), f"Expected {len(code_blocks)} code chunks, but got {len(code_chunks)}"

def test_all_chunks_have_correct_source_url(doc_and_chunks):
    doc, chunks = doc_and_chunks
    for chunk in chunks:
        assert chunk.source_url == doc.url


def test_no_empty_chunks(doc_and_chunks):
    doc, chunks = doc_and_chunks
    for chunk in chunks:
        assert chunk.text.strip() != ""
