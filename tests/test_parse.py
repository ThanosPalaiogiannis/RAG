import pytest
from rag.ingestion.parse import parse_html, HeadingBlock, ParagraphBlock, CodeBlock

@pytest.fixture
def parsed_doc():
    with open("tests/fixtures/openai_docs.html", encoding="utf-8") as f:
        html = f.read()
    return parse_html(html, "https://developers.openai.com/api/docs")


def test_block_counts(parsed_doc):
    headings = [b for b in parsed_doc.blocks if isinstance(b, HeadingBlock)]
    paragraphs = [b for b in parsed_doc.blocks if isinstance(b, ParagraphBlock)]
    code_blocks = [b for b in parsed_doc.blocks if isinstance(b, CodeBlock)]

    assert len(headings) == 7
    assert len(paragraphs) == 3
    assert len(code_blocks) == 8


def test_languages_found(parsed_doc):
    languages = {b.language for b in parsed_doc.blocks if isinstance(b, CodeBlock)}
    assert languages == {"javascript", "csharp", "ruby", "bash", "go", "java", "python"}


def test_go_code_has_no_double_blank_lines(parsed_doc):
    go_block = next(
        b for b in parsed_doc.blocks if isinstance(b, CodeBlock) and b.language == "go"
    )
    assert "\n\n\n" not in go_block.code
    
def test_parse_html_raises_without_main():
    with pytest.raises(ValueError):
        parse_html("<html><body><p>no main here</p></body></html>", "https://example.com")
