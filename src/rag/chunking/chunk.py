from dataclasses import dataclass
from rag.ingestion.parse import HeadingBlock, ParagraphBlock, CodeBlock, ParsedDocument 
import tiktoken

encoding = tiktoken.get_encoding("cl100k_base")
PROSE_TOKEN_BUDGET = 300

@dataclass
class Chunk:
    text: str
    source_url: str
    section: str | None
    language: str | None


def chunk_document(doc:ParsedDocument) -> list[Chunk]:
    chunks: list[Chunk] = []
    current_section: str | None = None
    prose_buffer: list[str] = []
    for block in doc.blocks:
        if isinstance(block, HeadingBlock):
            if prose_buffer:
                text = " ".join(prose_buffer)
                chunk = Chunk(
                    text=text,
                    source_url=doc.url,
                    section=current_section,
                    language=None,
                )
                chunks.append(chunk)
                prose_buffer.clear()
            current_section = block.text
        elif isinstance(block, ParagraphBlock):
            candidate_text = " ".join(prose_buffer + [block.text])
            candidate_tokens = len(encoding.encode(candidate_text))
            if candidate_tokens > PROSE_TOKEN_BUDGET and prose_buffer:
                text = " ".join(prose_buffer)
                chunk = Chunk(
                    text=text,
                    source_url=doc.url,
                    section=current_section,
                    language=None,
                )
                chunks.append(chunk)
                prose_buffer.clear()
            prose_buffer.append(block.text)
        elif isinstance(block, CodeBlock):
            chunk = Chunk(
                text=block.code,
                source_url=doc.url,
                section=current_section,
                language=block.language,
            )
            chunks.append(chunk)

    if prose_buffer:
        text = " ".join(prose_buffer)
        chunk = Chunk(
            text=text,
            source_url=doc.url,
            section=current_section,
            language=None,
        )
        chunks.append(chunk)

    return chunks
if __name__ == "__main__":
    from rag.ingestion.fetch import fetch_html
    from rag.ingestion.parse import parse_html

    url = "https://developers.openai.com/api/docs"
    html = fetch_html(url)
    doc = parse_html(html, url)
    chunks = chunk_document(doc)

    print(f"Total chunks: {len(chunks)}")
    for c in chunks[:5]:
        print(f"--- section={c.section!r} language={c.language!r} ---")
        print(c.text)
        print()
