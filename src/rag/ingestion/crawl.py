import time
from collections import deque
from urllib.parse import urljoin, urldefrag, urlparse
import httpx
import bs4
from rag.ingestion.fetch import fetch_html
from rag.ingestion.parse import (
    parse_html,
    ParsedDocument,
    HeadingBlock,
    ParagraphBlock,
    CodeBlock,
)
import hashlib

ALLOWED_PATH_PREFIX = "/api/docs"


def crawl(start_url: str,  max_pages: int | None = None) -> list[ParsedDocument]:
    visited: set[str] = set()
    queue = deque([start_url])
    documents = []
    seen_fingerprints: set[str] = set()

    while queue and (max_pages is None or len(visited) < max_pages):
        current_url = queue.popleft()

        if current_url in visited:
            continue

        visited.add(current_url)
        try:
            html = fetch_html(current_url)
        except httpx.HTTPStatusError:
            continue
        time.sleep(0.5)

        parsed_doc = parse_html(html, current_url)
        fingerprint = document_fingerprint(parsed_doc)
        if fingerprint not in seen_fingerprints:
            seen_fingerprints.add(fingerprint)
            documents.append(parsed_doc)

        discovered_links = discover_links(html, current_url)

        for link in sorted(discovered_links):
            if link not in visited and link not in queue:
                queue.append(link)

    return documents


def discover_links(html: str, base_url: str,) -> set[str]:
    soup = bs4.BeautifulSoup(html, "lxml")
    links = set()

    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]

        absolute_url = urljoin(base_url, href)

        cleaned_url, _ = urldefrag(absolute_url)

        parsed_url = urlparse(cleaned_url)
        if parsed_url.path.endswith(".xml"):
            continue
        if (parsed_url.netloc == "developers.openai.com" and parsed_url.path.startswith(ALLOWED_PATH_PREFIX)):

            links.add(cleaned_url)

    return links


def document_fingerprint(document: ParsedDocument) -> str:
    parts = []
    for block in document.blocks:
        if isinstance(block, HeadingBlock):
            parts.append(f"HEADING|{block.level}|{block.text}")
        elif isinstance(block, ParagraphBlock):
            parts.append(f"PARAGRAPH|{block.text}")
        elif isinstance(block, CodeBlock):
            parts.append(f"CODE|{block.language}|{block.code}")
    content ="\n".join(parts)
    content_bytes = content.encode("utf-8")
    return hashlib.sha256(content_bytes).hexdigest()

