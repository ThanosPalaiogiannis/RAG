import bs4
from urllib.parse import urljoin, urldefrag, urlparse   
from collections import deque
from rag.ingestion.fetch import fetch_html
from rag.ingestion.parse import parse_html, ParsedDocument

def crawl(start_url: str, max_pages: int = 10) -> list[ParsedDocument]:
    visited: set[str] = set()
    queue = deque([start_url])
    documents = []
    while queue and len(visited) < max_pages:
        current_url = queue.popleft()
        if current_url in visited:
            continue
        visited.add(current_url)
        html = fetch_html(current_url)
        parsed_doc = parse_html(html, current_url)
        documents.append(parsed_doc)
        discovered_links = discover_links(html, current_url)
        for link in discovered_links:
            if link not in visited and link not in queue:
                queue.append(link)
    return documents
            
def discover_links(html: str, base_url: str) -> set[str]:
    soup = bs4.BeautifulSoup(html, "lxml")
    links = set()
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        absolute_url = urljoin(base_url, href)
        # Remove fragment identifiers
        cleaned_url, _ = urldefrag(absolute_url)
        parsed_url = urlparse(cleaned_url)
        if parsed_url.netloc == "developers.openai.com" and parsed_url.path.startswith("/api/docs"):
            links.add(cleaned_url)
    return links


if __name__ == "__main__":
    documents = crawl("https://developers.openai.com/api/docs", max_pages=10)
    print(f"Collected {len(documents)} document(s)")
    for doc in documents:
        print(f"  {doc.url} ({len(doc.blocks)} blocks)")

