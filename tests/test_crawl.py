from rag.ingestion.crawl import  discover_links
import pytest
from urllib.parse import urlparse

@pytest.fixture
def discovered_links():
    with open("tests/fixtures/openai_docs.html", encoding="utf-8") as f:
        html = f.read()
    return discover_links(html, "https://developers.openai.com/api/docs")

def test_discovers_links(discovered_links):
    assert len(discovered_links) > 0
    
def test_all_links_same_domain(discovered_links):
    for link in discovered_links:
        parsed_url = urlparse(link)
        assert parsed_url.netloc == "developers.openai.com"

def test_all_paths_uder_api_docs(discovered_links):
    for link in discovered_links:
        parsed_url = urlparse(link)
        assert parsed_url.path.startswith("/api/docs")
        
def test_no_fragments(discovered_links):
    for link in discovered_links:
        assert "#" not in link