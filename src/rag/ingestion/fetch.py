import httpx

def fetch_html(url: str) -> str:
    headers = {"User-Agent": "RAG-Ingestion/0.1"}
    response = httpx.get(url, headers=headers, timeout=10.0)
    response.raise_for_status()
    return response.text

