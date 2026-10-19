from fastapi.testclient import TestClient
from rag.response import RAGResponse
from rag.exceptions import RAGServiceError,RAGConfigurationError
from rag.api import app

client = TestClient(app)


def test_ask_rejects_whitespace_query():
    response = client.post(
        "/ask",
        json={"query": "   "}
    )
    assert response.status_code == 422

def test_ask_returns_answer(monkeypatch):

    def fake_answer_query(query: str) -> RAGResponse:
        return RAGResponse(
            answer="Mock answer",
            sources=[]
        )

    monkeypatch.setattr("rag.api.answer_query",fake_answer_query)
    response = client.post(
        "/ask",
        json={"query": "What are embeddings?"}
    )
    assert response.status_code == 200
    assert response.json()["answer"] == "Mock answer"


def test_ask_rejects_empty_query():
    response = client.post(
        "/ask",
        json={"query":""}
    )
    assert response.status_code == 422

def test_ask_returns_503_on_service_failure(monkeypatch):

    def fake_answer_query(query: str) -> RAGResponse:
        raise RAGServiceError("Test service failure")

    monkeypatch.setattr("rag.api.answer_query",fake_answer_query)

    response = client.post(
        "/ask",
        json={"query": "How do embeddings work?"}
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "RAG service temporarily unavailable"

def test_ask_returns_500_on_configuration_failure(monkeypatch):
    def fake_answer_query(query: str) -> RAGResponse:
        raise RAGConfigurationError("RAG configuration failed")

    monkeypatch.setattr("rag.api.answer_query",fake_answer_query)
    response = client.post(
        "/ask",
        json={"query": "How do embeddings work?"}
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "Internal server error"