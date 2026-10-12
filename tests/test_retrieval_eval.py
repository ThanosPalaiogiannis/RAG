from types import SimpleNamespace

import pytest

from rag.evaluation import retrieval_eval


def test_evaluate_case_returns_correct_rank(monkeypatch):
    def fake_retrieve(query: str, limit: int = 5):
        return [
            SimpleNamespace(payload={"section": "Wrong section"}),
            SimpleNamespace(payload={"section": "Expected section"}),
            SimpleNamespace(payload={"section": "Another section"}),
        ]

    monkeypatch.setattr(
        retrieval_eval,
        "retrieve",
        fake_retrieve,
    )

    case = {
        "query": "test question",
        "expected_section": "Expected section",
    }

    rank = retrieval_eval.evaluate_case(case,retriever=fake_retrieve, k=3)

    assert rank == 2


def test_evaluate_case_returns_none_when_section_not_found(monkeypatch):
    def fake_retrieve(query: str, limit: int = 5):
        return [
            SimpleNamespace(payload={"section": "Section A"}),
            SimpleNamespace(payload={"section": "Section B"}),
            SimpleNamespace(payload={"section": "Section C"}),
        ]

    monkeypatch.setattr(
        retrieval_eval,
        "retrieve",
        fake_retrieve,
    )

    case = {
        "query": "test question",
        "expected_section": "Missing section",
    }

    rank = retrieval_eval.evaluate_case(case,retriever=fake_retrieve, k=3)

    assert rank is None


def test_evaluate_retrieval_calculates_metrics(monkeypatch):
    ranks = iter([1, 2, None])

    def fake_evaluate_case(case: dict,retriever , k: int = 5):
        return next(ranks)

    monkeypatch.setattr(
        retrieval_eval,
        "evaluate_case",
        fake_evaluate_case,
    )

    cases = [
        {"query": "query 1", "expected_section": "section 1"},
        {"query": "query 2", "expected_section": "section 2"},
        {"query": "query 3", "expected_section": "section 3"},
    ]

    _, recall_at_k, mrr = retrieval_eval.evaluate_retrieval(
    cases,
    retriever=retrieval_eval.retrieve,
    k=5,
    )
    assert recall_at_k == pytest.approx(2 / 3)
    assert mrr == pytest.approx(0.5)