"""
Tests for POST /generate API route.

Unit tests mock retrieval and generation so no database or OpenAI is required.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.routes.generate import GenerateResponse
from api.routes.strategies import DocumentResponse


@pytest.fixture
def client() -> TestClient:
    """FastAPI test client."""
    return TestClient(app)


@patch("api.main.generate_endpoint")
def test_generate_endpoint_returns_answer_and_metadata(
    mock_generate_endpoint: AsyncMock,
    client: TestClient,
) -> None:
    """POST /generate returns 200 with answer, documents, model, tokens, cost when endpoint returns success."""
    mock_generate_endpoint.return_value = (
        GenerateResponse(
            answer="RAG is retrieval-augmented generation.",
            documents=[
                DocumentResponse(
                    id="1",
                    content="RAG combines retrieval and generation.",
                    title="",
                    source="",
                    similarity=0.9,
                    metadata={},
                ),
            ],
            model="gpt-4o-mini",
            input_tokens=100,
            output_tokens=20,
            cost_usd=0.003,
            retrieval_latency_ms=50,
            generation_latency_ms=200,
            context_truncated=False,
        ),
        None,
        200,
    )

    response = client.post(
        "/generate",
        json={"query": "What is RAG?", "strategy": "standard", "limit": 5},
    )

    if response.status_code == 401:
        pytest.skip("API key required")
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["answer"] == "RAG is retrieval-augmented generation."
    assert len(data["documents"]) == 1
    assert data["documents"][0]["id"] == "1"
    assert data["model"] == "gpt-4o-mini"
    assert data["input_tokens"] == 100
    assert data["output_tokens"] == 20
    assert data["cost_usd"] == 0.003
    assert data["retrieval_latency_ms"] == 50
    assert data["generation_latency_ms"] == 200
    assert data["context_truncated"] is False


@patch("api.main.generate_endpoint")
def test_generate_endpoint_returns_error_status(
    mock_generate_endpoint: AsyncMock,
    client: TestClient,
) -> None:
    """POST /generate returns 502/503 and message when endpoint returns retrieval/generation error."""
    mock_generate_endpoint.return_value = (
        None,
        "An error occurred while generating the response.",
        502,
    )

    response = client.post(
        "/generate",
        json={"query": "What is RAG?", "strategy": "standard", "limit": 5},
    )

    if response.status_code == 401:
        pytest.skip("API key required")
    assert response.status_code == 502
    data = response.json()
    # FastAPI HTTPException returns {"detail": "..."}
    assert "detail" in data
    assert "generating" in data["detail"].lower() or "error" in data["detail"].lower()
