"""
Pytest Configuration and Fixtures.

Provides shared fixtures for testing RAG-Advanced modules.

Fixtures:
    - test_db_pool: Async PostgreSQL connection pool for testing
    - test_redis: Redis client for testing
    - mock_openai: Mocked OpenAI client
    - sample_documents: Sample documents for testing
    - sample_ground_truth: Ground truth data for evaluation testing
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Generator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio


# =============================================================================
# Event Loop Configuration
# =============================================================================


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# =============================================================================
# Database Fixtures
# =============================================================================


@pytest_asyncio.fixture
async def test_db_pool() -> AsyncGenerator[AsyncMock, None]:
    """
    Mock database connection pool for unit tests.
    
    For integration tests, use a real test database by setting
    TEST_DATABASE_URL environment variable.
    """
    mock_pool = AsyncMock()
    mock_conn = AsyncMock()
    
    # Setup mock connection context manager
    mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
    mock_pool.acquire.return_value.__aexit__.return_value = None
    
    # Setup common query results
    mock_conn.fetch.return_value = []
    mock_conn.fetchrow.return_value = None
    mock_conn.fetchval.return_value = None
    mock_conn.execute.return_value = "OK"
    
    yield mock_pool


# =============================================================================
# Redis Fixtures
# =============================================================================


@pytest_asyncio.fixture
async def test_redis() -> AsyncGenerator[AsyncMock, None]:
    """
    Mock Redis client for unit tests.
    
    For integration tests, use a real test Redis by setting
    TEST_REDIS_URL environment variable.
    """
    mock_redis = AsyncMock()
    
    # Setup common Redis operations
    mock_redis.get.return_value = None
    mock_redis.set.return_value = True
    mock_redis.delete.return_value = 1
    mock_redis.zadd.return_value = 1
    mock_redis.zcard.return_value = 0
    mock_redis.zremrangebyscore.return_value = 0
    
    yield mock_redis


# =============================================================================
# OpenAI Fixtures
# =============================================================================


@pytest.fixture
def mock_openai() -> MagicMock:
    """
    Mock OpenAI client for testing without API calls.
    
    Returns:
        MagicMock: Mocked OpenAI client with common responses.
    """
    mock_client = MagicMock()
    
    # Mock embeddings
    mock_embedding_response = MagicMock()
    mock_embedding_response.data = [
        MagicMock(embedding=[0.1] * 1536)  # text-embedding-3-small dimensions
    ]
    mock_embedding_response.usage = MagicMock(
        prompt_tokens=10,
        total_tokens=10,
    )
    mock_client.embeddings.create = AsyncMock(return_value=mock_embedding_response)
    
    # Mock chat completions
    mock_chat_response = MagicMock()
    mock_chat_response.choices = [
        MagicMock(
            message=MagicMock(content="Mock response"),
            finish_reason="stop",
        )
    ]
    mock_chat_response.usage = MagicMock(
        prompt_tokens=50,
        completion_tokens=20,
        total_tokens=70,
    )
    mock_client.chat.completions.create = AsyncMock(return_value=mock_chat_response)
    
    return mock_client


# =============================================================================
# Sample Data Fixtures
# =============================================================================


@pytest.fixture
def sample_documents() -> list[dict[str, Any]]:
    """
    Sample documents for testing retrieval strategies.
    
    Returns:
        List of document dictionaries with id, content, and metadata.
    """
    return [
        {
            "id": "doc_1",
            "content": "Machine learning is a subset of artificial intelligence.",
            "metadata": {"source": "test", "type": "definition"},
        },
        {
            "id": "doc_2",
            "content": "Deep learning uses neural networks with multiple layers.",
            "metadata": {"source": "test", "type": "definition"},
        },
        {
            "id": "doc_3",
            "content": "Natural language processing enables computers to understand human language.",
            "metadata": {"source": "test", "type": "definition"},
        },
        {
            "id": "doc_4",
            "content": "Retrieval-augmented generation combines retrieval with LLM generation.",
            "metadata": {"source": "test", "type": "definition"},
        },
        {
            "id": "doc_5",
            "content": "Vector databases store embeddings for semantic search.",
            "metadata": {"source": "test", "type": "technology"},
        },
    ]


@pytest.fixture
def sample_ground_truth() -> list[dict[str, Any]]:
    """
    Sample ground truth data for evaluation testing.
    
    Returns:
        List of query/ground truth pairs for evaluation.
    """
    return [
        {
            "query_id": "q1",
            "query": "What is machine learning?",
            "relevant_doc_ids": ["doc_1", "doc_2"],
            "relevance_scores": {"doc_1": 2, "doc_2": 1},  # 2=highly, 1=partial
        },
        {
            "query_id": "q2",
            "query": "How does RAG work?",
            "relevant_doc_ids": ["doc_4"],
            "relevance_scores": {"doc_4": 2},
        },
        {
            "query_id": "q3",
            "query": "What are vector databases?",
            "relevant_doc_ids": ["doc_5"],
            "relevance_scores": {"doc_5": 2},
        },
    ]


@pytest.fixture
def sample_retrieved_docs() -> list[str]:
    """
    Sample retrieved document IDs for testing metrics calculation.
    
    Returns:
        List of document IDs in ranked order.
    """
    return ["doc_1", "doc_3", "doc_2", "doc_5", "doc_4"]


# =============================================================================
# Strategy Fixtures
# =============================================================================


@pytest.fixture
def strategy_config() -> dict[str, Any]:
    """
    Default strategy configuration for testing.
    
    Returns:
        Dictionary with common strategy configuration.
    """
    return {
        "limit": 5,
        "initial_k": 20,
        "final_k": 5,
        "num_variations": 3,
        "max_iterations": 2,
    }


# =============================================================================
# Markers Configuration
# =============================================================================


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests (fast, isolated)")
    config.addinivalue_line("markers", "integration: Integration tests (require external services)")
    config.addinivalue_line("markers", "e2e: End-to-end tests")
    config.addinivalue_line("markers", "slow: Slow running tests")
