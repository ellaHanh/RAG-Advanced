"""
Unit tests for reranking strategy.

Covers rerank-only mode (when input_documents from previous chain step is set):
no DB call, returns reranked subset of input_documents using original_query.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from orchestration.executor import ExecutionContext
from orchestration.models import Document, StrategyConfig
from strategies.agents.reranking import _reranking_search_impl


@pytest.fixture
def input_docs() -> list[Document]:
    """Three documents for rerank-only input."""
    return [
        Document(id="c1", content="First chunk", title="A", source="", similarity=0.7),
        Document(id="c2", content="Second chunk", title="B", source="", similarity=0.8),
        Document(id="c3", content="Third chunk", title="C", source="", similarity=0.6),
    ]


@pytest.fixture
def ctx_with_input_docs(input_docs: list[Document]) -> ExecutionContext:
    """ExecutionContext with input_documents and original_query (chain mode)."""
    return ExecutionContext(
        query="current",
        config=StrategyConfig(limit=5, initial_k=20, final_k=2),
        cost_tracker=MagicMock(),
        metadata={},
        input_documents=input_docs,
        original_query="original user query",
    )


@pytest.mark.asyncio
async def test_rerank_only_returns_subset_of_input_documents(
    ctx_with_input_docs: ExecutionContext,
    input_docs: list[Document],
):
    """When input_documents is set, strategy returns reranked subset of size final_k."""
    mock_reranker = MagicMock()
    # Scores: middle doc best, then first, then last -> order c2, c1, c3; take top 2
    mock_reranker.predict.return_value = [0.5, 0.9, 0.3]

    with patch("strategies.agents.reranking._get_reranker", return_value=mock_reranker):
        result = await _reranking_search_impl(
            ctx_with_input_docs,
            pool=MagicMock(),
            embed_query_fn=MagicMock(),
        )

    assert len(result) == 2
    assert result[0].id == "c2"
    assert result[1].id == "c1"
    assert result[0].content == "Second chunk"
    assert result[1].content == "First chunk"
    mock_reranker.predict.assert_called_once()
    pairs = mock_reranker.predict.call_args[0][0]
    assert len(pairs) == 3
    assert pairs[0][0] == "original user query"
    assert pairs[0][1] == "First chunk"


@pytest.mark.asyncio
async def test_rerank_only_does_not_call_pool(
    ctx_with_input_docs: ExecutionContext,
):
    """When input_documents is set, pool.acquire() is not called (no DB retrieval)."""
    mock_pool = MagicMock()
    mock_reranker = MagicMock()
    mock_reranker.predict.return_value = [0.1, 0.2, 0.3]

    with patch("strategies.agents.reranking._get_reranker", return_value=mock_reranker):
        await _reranking_search_impl(
            ctx_with_input_docs,
            pool=mock_pool,
            embed_query_fn=MagicMock(),
        )

    mock_pool.acquire.assert_not_called()


@pytest.mark.asyncio
async def test_rerank_only_uses_original_query_for_scoring(
    input_docs: list[Document],
):
    """Rerank-only mode uses original_query (not current query) for cross-encoder."""
    ctx = ExecutionContext(
        query="current query",
        config=StrategyConfig(final_k=1),
        cost_tracker=MagicMock(),
        input_documents=input_docs,
        original_query="original user query",
    )
    mock_reranker = MagicMock()
    mock_reranker.predict.return_value = [0.9]

    with patch("strategies.agents.reranking._get_reranker", return_value=mock_reranker):
        await _reranking_search_impl(ctx, pool=MagicMock(), embed_query_fn=MagicMock())

    call_args = mock_reranker.predict.call_args[0][0]
    assert call_args[0][0] == "original user query"
