"""
Unit tests for embedder (strategies.utils.embedder).

Tests backend selection, get_embedding_dimensions, and OpenAI path with mocks.
BGE-M3 path is tested with mock (no model download) so CI stays fast.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from strategies.utils.embedder import (
    BGE_M3_DIMENSIONS,
    DEFAULT_DIMENSIONS,
    embed_documents,
    embed_query,
    get_embedding_dimensions,
)


# =============================================================================
# get_embedding_dimensions and _get_backend (via dimensions)
# =============================================================================


def test_get_embedding_dimensions_default_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    """With EMBEDDING_BACKEND unset or openai, return 1536 or EMBEDDING_DIMENSIONS."""
    monkeypatch.delenv("EMBEDDING_BACKEND", raising=False)
    monkeypatch.delenv("EMBEDDING_DIMENSIONS", raising=False)
    assert get_embedding_dimensions() == DEFAULT_DIMENSIONS

    monkeypatch.setenv("EMBEDDING_BACKEND", "openai")
    assert get_embedding_dimensions() == DEFAULT_DIMENSIONS

    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "512")
    assert get_embedding_dimensions() == 512


def test_get_embedding_dimensions_bge_m3(monkeypatch: pytest.MonkeyPatch) -> None:
    """With EMBEDDING_BACKEND=bge-m3, return 1024 regardless of EMBEDDING_DIMENSIONS."""
    monkeypatch.setenv("EMBEDDING_BACKEND", "bge-m3")
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "1536")
    assert get_embedding_dimensions() == BGE_M3_DIMENSIONS


def test_get_embedding_dimensions_backend_case_insensitive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Backend is normalized to lowercase."""
    monkeypatch.setenv("EMBEDDING_BACKEND", "BGE-M3")
    assert get_embedding_dimensions() == BGE_M3_DIMENSIONS


# =============================================================================
# OpenAI path (mocked)
# =============================================================================


@pytest.fixture(autouse=True)
def _clear_embedder_module_cache() -> None:
    """Ensure we don't leak backend state between tests (bge_m3 model cache)."""
    import strategies.utils.embedder as embedder_mod
    embedder_mod._bge_m3_model = None


@pytest.mark.asyncio
async def test_embed_query_openai_mocked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """embed_query with OpenAI backend returns vector of expected length (mocked)."""
    monkeypatch.setenv("EMBEDDING_BACKEND", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    mock_response = MagicMock()
    mock_response.data = [MagicMock(embedding=[0.1] * 1536)]

    mock_client = MagicMock()
    mock_client.embeddings.create = AsyncMock(return_value=mock_response)

    with patch("openai.AsyncOpenAI", return_value=mock_client):
        result = await embed_query("hello")
    assert len(result) == 1536
    mock_client.embeddings.create.assert_called_once()


@pytest.mark.asyncio
async def test_embed_query_openai_raises_without_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """embed_query with openai backend raises when OPENAI_API_KEY not set."""
    monkeypatch.setenv("EMBEDDING_BACKEND", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        await embed_query("hello")


@pytest.mark.asyncio
async def test_embed_documents_openai_mocked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """embed_documents with OpenAI backend returns list of vectors (mocked)."""
    monkeypatch.setenv("EMBEDDING_BACKEND", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    mock_response = MagicMock()
    mock_response.data = [
        MagicMock(embedding=[0.1] * 1536),
        MagicMock(embedding=[0.2] * 1536),
    ]

    mock_client = MagicMock()
    mock_client.embeddings.create = AsyncMock(return_value=mock_response)

    with patch("openai.AsyncOpenAI", return_value=mock_client):
        result = await embed_documents(["a", "b"], batch_size=10)
    assert len(result) == 2
    assert len(result[0]) == 1536
    assert len(result[1]) == 1536
    mock_client.embeddings.create.assert_called_once()


@pytest.mark.asyncio
async def test_embed_documents_empty_returns_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """embed_documents with empty list returns [] (no backend call)."""
    monkeypatch.setenv("EMBEDDING_BACKEND", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    with patch("openai.AsyncOpenAI"):
        result = await embed_documents([])
    assert result == []


# =============================================================================
# BGE-M3 path (mocked SentenceTransformer and run_in_executor)
# =============================================================================


@pytest.mark.asyncio
async def test_embed_query_bge_m3_mocked(monkeypatch: pytest.MonkeyPatch) -> None:
    """embed_query with bge-m3 backend returns 1024-dim vector (mocked)."""
    import numpy as np

    monkeypatch.setenv("EMBEDDING_BACKEND", "bge-m3")
    fake_arr = np.array([[0.5] * BGE_M3_DIMENSIONS])

    with (
        patch("sentence_transformers.SentenceTransformer") as mock_st,
        patch("strategies.utils.embedder.asyncio.get_event_loop") as mock_loop,
    ):
        mock_model = MagicMock()
        mock_model.encode.return_value = fake_arr
        mock_st.return_value = mock_model

        async def run_in_executor(executor, func, *args):  # noqa: ARG001
            return func(*args)

        loop = MagicMock()
        loop.run_in_executor = AsyncMock(side_effect=run_in_executor)
        mock_loop.return_value = loop

        result = await embed_query("test query")
    assert len(result) == BGE_M3_DIMENSIONS
    mock_model.encode.assert_called()


@pytest.mark.asyncio
async def test_embed_documents_bge_m3_mocked(monkeypatch: pytest.MonkeyPatch) -> None:
    """embed_documents with bge-m3 returns list of 1024-dim vectors (mocked)."""
    monkeypatch.setenv("EMBEDDING_BACKEND", "bge-m3")

    import numpy as np
    fake_arr = np.array([[0.1] * BGE_M3_DIMENSIONS, [0.2] * BGE_M3_DIMENSIONS])

    with (
        patch("sentence_transformers.SentenceTransformer") as mock_st,
        patch("strategies.utils.embedder.asyncio.get_event_loop") as mock_loop,
    ):
        mock_model = MagicMock()
        mock_model.encode.return_value = fake_arr
        mock_st.return_value = mock_model

        async def run_in_executor(executor, func, *args):  # noqa: ARG001
            return func(*args)

        loop = MagicMock()
        loop.run_in_executor = AsyncMock(side_effect=run_in_executor)
        mock_loop.return_value = loop

        result = await embed_documents(["doc1", "doc2"], batch_size=64)
    assert len(result) == 2
    assert len(result[0]) == BGE_M3_DIMENSIONS
    assert len(result[1]) == BGE_M3_DIMENSIONS
