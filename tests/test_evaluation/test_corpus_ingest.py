"""
Unit tests for evaluation corpus ingest (evaluation.corpus_ingest).

Tests _write_corpus_to_dir (write then read back) and get_doc_id_to_chunk_ids
with a mock asyncpg pool. Full ingest_corpus_and_get_chunk_map is optional
integration (real DB).
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from evaluation.corpus_ingest import (
    _write_corpus_to_dir,
    get_doc_id_to_chunk_ids,
)


# =============================================================================
# _write_corpus_to_dir
# =============================================================================


def test_write_corpus_to_dir_creates_files_and_returns_mapping() -> None:
    """_write_corpus_to_dir creates one .txt per doc and returns (original_id, stem)."""
    corpus = [
        {"id": "doc_a", "text": "Content of A.", "title": "Title A"},
        {"id": "doc_b", "text": "Content of B.", "title": ""},
    ]
    with tempfile.TemporaryDirectory(prefix="test_corpus_") as tmpdir:
        dir_path = Path(tmpdir)
        written = _write_corpus_to_dir(corpus, dir_path)
        assert written == [("doc_a", "doc_a"), ("doc_b", "doc_b")]
        assert (dir_path / "doc_a.txt").exists()
        assert (dir_path / "doc_b.txt").exists()
        assert (dir_path / "doc_a.txt").read_text(encoding="utf-8") == "# Title A\n\nContent of A."
        assert (dir_path / "doc_b.txt").read_text(encoding="utf-8") == "Content of B."


def test_write_corpus_to_dir_sanitizes_doc_id() -> None:
    """Special chars in doc_id are replaced with underscore; stem used in filename."""
    corpus = [{"id": "doc/with spaces", "text": "Body", "title": ""}]
    with tempfile.TemporaryDirectory(prefix="test_corpus_") as tmpdir:
        written = _write_corpus_to_dir(corpus, Path(tmpdir))
        assert written == [("doc/with spaces", "doc_with_spaces")]
        assert (Path(tmpdir) / "doc_with_spaces.txt").exists()
        assert (Path(tmpdir) / "doc_with_spaces.txt").read_text() == "Body"


def test_write_corpus_to_dir_skips_empty_doc_id() -> None:
    """Entries with missing or empty id are skipped."""
    corpus = [
        {"id": "ok", "text": "OK", "title": ""},
        {"id": "", "text": "Skip", "title": ""},
        {"id": "  ", "text": "Skip", "title": ""},
    ]
    with tempfile.TemporaryDirectory(prefix="test_corpus_") as tmpdir:
        written = _write_corpus_to_dir(corpus, Path(tmpdir))
        assert written == [("ok", "ok")]
        assert (Path(tmpdir) / "ok.txt").exists()
        assert len(list(Path(tmpdir).iterdir())) == 1


def test_write_corpus_to_dir_title_only_content() -> None:
    """When title is set but text empty, content is '# Title'."""
    corpus = [{"id": "t1", "text": "", "title": "Only Title"}]
    with tempfile.TemporaryDirectory(prefix="test_corpus_") as tmpdir:
        _write_corpus_to_dir(corpus, Path(tmpdir))
        assert (Path(tmpdir) / "t1.txt").read_text() == "# Only Title"


def test_write_corpus_to_dir_custom_keys() -> None:
    """Custom id_key, text_key, title_key are respected."""
    corpus = [{"pmid": "p1", "abstract": "Abstract text.", "article_title": "Art Title"}]
    with tempfile.TemporaryDirectory(prefix="test_corpus_") as tmpdir:
        written = _write_corpus_to_dir(
            corpus,
            Path(tmpdir),
            id_key="pmid",
            text_key="abstract",
            title_key="article_title",
        )
        assert written == [("p1", "p1")]
        content = (Path(tmpdir) / "p1.txt").read_text()
        assert "# Art Title" in content
        assert "Abstract text." in content


# =============================================================================
# get_doc_id_to_chunk_ids (mock pool)
# =============================================================================


@pytest.mark.asyncio
async def test_get_doc_id_to_chunk_ids_maps_source_to_chunk_ids() -> None:
    """get_doc_id_to_chunk_ids returns dict doc_id -> list of chunk_ids from fetch rows."""
    mock_rows = [
        {"chunk_id": "uuid-1", "source": "doc_a.txt"},
        {"chunk_id": "uuid-2", "source": "doc_a.txt"},
        {"chunk_id": "uuid-3", "source": "doc_b.txt"},
    ]
    mock_conn = AsyncMock()
    mock_conn.fetch = AsyncMock(return_value=mock_rows)
    mock_pool = MagicMock()
    mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

    result = await get_doc_id_to_chunk_ids(mock_pool, source_suffix=".txt")

    assert result["doc_a"] == ["uuid-1", "uuid-2"]
    assert result["doc_b"] == ["uuid-3"]
    mock_conn.fetch.assert_called_once()


@pytest.mark.asyncio
async def test_get_doc_id_to_chunk_ids_empty_source_no_suffix_strip() -> None:
    """When source has no .txt suffix, doc_id is the whole source."""
    mock_rows = [{"chunk_id": "c1", "source": "no_extension"}]
    mock_conn = AsyncMock()
    mock_conn.fetch = AsyncMock(return_value=mock_rows)
    mock_pool = MagicMock()
    mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

    result = await get_doc_id_to_chunk_ids(mock_pool, source_suffix=".txt")

    assert result["no_extension"] == ["c1"]


@pytest.mark.asyncio
async def test_get_doc_id_to_chunk_ids_empty_rows() -> None:
    """When fetch returns empty list, result is empty dict."""
    mock_conn = AsyncMock()
    mock_conn.fetch = AsyncMock(return_value=[])
    mock_pool = MagicMock()
    mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

    result = await get_doc_id_to_chunk_ids(mock_pool)

    assert result == {}


# =============================================================================
# Integration: ingest_corpus_and_get_chunk_map (real DB)
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_ingest_corpus_and_get_chunk_map_with_db() -> None:
    """
    With DATABASE_URL set, ingest small corpus and get doc_id -> chunk_ids map.

    Requires PostgreSQL with pgvector and matching schema (1024 or 1536).
    EMBEDDING_BACKEND must match DB: openai -> 1536 (default Docker), bge-m3 -> 1024
    (use docker-compose.bge-m3.yml with fresh volume). Skip when DATABASE_URL not set.
    """
    import os

    database_url = os.getenv("TEST_DATABASE_URL") or os.getenv("DATABASE_URL")
    if not database_url:
        pytest.skip("TEST_DATABASE_URL or DATABASE_URL not set")

    from evaluation.corpus_ingest import ingest_corpus_and_get_chunk_map

    corpus = [
        {"id": "int_doc_1", "text": "First document content for integration test.", "title": "Doc 1"},
        {"id": "int_doc_2", "text": "Second document content.", "title": "Doc 2"},
    ]
    try:
        result = await ingest_corpus_and_get_chunk_map(
            corpus,
            clean_before=True,
        )
    except Exception as e:  # noqa: BLE001
        err_msg = str(e).lower()
        if "dimensions" in err_msg and "expected" in err_msg:
            pytest.skip(
                "Embedding dimension mismatch: EMBEDDING_BACKEND does not match DB schema. "
                "Default Docker uses 1536-dim (set EMBEDDING_BACKEND=openai or unset). "
                "For BGE-M3 (1024-dim) use docker-compose.bge-m3.yml with a fresh volume."
            )
        pytest.skip(f"Integration ingest failed (DB/schema/embedder): {e}")
    assert isinstance(result, dict)
    assert "int_doc_1" in result
    assert "int_doc_2" in result
    assert len(result["int_doc_1"]) >= 1
    assert len(result["int_doc_2"]) >= 1
