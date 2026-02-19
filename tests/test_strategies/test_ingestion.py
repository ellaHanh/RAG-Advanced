"""
Unit tests for the ingestion pipeline.

Tests chunker, document reader (text only), models, and find_document_files
without database or OpenAI.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from strategies.ingestion.chunker import chunk_document
from strategies.ingestion.document_reader import extract_title, read_document
from strategies.ingestion.models import DocumentChunk, IngestionConfig, IngestionResult


class TestIngestionConfig:
    """IngestionConfig validation."""

    def test_defaults(self) -> None:
        cfg = IngestionConfig()
        assert cfg.chunk_size == 1000
        assert cfg.chunk_overlap == 200
        assert cfg.use_semantic_chunking is True
        assert cfg.max_tokens == 512

    def test_overlap_less_than_size(self) -> None:
        with pytest.raises(ValueError, match="chunk_overlap"):
            IngestionConfig(chunk_size=500, chunk_overlap=500)
        IngestionConfig(chunk_size=500, chunk_overlap=199)

    def test_bounds(self) -> None:
        cfg = IngestionConfig(chunk_size=200, chunk_overlap=50)
        assert cfg.chunk_size == 200


class TestDocumentChunk:
    """DocumentChunk model."""

    def test_minimal(self) -> None:
        c = DocumentChunk(content="hello", index=0)
        assert c.content == "hello"
        assert c.index == 0
        assert c.metadata == {}
        assert c.embedding is None

    def test_with_metadata_and_embedding(self) -> None:
        c = DocumentChunk(
            content="world",
            index=1,
            metadata={"title": "Doc"},
            embedding=[0.1] * 1536,
        )
        assert len(c.embedding) == 1536


class TestIngestionResult:
    """IngestionResult model."""

    def test_ok_result(self) -> None:
        r = IngestionResult(
            document_id="uuid-1",
            title="Test",
            chunks_created=5,
            processing_time_ms=100.0,
        )
        assert r.errors == []

    def test_error_result(self) -> None:
        r = IngestionResult(
            document_id="",
            title="Fail",
            chunks_created=0,
            processing_time_ms=50.0,
            errors=["Something broke"],
        )
        assert len(r.errors) == 1


class TestChunker:
    """chunk_document behavior."""

    def test_empty_content(self) -> None:
        config = IngestionConfig(chunk_size=100, chunk_overlap=20)
        out = chunk_document("", "T", "s", config)
        assert out == []
        out = chunk_document("   \n\n  ", "T", "s", config)
        assert out == []

    def test_simple_chunking(self) -> None:
        config = IngestionConfig(chunk_size=100, chunk_overlap=20)
        content = "First paragraph here.\n\nSecond paragraph there.\n\nThird short."
        chunks = chunk_document(content, "Title", "source.md", config)
        assert len(chunks) >= 1
        for c in chunks:
            assert isinstance(c, DocumentChunk)
            assert c.content
            assert c.index >= 0
            assert "title" in c.metadata
            assert c.metadata["title"] == "Title"
            assert "chunk_method" in c.metadata
            assert "total_chunks" in c.metadata

    def test_concatenated_content_matches(self) -> None:
        config = IngestionConfig(chunk_size=100, chunk_overlap=10)
        content = "A" * 30 + "\n\n" + "B" * 30 + "\n\n" + "C" * 30
        chunks = chunk_document(content, "T", "s", config)
        combined = "\n\n".join(c.content for c in chunks)
        assert "A" in combined and "B" in combined and "C" in combined


class TestDocumentReader:
    """read_document and extract_title (text files only in unit test)."""

    def test_read_missing_file(self) -> None:
        with pytest.raises(FileNotFoundError):
            read_document("/nonexistent/path/file.md")

    def test_read_text_file(self, tmp_path: Path) -> None:
        f = tmp_path / "doc.md"
        f.write_text("# My Title\n\nBody here.", encoding="utf-8")
        content, docling_doc = read_document(str(f))
        assert "My Title" in content
        assert "Body here" in content
        assert docling_doc is None

    def test_extract_title_from_heading(self) -> None:
        text = "Other stuff\n\n# The Real Title\n\nBody"
        assert extract_title(text, "fallback.md") == "The Real Title"

    def test_extract_title_fallback_to_filename(self) -> None:
        text = "No heading here"
        assert extract_title(text, "/path/to/my-doc.md") == "my-doc"


class TestFindDocumentFiles:
    """find_document_files discovers supported files."""

    def test_find_files(self, tmp_path: Path) -> None:
        (tmp_path / "a.md").write_text("a")
        (tmp_path / "b.txt").write_text("b")
        (tmp_path / "skip.py").write_text("x")
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "c.md").write_text("c")
        from strategies.ingestion.ingest import find_document_files

        files = find_document_files(str(tmp_path))
        assert len(files) == 3
        assert any("a.md" in p for p in files)
        assert any("b.txt" in p for p in files)
        assert any("c.md" in p for p in files)
        assert not any("skip.py" in p for p in files)

    def test_nonexistent_dir(self) -> None:
        from strategies.ingestion.ingest import find_document_files

        files = find_document_files("/nonexistent/dir")
        assert files == []
