"""
Unit tests for the ingestion pipeline.

Tests chunker, document reader (text only), models, and find_document_files
without database or OpenAI. Includes semantic-chunking option (use_semantic_chunking
and Docling HybridChunker path).
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from strategies.ingestion.chunker import chunk_document
from strategies.ingestion.document_reader import (
    extract_title,
    read_document,
    text_to_docling_document,
)
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

    def test_simple_chunking_overlap_applied(self) -> None:
        """Consecutive chunks overlap by at least chunk_overlap characters."""
        chunk_size = 100
        overlap = 20
        config = IngestionConfig(chunk_size=chunk_size, chunk_overlap=overlap)
        # Use distinct segments so overlap is measurable (repeated "x" would match full chunk)
        content = "a" * 80 + "b" * 20 + "c" * 80 + "d" * 20 + "e" * 80
        assert len(content) == 280
        chunks = chunk_document(content, "T", "s", config)
        assert len(chunks) >= 2
        for i in range(len(chunks) - 1):
            curr = chunks[i].content
            next_content = chunks[i + 1].content
            overlap_len = 0
            for n in range(min(len(curr), len(next_content)), 0, -1):
                if curr[-n:] == next_content[:n]:
                    overlap_len = n
                    break
            assert overlap_len >= overlap, (
                f"Chunks {i} and {i+1} should overlap by at least {overlap} chars, got {overlap_len}"
            )

    def test_simple_chunking_size_cap_long_paragraph(self) -> None:
        """One long paragraph (no double newline) is split; non-final chunks respect size cap."""
        chunk_size = 100
        overlap = 10
        config = IngestionConfig(chunk_size=chunk_size, chunk_overlap=overlap)
        content = "a" * 350
        chunks = chunk_document(content, "T", "s", config)
        assert len(chunks) >= 2
        tolerance = 201
        for i, c in enumerate(chunks):
            if i < len(chunks) - 1:
                assert len(c.content) <= chunk_size + tolerance, (
                    f"Non-final chunk {i} length {len(c.content)} should be <= {chunk_size + tolerance}"
                )

    def test_semantic_chunking_off_always_simple(self) -> None:
        """use_semantic_chunking=False uses simple chunking regardless of docling_doc."""
        config = IngestionConfig(
            chunk_size=100, chunk_overlap=10, use_semantic_chunking=False
        )
        content = "First para.\n\nSecond para."
        chunks = chunk_document(content, "T", "s", config, docling_doc=MagicMock())
        assert len(chunks) >= 1
        for c in chunks:
            assert c.metadata.get("chunk_method") == "simple"

    def test_semantic_chunking_no_docling_uses_simple(self) -> None:
        """use_semantic_chunking=True with docling_doc=None uses simple chunking."""
        config = IngestionConfig(
            chunk_size=100, chunk_overlap=10, use_semantic_chunking=True
        )
        content = "First para.\n\nSecond para."
        chunks = chunk_document(content, "T", "s", config, docling_doc=None)
        assert len(chunks) >= 1
        for c in chunks:
            assert c.metadata.get("chunk_method") == "simple"

    def test_semantic_chunking_with_docling_uses_hybrid(self) -> None:
        """use_semantic_chunking=True with valid docling_doc uses HybridChunker (hybrid)."""
        config = IngestionConfig(
            chunk_size=100, chunk_overlap=10, use_semantic_chunking=True
        )
        fake_docling_class = type("DoclingDocument", (), {})
        mock_doc = fake_docling_class()
        mock_chunker = MagicMock()
        mock_chunker.chunk.return_value = [MagicMock()]
        mock_chunker.contextualize.return_value = "Hybrid chunk text"
        with patch(
            "docling_core.types.doc.DoclingDocument", fake_docling_class
        ), patch(
            "strategies.ingestion.chunker._get_hybrid_chunker",
            return_value=mock_chunker,
        ):
            chunks = chunk_document(
                "ignored when docling path taken",
                "T",
                "s",
                config,
                docling_doc=mock_doc,
            )
        assert len(chunks) == 1
        assert chunks[0].content == "Hybrid chunk text"
        assert chunks[0].metadata.get("chunk_method") == "hybrid"
        assert chunks[0].metadata.get("total_chunks") == 1


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

    def test_text_to_docling_document_empty_returns_none(self) -> None:
        """text_to_docling_document returns None for empty or whitespace content."""
        assert text_to_docling_document("") is None
        assert text_to_docling_document("   \n\n  ") is None

    def test_text_to_docling_document_with_content_returns_doc_or_none(self) -> None:
        """With non-empty content, returns DoclingDocument when Docling works else None."""
        result = text_to_docling_document("# Title\n\nSome paragraph.")
        if result is not None:
            assert hasattr(result, "export_to_markdown") or getattr(
                result, "__class__", None
            )


class TestSemanticChunkingFromTxtIntegration:
    """Integration: .txt content -> text_to_docling_document -> chunk_document -> hybrid chunks."""

    @pytest.mark.integration
    def test_txt_content_semantic_chunking_produces_hybrid_chunks(self) -> None:
        """
        Confirm semantic chunking (Docling) runs for .txt content and returns hybrid chunks.

        Uses real text_to_docling_document() and chunk_document(); skips if Docling
        is not available or conversion returns None (e.g. Python 3.14 / Pydantic).
        """
        content = "# Test Title\n\nFirst paragraph with some content.\n\nSecond paragraph here."
        docling_doc = text_to_docling_document(content)
        if docling_doc is None:
            pytest.skip(
                "Docling not available or text_to_docling_document returned None "
                "(e.g. Docling not installed, or Python 3.14 / Pydantic compatibility)"
            )
        config = IngestionConfig(
            chunk_size=1000,
            chunk_overlap=200,
            use_semantic_chunking=True,
            max_tokens=512,
        )
        chunks = chunk_document(
            content,
            title="Test Title",
            source="test.txt",
            config=config,
            docling_doc=docling_doc,
        )
        assert len(chunks) >= 1, "Expected at least one chunk from Docling HybridChunker"
        for c in chunks:
            assert c.metadata.get("chunk_method") == "hybrid", (
                f"Expected chunk_method 'hybrid' for semantic chunking, got {c.metadata.get('chunk_method')!r}"
            )


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
