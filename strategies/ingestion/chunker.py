"""
Document chunking for ingestion.

Aligned with all-rag-strategies: Docling HybridChunker when a DoclingDocument is
available (PDF/DOCX etc.); otherwise simple paragraph/size-based chunking. Reuses
a single tokenizer and HybridChunker per max_tokens to avoid re-loading per document.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from strategies.ingestion.models import DocumentChunk, IngestionConfig

logger = logging.getLogger(__name__)

# Lazy-initialized Docling chunker (one per max_tokens) to match all-rag-strategies reuse
_hybrid_chunker_cache: dict[int, Any] = {}
_TOKENIZER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # same as all-rag-strategies

# Max characters to look back for sentence boundary in simple chunker (no config field)
_SIMPLE_CHUNK_SENTENCE_LOOKBACK = 200


def chunk_document(
    content: str,
    title: str,
    source: str,
    config: IngestionConfig,
    metadata: dict[str, Any] | None = None,
    docling_doc: Any = None,
) -> list[DocumentChunk]:
    """
    Chunk document content into overlapping segments.

    Uses Docling HybridChunker when docling_doc is provided and config.use_semantic_chunking
    is True; otherwise uses simple paragraph/size-based chunking.

    Args:
        content: Full document text (markdown or plain).
        title: Document title for metadata.
        source: Document source path for metadata.
        config: Chunk size, overlap, and semantic flag.
        metadata: Extra metadata to attach to each chunk.
        docling_doc: Optional DoclingDocument for structure-aware chunking.

    Returns:
        List of DocumentChunk (content, index, metadata; no embedding).
    """
    if not (content or "").strip():
        return []

    base_metadata: dict[str, Any] = {
        "title": title,
        "source": source,
        **(metadata or {}),
    }

    if config.use_semantic_chunking and docling_doc is not None:
        chunks = _chunk_with_docling(docling_doc, base_metadata, max_tokens=config.max_tokens)
        if chunks:
            return chunks
        logger.warning("Docling chunking returned nothing; falling back to simple chunking")

    return _chunk_simple(content, config, base_metadata)


def _chunk_simple(
    content: str,
    config: IngestionConfig,
    base_metadata: dict[str, Any],
) -> list[DocumentChunk]:
    """
    Character-based sliding window chunking with overlap and optional sentence-boundary break.

    Enforces chunk_size (with optional break at .!?\\n within lookback), applies chunk_overlap
    between consecutive chunks, and always advances so long paragraphs are split.
    """
    size = config.chunk_size
    overlap = config.chunk_overlap
    chunks: list[DocumentChunk] = []
    chunk_index = 0
    start = 0
    n = len(content)

    while start < n:
        end = start + size

        if end >= n:
            chunk_text = content[start:]
            if chunk_text.strip():
                chunks.append(
                    DocumentChunk(
                        content=chunk_text.strip(),
                        index=chunk_index,
                        metadata={**base_metadata, "chunk_method": "simple"},
                    )
                )
                chunk_index += 1
            break
        end_inclusive = end - 1
        lookback_start = max(start, end - _SIMPLE_CHUNK_SENTENCE_LOOKBACK)
        chunk_end = end
        for i in range(end_inclusive, lookback_start - 1, -1):
            if i < n and content[i] in ".!?\n":
                chunk_end = i + 1
                break
        chunk_text = content[start:chunk_end]

        if chunk_text.strip():
            chunks.append(
                DocumentChunk(
                    content=chunk_text.strip(),
                    index=chunk_index,
                    metadata={**base_metadata, "chunk_method": "simple"},
                )
            )
            chunk_index += 1

        start = max(start + 1, chunk_end - overlap) if overlap > 0 else chunk_end

    for c in chunks:
        c.metadata["total_chunks"] = len(chunks)
    logger.info("Created %d chunks (simple)", len(chunks))
    return chunks


def _get_hybrid_chunker(max_tokens: int) -> Any:
    """Return a Docling HybridChunker for the given max_tokens (cached per max_tokens)."""
    if max_tokens not in _hybrid_chunker_cache:
        from docling.chunking import HybridChunker
        from transformers import AutoTokenizer

        logger.info("Initializing Docling HybridChunker (tokenizer=%s, max_tokens=%d)", _TOKENIZER_MODEL, max_tokens)
        tokenizer = AutoTokenizer.from_pretrained(_TOKENIZER_MODEL)
        _hybrid_chunker_cache[max_tokens] = HybridChunker(
            tokenizer=tokenizer,
            max_tokens=max_tokens,
            merge_peers=True,
        )
    return _hybrid_chunker_cache[max_tokens]


def _chunk_with_docling(
    docling_doc: Any,
    base_metadata: dict[str, Any],
    max_tokens: int = 512,
) -> list[DocumentChunk]:
    """Use Docling HybridChunker (same API as all-rag-strategies DoclingHybridChunker)."""
    try:
        from docling_core.types.doc import DoclingDocument

        if not isinstance(docling_doc, DoclingDocument):
            return []
        chunker = _get_hybrid_chunker(max_tokens)
        chunk_iter = chunker.chunk(dl_doc=docling_doc)
        chunks_list = list(chunk_iter)
        result: list[DocumentChunk] = []
        for i, ch in enumerate(chunks_list):
            text = chunker.contextualize(chunk=ch).strip()
            if not text:
                continue
            result.append(
                DocumentChunk(
                    content=text,
                    index=i,
                    metadata={**base_metadata, "chunk_method": "hybrid", "total_chunks": len(chunks_list)},
                )
            )
        if result:
            logger.info("Created %d chunks (Docling HybridChunker)", len(result))
        return result
    except Exception as e:
        logger.warning("Docling HybridChunker failed: %s", e)
        return []
