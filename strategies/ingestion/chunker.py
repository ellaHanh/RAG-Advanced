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
    """Paragraph-aware sliding window chunking."""
    size = config.chunk_size
    overlap = config.chunk_overlap
    paragraphs = re.split(r"\n\s*\n", content)
    chunks: list[DocumentChunk] = []
    current: list[str] = []
    current_len = 0
    chunk_index = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        para_len = len(para) + 2  # +2 for "\n\n"
        if current_len + para_len <= size and current:
            current.append(para)
            current_len += para_len
        else:
            if current:
                text = "\n\n".join(current)
                if text.strip():
                    chunks.append(
                        DocumentChunk(
                            content=text.strip(),
                            index=chunk_index,
                            metadata={**base_metadata, "chunk_method": "simple"},
                        )
                    )
                    chunk_index += 1
            current = [para]
            current_len = para_len

    if current:
        text = "\n\n".join(current)
        if text.strip():
            chunks.append(
                DocumentChunk(
                    content=text.strip(),
                    index=chunk_index,
                    metadata={**base_metadata, "chunk_method": "simple"},
                )
            )

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
