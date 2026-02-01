"""
Ingestion embedder: batch-embed document chunks using RAG-Advanced embedder.
"""

from __future__ import annotations

import logging

from strategies.ingestion.models import DocumentChunk
from strategies.utils.embedder import embed_documents

logger = logging.getLogger(__name__)


async def embed_chunks(chunks: list[DocumentChunk], batch_size: int = 100) -> list[DocumentChunk]:
    """
    Attach embeddings to chunks in place and return the same list.

    Args:
        chunks: Chunks with content; embedding will be set.
        batch_size: Max texts per OpenAI API call.

    Returns:
        The same chunks with embedding field set (1536 dims).
    """
    if not chunks:
        return chunks
    texts = [c.content for c in chunks]
    vectors = await embed_documents(texts, batch_size=batch_size)
    for c, vec in zip(chunks, vectors):
        c.embedding = vec
    logger.info("Embedded %d chunks", len(chunks))
    return chunks
