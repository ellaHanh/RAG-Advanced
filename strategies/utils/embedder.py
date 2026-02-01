"""
RAG-Advanced query embedder for vector search.

Uses OpenAI text-embedding-3-small (1536 dims) by default.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "text-embedding-3-small"
DEFAULT_DIMENSIONS = 1536


async def embed_query(
    text: str,
    model: str | None = None,
    dimensions: int | None = None,
) -> list[float]:
    """
    Generate embedding for a single query/text.

    Args:
        text: Text to embed.
        model: OpenAI embedding model (default from env or text-embedding-3-small).
        dimensions: Output dimensions (model-dependent).

    Returns:
        Embedding vector as list of floats.

    Raises:
        ValueError: If OPENAI_API_KEY is not set.
    """
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("OPENAI_API_KEY must be set for embeddings")

    model = model or os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL)
    dims = dimensions or int(os.getenv("EMBEDDING_DIMENSIONS", str(DEFAULT_DIMENSIONS)))

    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=key)
    kwargs: dict[str, Any] = {"model": model, "input": text}
    if model.startswith("text-embedding-3"):
        kwargs["dimensions"] = dims

    response = await client.embeddings.create(**kwargs)
    return list(response.data[0].embedding)


async def embed_documents(
    texts: list[str],
    model: str | None = None,
    dimensions: int | None = None,
    batch_size: int = 100,
) -> list[list[float]]:
    """
    Generate embeddings for multiple texts (batch API).

    Args:
        texts: List of texts to embed.
        model: OpenAI embedding model (default from env or text-embedding-3-small).
        dimensions: Output dimensions (model-dependent).
        batch_size: Max texts per API call (OpenAI limit is high; 100 is safe).

    Returns:
        List of embedding vectors in same order as texts.

    Raises:
        ValueError: If OPENAI_API_KEY is not set.
    """
    if not texts:
        return []
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("OPENAI_API_KEY must be set for embeddings")

    model = model or os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL)
    dims = dimensions or int(os.getenv("EMBEDDING_DIMENSIONS", str(DEFAULT_DIMENSIONS)))

    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=key)
    out: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        filled = [t if (t and t.strip()) else " " for t in batch]
        kwargs: dict[str, Any] = {"model": model, "input": filled}
        if model.startswith("text-embedding-3"):
            kwargs["dimensions"] = dims
        response = await client.embeddings.create(**kwargs)
        for d in response.data:
            out.append(list(d.embedding))
    return out
