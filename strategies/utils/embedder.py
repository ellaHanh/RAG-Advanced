"""
RAG-Advanced embedder for vector search and ingestion.

Supports two backends:
- openai: OpenAI text-embedding-3-small (1536 dims); requires OPENAI_API_KEY.
- bge-m3: BAAI/bge-m3 via sentence-transformers (1024 dims), CPU-friendly for local dev
  and BioASQ; no API key. Use schema_1024.sql (pgvector vector(1024)) with this backend.

Set EMBEDDING_BACKEND=openai (default) or bge-m3. pgvector is used for storage in both cases.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from typing import Any

logger = logging.getLogger(__name__)

_bge_m3_lock = threading.Lock()

# OpenAI default (existing behavior)
DEFAULT_MODEL = "text-embedding-3-small"
DEFAULT_DIMENSIONS = 1536

# BGE-M3 (sentence-transformers, CPU-optimized)
BGE_M3_MODEL = "BAAI/bge-m3"
BGE_M3_DIMENSIONS = 1024


def _get_backend() -> str:
    """Return embedding backend: openai or bge-m3 (case-insensitive)."""
    return (os.getenv("EMBEDDING_BACKEND") or "openai").strip().lower()


def get_embedding_dimensions() -> int:
    """
    Return the embedding dimension for the active backend.

    Use this to ensure schema (pgvector column size) matches:
    - openai: 1536 (or EMBEDDING_DIMENSIONS); use strategies/utils/schema.sql.
    - bge-m3: 1024; use strategies/utils/schema_1024.sql.
    """
    backend = _get_backend()
    if backend == "bge-m3":
        return BGE_M3_DIMENSIONS
    return int(os.getenv("EMBEDDING_DIMENSIONS", str(DEFAULT_DIMENSIONS)))


# -----------------------------------------------------------------------------
# OpenAI backend
# -----------------------------------------------------------------------------


async def _embed_query_openai(
    text: str,
    model: str | None = None,
    dimensions: int | None = None,
) -> list[float]:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("OPENAI_API_KEY must be set for OpenAI embeddings")
    model = model or os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL)
    dims = dimensions or int(os.getenv("EMBEDDING_DIMENSIONS", str(DEFAULT_DIMENSIONS)))
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=key)
    kwargs: dict[str, Any] = {"model": model, "input": text}
    if model.startswith("text-embedding-3"):
        kwargs["dimensions"] = dims
    response = await client.embeddings.create(**kwargs)
    return list(response.data[0].embedding)


async def _embed_documents_openai(
    texts: list[str],
    model: str | None = None,
    dimensions: int | None = None,
    batch_size: int = 100,
) -> list[list[float]]:
    if not texts:
        return []
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("OPENAI_API_KEY must be set for OpenAI embeddings")
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


# -----------------------------------------------------------------------------
# BGE-M3 backend (sentence-transformers, CPU)
# -----------------------------------------------------------------------------

_bge_m3_model: Any = None


def _get_bge_m3_model() -> Any:
    """Lazy-load BGE-M3 model (CPU by default for local dev).

    Thread-safe: uses a lock so concurrent callers don't trigger
    duplicate downloads / loads.
    """
    global _bge_m3_model
    if _bge_m3_model is not None:
        return _bge_m3_model
    with _bge_m3_lock:
        if _bge_m3_model is not None:
            return _bge_m3_model
        from sentence_transformers import SentenceTransformer
        device = os.getenv("EMBEDDING_DEVICE", "cpu")
        model_kwargs = {"low_cpu_mem_usage": False}
        logger.info("Loading BGE-M3 model (device=%s) — this may take a few minutes …", device)
        _bge_m3_model = SentenceTransformer(
            BGE_M3_MODEL,
            device=device,
            model_kwargs=model_kwargs,
        )
        logger.info("Loaded BGE-M3 embedder (device=%s, dims=%d)", device, BGE_M3_DIMENSIONS)
    return _bge_m3_model


def _encode_bge_m3_sync(texts: list[str]) -> list[list[float]]:
    """Sync encode with BGE-M3; run from thread pool."""
    model = _get_bge_m3_model()
    import numpy as np
    arr = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    if isinstance(arr, np.ndarray) and arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return [list(row) for row in arr]


async def _embed_query_bge_m3(text: str) -> list[float]:
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _encode_bge_m3_sync, [text or " "])
    return result[0]


async def _embed_documents_bge_m3(
    texts: list[str],
    batch_size: int = 64,
) -> list[list[float]]:
    if not texts:
        return []
    filled = [t if (t and t.strip()) else " " for t in texts]
    loop = asyncio.get_event_loop()
    out: list[list[float]] = []
    for i in range(0, len(filled), batch_size):
        batch = filled[i : i + batch_size]
        batch_result = await loop.run_in_executor(None, _encode_bge_m3_sync, batch)
        out.extend(batch_result)
    return out


# -----------------------------------------------------------------------------
# Public API (backend-dispatched)
# -----------------------------------------------------------------------------


async def embed_query(
    text: str,
    model: str | None = None,
    dimensions: int | None = None,
) -> list[float]:
    """
    Generate embedding for a single query/text.

    Backend is chosen by EMBEDDING_BACKEND (openai | bge-m3).
    """
    if _get_backend() == "bge-m3":
        return await _embed_query_bge_m3(text)
    return await _embed_query_openai(text, model=model, dimensions=dimensions)


async def embed_documents(
    texts: list[str],
    model: str | None = None,
    dimensions: int | None = None,
    batch_size: int = 100,
) -> list[list[float]]:
    """
    Generate embeddings for multiple texts (batch).

    Backend is chosen by EMBEDDING_BACKEND. For bge-m3, batch_size applies
    to local batching (default 64 for CPU memory).
    """
    if not texts:
        return []
    if _get_backend() == "bge-m3":
        return await _embed_documents_bge_m3(texts, batch_size=min(batch_size, 64))
    return await _embed_documents_openai(
        texts, model=model, dimensions=dimensions, batch_size=batch_size
    )


async def warmup_embedder() -> None:
    """Pre-load the embedding model so the first request doesn't time out.

    For bge-m3, downloads (~2.2 GB) and loads the SentenceTransformer model
    into memory.  For openai, this is a no-op (no local model to load).

    Call this during application startup (e.g. in FastAPI lifespan) so the
    model is ready before the API accepts traffic.
    """
    backend = _get_backend()
    if backend != "bge-m3":
        logger.info("Embedder warmup: backend=%s (no local model to pre-load)", backend)
        return
    logger.info("Embedder warmup: pre-loading BGE-M3 model …")
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _get_bge_m3_model)
    logger.info("Embedder warmup: BGE-M3 model ready")
