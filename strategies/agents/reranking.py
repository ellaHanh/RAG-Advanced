"""
Reranking strategy: two-stage retrieval (vector search + cross-encoder).

Aligned with all-rag-strategies: Stage 1 vector search for candidates,
Stage 2 cross-encoder (ms-marco-MiniLM-L-6-v2) rerank to top final_k.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Coroutine

from orchestration.errors import StrategyExecutionError
from orchestration.executor import ExecutionContext
from orchestration.models import Document

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "text-embedding-3-small"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_reranker: Any = None


def _get_reranker() -> Any:
    """Lazy-load cross-encoder (same as all-rag-strategies)."""
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder

        logger.info("Loading cross-encoder for re-ranking: %s", CROSS_ENCODER_MODEL)
        _reranker = CrossEncoder(CROSS_ENCODER_MODEL)
        logger.info("Cross-encoder loaded")
    return _reranker


async def _reranking_search_impl(
    ctx: ExecutionContext,
    pool: Any,
    embed_query_fn: Callable[[str], Coroutine[Any, Any, list[float]]],
) -> list[Document]:
    """Stage 1: vector search for initial_k candidates; Stage 2: cross-encoder rerank to final_k."""
    if pool is None:
        raise StrategyExecutionError(
            "Database not configured. Set DATABASE_URL to use this strategy.",
            details={"strategy": "reranking"},
        )
    query = ctx.query
    initial_k = ctx.config.initial_k
    final_k = ctx.config.final_k

    embedding = await embed_query_fn(query)
    token_count = max(1, len(query) // 4)
    ctx.add_embedding_cost(EMBEDDING_MODEL, token_count)

    embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, document_id, content, metadata, title, source, similarity
            FROM match_chunks($1::vector, $2)
            """,
            embedding_str,
            initial_k,
        )

    if not rows:
        return []

    reranker = _get_reranker()
    pairs = [[query, row["content"] or ""] for row in rows]
    scores = reranker.predict(pairs)

    reranked = sorted(
        zip(rows, scores),
        key=lambda x: x[1],
        reverse=True,
    )[:final_k]

    def _normalize_metadata(raw: Any) -> dict[str, Any]:
        if raw is None:
            return {}
        if isinstance(raw, dict):
            return dict(raw)
        if isinstance(raw, str):
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return {}
        return {}

    # Document.similarity must be in [0, 1]; cross-encoder scores can be any real number
    def _clamp_similarity(s: float) -> float:
        return max(0.0, min(1.0, float(s)))

    return [
        Document(
            id=str(row["id"]),
            content=row["content"] or "",
            title=row["title"] or "",
            source=row["source"] or "",
            similarity=_clamp_similarity(score),
            metadata=_normalize_metadata(row["metadata"]),
        )
        for row, score in reranked
    ]


def make_reranking_strategy(
    pool: Any,
    embed_query_fn: Callable[[str], Coroutine[Any, Any, list[float]]],
):
    """Return an async strategy function that closes over pool and embed_query_fn."""

    async def reranking_search(ctx: ExecutionContext) -> list[Document]:
        try:
            return await _reranking_search_impl(ctx, pool, embed_query_fn)
        except Exception as e:
            logger.exception("Reranking search failed: %s", e)
            raise

    return reranking_search
