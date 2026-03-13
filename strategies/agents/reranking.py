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
        # Avoid "meta tensor" load path in newer PyTorch/transformers (e.g. in Docker)
        _reranker = CrossEncoder(
            CROSS_ENCODER_MODEL,
            model_kwargs={"low_cpu_mem_usage": False},
        )
        logger.info("Cross-encoder loaded")
    return _reranker


def _normalize_metadata(raw: Any) -> dict[str, Any]:
    """Normalize metadata from DB or dict."""
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


def _clamp_similarity(s: float) -> float:
    """Clamp similarity to [0, 1] for Document."""
    return max(0.0, min(1.0, float(s)))


async def _reranking_search_impl(
    ctx: ExecutionContext,
    pool: Any,
    embed_query_fn: Callable[[str], Coroutine[Any, Any, list[float]]],
) -> list[Document]:
    """
    Stage 1: vector search for initial_k candidates; Stage 2: cross-encoder rerank to final_k.
    When ctx.input_documents is set (e.g. from previous chain step), skip Stage 1 and only rerank.
    """
    query = ctx.original_query if ctx.original_query is not None else ctx.query
    final_k = ctx.config.final_k

    # Rerank-only mode: use documents from previous step (no retrieval)
    if ctx.input_documents:
        candidates = ctx.input_documents
        reranker = _get_reranker()
        pairs = [[query, doc.content or ""] for doc in candidates]
        scores = reranker.predict(pairs)
        scored_docs = sorted(
            zip(candidates, scores),
            key=lambda x: x[1],
            reverse=True,
        )[:final_k]
        return [
            Document(
                id=doc.id,
                content=doc.content,
                title=doc.title or "",
                source=doc.source or "",
                similarity=_clamp_similarity(score),
                metadata=dict(doc.metadata) if doc.metadata else {},
            )
            for doc, score in scored_docs
        ]

    # Full mode: Stage 1 vector search + Stage 2 rerank
    if pool is None:
        raise StrategyExecutionError(
            "Database not configured. Set DATABASE_URL to use this strategy.",
            details={"strategy": "reranking"},
        )
    initial_k = ctx.config.initial_k

    embedding = await embed_query_fn(ctx.query)
    token_count = max(1, len(ctx.query) // 4)
    ctx.add_embedding_cost(EMBEDDING_MODEL, token_count)

    embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"

    async with pool.acquire() as conn:
        await conn.execute("SET LOCAL ivfflat.probes = 10")
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
