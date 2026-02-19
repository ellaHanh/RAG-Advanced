"""
Standard semantic search strategy.

Vector search using query embedding and match_chunks in PostgreSQL.
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


async def _standard_search_impl(
    ctx: ExecutionContext,
    pool: Any,
    embed_query_fn: Callable[[str], Coroutine[Any, Any, list[float]]],
) -> list[Document]:
    """Core implementation: embed query, then match_chunks."""
    if pool is None:
        raise StrategyExecutionError(
            "Database not configured. Set DATABASE_URL to use this strategy.",
            details={"strategy": "standard"},
        )
    query = ctx.query
    limit = ctx.config.limit

    embedding = await embed_query_fn(query)
    token_count = max(1, len(query) // 4)
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
            limit,
        )

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

    return [
        Document(
            id=str(row["id"]),
            content=row["content"] or "",
            title=row["title"] or "",
            source=row["source"] or "",
            similarity=float(row["similarity"]) if row["similarity"] is not None else 0.0,
            metadata=_normalize_metadata(row["metadata"]),
        )
        for row in rows
    ]


def make_standard_strategy(
    pool: Any,
    embed_query_fn: Callable[[str], Coroutine[Any, Any, list[float]]],
):
    """Return an async strategy function that closes over pool and embed_query_fn."""

    async def standard_search(ctx: ExecutionContext) -> list[Document]:
        try:
            return await _standard_search_impl(ctx, pool, embed_query_fn)
        except Exception as e:
            logger.exception("Standard search failed: %s", e)
            raise

    return standard_search
