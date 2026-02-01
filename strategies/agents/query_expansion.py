"""
Query expansion strategy.

Expand query via LLM (one variation), then run standard vector search with the
expanded query. Cheaper than multi_query (one embed + one search).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Coroutine

from orchestration.errors import StrategyExecutionError
from orchestration.executor import ExecutionContext
from orchestration.models import Document

from strategies.agents.query_utils import expand_query as expand_query_fn

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "text-embedding-3-small"
NUM_VARIATIONS_DEFAULT = 1


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


async def _query_expansion_search_impl(
    ctx: ExecutionContext,
    pool: Any,
    embed_query_fn: Callable[[str], Coroutine[Any, Any, list[float]]],
) -> list[Document]:
    """
    Expand query (LLM), then run standard search with the first expanded variation.
    """
    if pool is None:
        raise StrategyExecutionError(
            "Database not configured. Set DATABASE_URL to use this strategy.",
            details={"strategy": "query_expansion"},
        )
    query = ctx.query
    limit = ctx.config.limit
    num_variations = getattr(ctx.config, "num_variations", NUM_VARIATIONS_DEFAULT)
    if num_variations < 1:
        num_variations = 1

    queries = await expand_query_fn(query, num_variations, ctx)
    search_query = queries[1] if len(queries) > 1 else queries[0]

    embedding = await embed_query_fn(search_query)
    token_count = max(1, len(search_query) // 4)
    ctx.add_embedding_cost(EMBEDDING_MODEL, token_count)
    embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, document_id, content, metadata, title, source, similarity
            FROM match_chunks($1::vector, $2)
            """,
            embedding_str,
            limit,
        )

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


def make_query_expansion_strategy(
    pool: Any,
    embed_query_fn: Callable[[str], Coroutine[Any, Any, list[float]]],
):
    """Return an async strategy function for query expansion (expand then single search)."""

    async def query_expansion_search(ctx: ExecutionContext) -> list[Document]:
        try:
            return await _query_expansion_search_impl(ctx, pool, embed_query_fn)
        except Exception as e:
            logger.exception("Query expansion search failed: %s", e)
            raise

    return query_expansion_search
