"""
Multi-query RAG strategy.

Query expansion via LLM, then parallel vector search for each variation;
results are deduplicated by chunk ID and ranked by best similarity.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Callable, Coroutine

from orchestration.errors import StrategyExecutionError
from orchestration.executor import ExecutionContext
from orchestration.models import Document

from strategies.agents.query_utils import expand_query as expand_query_fn

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "text-embedding-3-small"
NUM_VARIATIONS_DEFAULT = 3


async def _multi_query_search_impl(
    ctx: ExecutionContext,
    pool: Any,
    embed_query_fn: Callable[[str], Coroutine[Any, Any, list[float]]],
) -> list[Document]:
    """
    Run multi-query retrieval: expand query, search in parallel, dedupe and rank.
    """
    if pool is None:
        raise StrategyExecutionError(
            "Database not configured. Set DATABASE_URL to use this strategy.",
            details={"strategy": "multi_query"},
        )
    query = ctx.query
    limit = ctx.config.limit
    num_variations = getattr(ctx.config, "num_variations", NUM_VARIATIONS_DEFAULT)
    if num_variations > 10:
        num_variations = 10

    queries = await expand_query_fn(query, num_variations, ctx)

    # Embed all queries (parallel)
    embeddings = await asyncio.gather(*[embed_query_fn(q) for q in queries])
    for q in queries:
        token_count = max(1, len(q) // 4)
        ctx.add_embedding_cost(EMBEDDING_MODEL, token_count)

    embedding_strs = ["[" + ",".join(str(x) for x in emb) + "]" for emb in embeddings]
    per_query_limit = max(limit, 10)

    async with pool.acquire() as conn:
        tasks = [
            conn.fetch(
                """
                SELECT id, document_id, content, metadata, title, source, similarity
                FROM match_chunks($1::vector, $2)
                """,
                emb_str,
                per_query_limit,
            )
            for emb_str in embedding_strs
        ]
        rows_list = await asyncio.gather(*tasks)

    # Deduplicate by chunk id, keep highest similarity
    seen: dict[Any, dict[str, Any]] = {}
    for rows in rows_list:
        for row in rows:
            r = dict(row) if hasattr(row, "keys") else row
            row_id = r.get("id")
            sim = float(r.get("similarity") or 0.0)
            if row_id not in seen or sim > float(seen[row_id].get("similarity") or 0):
                seen[row_id] = r

    # Sort by similarity descending and take top limit
    sorted_rows = sorted(
        seen.values(),
        key=lambda r: float(r.get("similarity") or 0),
        reverse=True,
    )[:limit]

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
            content=row.get("content") or "",
            title=row.get("title") or "",
            source=row.get("source") or "",
            similarity=float(row.get("similarity") or 0.0),
            metadata=_normalize_metadata(row.get("metadata")),
        )
        for row in sorted_rows
    ]


def make_multi_query_strategy(
    pool: Any,
    embed_query_fn: Callable[[str], Coroutine[Any, Any, list[float]]],
):
    """Return an async strategy function for multi-query RAG."""

    async def multi_query_search(ctx: ExecutionContext) -> list[Document]:
        try:
            return await _multi_query_search_impl(ctx, pool, embed_query_fn)
        except Exception as e:
            logger.exception("Multi-query search failed: %s", e)
            raise

    return multi_query_search
