"""
Agentic RAG strategy.

Runs standard vector search and adds full-document retrieval for the top result's
document (so the user gets chunks + full context of the top doc). No in-loop LLM
choice; this is a composite strategy.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Coroutine

from orchestration.errors import StrategyExecutionError
from orchestration.executor import ExecutionContext
from orchestration.models import Document

from strategies.agents.standard import _standard_search_impl

logger = logging.getLogger(__name__)


async def _agentic_search_impl(
    ctx: ExecutionContext,
    pool: Any,
    embed_query_fn: Callable[[str], Coroutine[Any, Any, list[float]]],
) -> list[Document]:
    """
    Standard search (limit 5) + fetch full document for top chunk's document_id;
    prepend full doc as first result (id = document_id, content = full doc content).
    """
    if pool is None:
        raise StrategyExecutionError(
            "Database not configured. Set DATABASE_URL to use this strategy.",
            details={"strategy": "agentic"},
        )

    chunks = await _standard_search_impl(ctx, pool, embed_query_fn)
    if not chunks:
        return []

    top_chunk_id = chunks[0].id
    async with pool.acquire() as conn:
        doc_id_row = await conn.fetchrow(
            "SELECT document_id FROM chunks WHERE id = $1::uuid",
            top_chunk_id,
        )
        if not doc_id_row:
            return chunks
        top_document_id = doc_id_row["document_id"]
        row = await conn.fetchrow(
            """
            SELECT id, title, source, content
            FROM documents
            WHERE id = $1
            """,
            top_document_id,
        )

    if not row:
        return chunks

    full_doc = Document(
        id=str(row["id"]),
        content=row["content"] or "",
        title=row["title"] or "",
        source=row["source"] or "",
        similarity=1.0,
        metadata={"full_document": True},
    )
    return [full_doc] + chunks


def make_agentic_strategy(
    pool: Any,
    embed_query_fn: Callable[[str], Coroutine[Any, Any, list[float]]],
):
    """Return an async strategy function for agentic RAG (chunks + full doc for top)."""

    async def agentic_search(ctx: ExecutionContext) -> list[Document]:
        try:
            return await _agentic_search_impl(ctx, pool, embed_query_fn)
        except Exception as e:
            logger.exception("Agentic search failed: %s", e)
            raise

    return agentic_search
