"""
Self-reflective RAG strategy.

Search → grade relevance (LLM) → if below threshold, refine query (LLM) → search again.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Callable, Coroutine

from orchestration.errors import StrategyExecutionError
from orchestration.executor import ExecutionContext
from orchestration.models import Document

from strategies.agents.query_utils import parse_grade_from_llm

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "text-embedding-3-small"
GRADE_MODEL = "gpt-4o-mini"
REFINE_MODEL = "gpt-4o-mini"
RELEVANCE_THRESHOLD_DEFAULT = 3.0


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


async def _self_reflective_search_impl(
    ctx: ExecutionContext,
    pool: Any,
    embed_query_fn: Callable[[str], Coroutine[Any, Any, list[float]]],
) -> list[Document]:
    """
    Initial search → grade (LLM) → if grade < threshold, refine query (LLM) → search again.
    """
    if pool is None:
        raise StrategyExecutionError(
            "Database not configured. Set DATABASE_URL to use this strategy.",
            details={"strategy": "self_reflective"},
        )
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    query = ctx.query
    limit = ctx.config.limit
    threshold = getattr(ctx.config, "relevance_threshold", RELEVANCE_THRESHOLD_DEFAULT)

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
            limit,
        )

    if not rows:
        return []

    # Grade relevance
    docs_preview = "\n".join(
        f"{i+1}. {(r['content'] or '')[:200]}..." for i, r in enumerate(rows)
    )
    grade_prompt = f"""Query: {query}

Retrieved Documents:
{docs_preview}

Grade the overall relevance of these documents to the query on a scale of 1-5:
1 = Not relevant at all
2 = Slightly relevant
3 = Moderately relevant
4 = Relevant
5 = Highly relevant

Respond with only a single number (1-5) and optionally a brief reason."""

    try:
        grade_response = await client.chat.completions.create(
            model=GRADE_MODEL,
            messages=[{"role": "user", "content": grade_prompt}],
            temperature=0,
        )
        grade_text = (grade_response.choices[0].message.content or "").strip()
        grade_score = parse_grade_from_llm(grade_text)
        in_tok = max(1, len(grade_prompt) // 4)
        out_tok = max(1, len(grade_text) // 4)
        ctx.add_llm_cost(GRADE_MODEL, in_tok, out_tok)
    except Exception as e:
        logger.warning("Grading failed, proceeding with results: %s", e)
        grade_score = 3

    if grade_score < threshold:
        refine_prompt = f"""The query "{query}" returned low-relevance results.
Suggest an improved, more specific query that might find better results.
Respond with only the improved query, nothing else."""

        try:
            refine_response = await client.chat.completions.create(
                model=REFINE_MODEL,
                messages=[{"role": "user", "content": refine_prompt}],
                temperature=0.7,
            )
            refined_query = (refine_response.choices[0].message.content or "").strip()
            in_tok = max(1, len(refine_prompt) // 4)
            out_tok = max(1, len(refined_query) // 4)
            ctx.add_llm_cost(REFINE_MODEL, in_tok, out_tok)

            refined_embedding = await embed_query_fn(refined_query)
            ctx.add_embedding_cost(EMBEDDING_MODEL, max(1, len(refined_query) // 4))
            refined_emb_str = "[" + ",".join(str(x) for x in refined_embedding) + "]"

            async with pool.acquire() as conn2:
                rows = await conn2.fetch(
                    """
                    SELECT id, document_id, content, metadata, title, source, similarity
                    FROM match_chunks($1::vector, $2)
                    """,
                    refined_emb_str,
                    limit,
                )
        except Exception as e:
            logger.warning("Refinement failed, keeping initial results: %s", e)

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


def make_self_reflective_strategy(
    pool: Any,
    embed_query_fn: Callable[[str], Coroutine[Any, Any, list[float]]],
):
    """Return an async strategy function for self-reflective RAG."""

    async def self_reflective_search(ctx: ExecutionContext) -> list[Document]:
        try:
            return await _self_reflective_search_impl(ctx, pool, embed_query_fn)
        except Exception as e:
            logger.exception("Self-reflective search failed: %s", e)
            raise

    return self_reflective_search
