"""
Shared utilities for query expansion and LLM-based strategies.
"""

from __future__ import annotations

import logging
import os
import re

from orchestration.executor import ExecutionContext

logger = logging.getLogger(__name__)

EXPANSION_MODEL = "gpt-4o-mini"


async def expand_query(
    query: str,
    num_variations: int,
    ctx: ExecutionContext,
) -> list[str]:
    """
    Generate query variations using an LLM.

    Args:
        query: Original search query.
        num_variations: Number of variations to generate (excluding original).
        ctx: Execution context for cost tracking.

    Returns:
        List of queries: [original] + up to num_variations variations.
    """
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    prompt = f"""Generate {num_variations} different variations of this search query.
Each variation should capture a different perspective or phrasing while maintaining the same intent.

Original query: {query}

Return only the {num_variations} variations, one per line, without numbers or bullets."""

    try:
        response = await client.chat.completions.create(
            model=EXPANSION_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        content = response.choices[0].message.content or ""
        variations = [v.strip() for v in content.split("\n") if v.strip()][:num_variations]
        in_tokens = max(1, len(prompt) // 4)
        out_tokens = max(1, len(content) // 4)
        ctx.add_llm_cost(EXPANSION_MODEL, in_tokens, out_tokens)
        return [query] + variations
    except Exception as e:
        logger.warning("Query expansion failed, using original only: %s", e)
        return [query]


def parse_grade_from_llm(content: str) -> int:
    """
    Parse a 1-5 grade from LLM response. Tolerates extra text.

    Returns:
        Integer 1-5, or 3 if unparseable.
    """
    content = content.strip()
    match = re.search(r"\b([1-5])\b", content)
    if match:
        return int(match.group(1))
    return 3
