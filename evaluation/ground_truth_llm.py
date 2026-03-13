"""
RAG-Advanced LLM-Assisted Ground Truth Generation.

Use an LLM to judge relevance of candidate documents for a query and produce
ground truth (relevant_doc_ids, relevance_scores) compatible with DatasetQuery.

Usage:
    from evaluation.ground_truth_llm import (
        generate_ground_truth_for_query,
        enrich_dataset_with_llm,
    )

    relevant_ids, scores = await generate_ground_truth_for_query(
        query="What is RAG?",
        candidates=[{"id": "doc1", "snippet": "RAG is retrieval-augmented..."}],
    )

    enriched = await enrich_dataset_with_llm(
        dataset,
        candidate_provider=async_get_candidates_for_query,
    )
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, Coroutine

from evaluation.datasets import Dataset, DatasetQuery

logger = logging.getLogger(__name__)

# Type for async candidate provider: (query: str) -> list of (doc_id, snippet)
CandidateProvider = Callable[[str], Coroutine[Any, Any, list[tuple[str, str]]]]


@dataclass
class CandidateDoc:
    """A candidate document for relevance judgment."""

    id: str
    snippet: str

    def to_tuple(self) -> tuple[str, str]:
        """Return (id, snippet) for provider compatibility."""
        return (self.id, self.snippet)


# Default model for relevance judgment (cheap, fast).
DEFAULT_LLM_MODEL = "gpt-4o-mini"

# Prompt instructing the LLM to return JSON with relevant_ids and optional relevance_scores.
_RELEVANCE_SYSTEM = """You are an expert at judging relevance for information retrieval evaluation.
Given a search query and a list of candidate documents (each with an id and a short snippet),
output which documents are relevant to the query and optionally assign a relevance score per document.
Relevance scores: 0 = not relevant, 1 = partially relevant, 2 = highly relevant.
Output valid JSON only, no markdown or extra text."""

_RELEVANCE_USER_TEMPLATE = """Query: {query}

Candidates (id, snippet):
{candidates_text}

Return a JSON object with:
- "relevant_ids": list of document ids that are relevant (at least partially)
- "relevance_scores": optional object mapping document id to 0, 1, or 2 (default 1 if omitted for relevant_ids)

Example: {{"relevant_ids": ["doc1", "doc3"], "relevance_scores": {{"doc1": 2, "doc3": 1}}}}
"""


async def generate_ground_truth_for_query(
    query: str,
    candidates: list[tuple[str, str]] | list[dict[str, str]] | list[CandidateDoc],
    *,
    model: str = DEFAULT_LLM_MODEL,
    api_key: str | None = None,
) -> tuple[list[str], dict[str, int]]:
    """
    Use an LLM to judge relevance of candidates for a query.

    Args:
        query: The search query.
        candidates: List of (doc_id, snippet) or dicts with "id"/"snippet" or CandidateDoc.
        model: OpenAI model name for completion.
        api_key: OpenAI API key (defaults to OPENAI_API_KEY env).

    Returns:
        (relevant_doc_ids, relevance_scores). Scores are 0, 1, or 2 per doc.
        If the LLM call fails, returns ([], {}).

    Raises:
        ValueError: If api_key is missing and OPENAI_API_KEY is not set.
    """
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("OPENAI_API_KEY must be set or pass api_key for LLM ground truth")

    # Normalize candidates to (id, snippet)
    normalized: list[tuple[str, str]] = []
    for c in candidates:
        if isinstance(c, tuple) and len(c) == 2:
            normalized.append((str(c[0]), str(c[1])))
        elif isinstance(c, dict):
            normalized.append((str(c.get("id", "")), str(c.get("snippet", ""))))
        elif isinstance(c, CandidateDoc):
            normalized.append((c.id, c.snippet))
        else:
            logger.warning("Skipping invalid candidate: %s", type(c))

    if not normalized:
        return ([], {})

    candidates_text = "\n".join(f"- {doc_id}: {snippet[:500]}" for doc_id, snippet in normalized)
    user_content = _RELEVANCE_USER_TEMPLATE.format(
        query=query,
        candidates_text=candidates_text,
    )

    try:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=key)
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _RELEVANCE_SYSTEM},
                {"role": "user", "content": user_content},
            ],
            temperature=0.0,
            max_tokens=1024,
        )
        content = response.choices[0].message.content
        if not content or not content.strip():
            return ([], {})

        # Parse JSON (allow markdown code block wrapper)
        text = content.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(
                line for line in lines if not line.startswith("```") and line.strip()
            )
        data = json.loads(text)

        relevant_ids: list[str] = list(data.get("relevant_ids", []))
        relevance_scores: dict[str, int] = {}
        raw_scores = data.get("relevance_scores") or {}
        for doc_id, score in raw_scores.items():
            if isinstance(score, int) and 0 <= score <= 2:
                relevance_scores[str(doc_id)] = score
        for doc_id in relevant_ids:
            if doc_id not in relevance_scores:
                relevance_scores[doc_id] = 1
        return (relevant_ids, relevance_scores)

    except json.JSONDecodeError as e:
        logger.warning("LLM ground truth JSON parse failed: %s", e)
        return ([], {})
    except Exception as e:
        logger.exception("LLM ground truth call failed: %s", e)
        return ([], {})


async def enrich_dataset_with_llm(
    dataset: Dataset,
    candidate_provider: CandidateProvider,
    *,
    model: str = DEFAULT_LLM_MODEL,
    api_key: str | None = None,
    only_missing: bool = True,
) -> Dataset:
    """
    Enrich a dataset with LLM-generated ground truth for each query.

    For each query, calls candidate_provider(query) to get (doc_id, snippet)
    candidates, then uses the LLM to produce relevant_doc_ids and relevance_scores.
    Results are merged into new DatasetQuery objects compatible with Dataset.

    Args:
        dataset: Source dataset (queries may have empty or partial ground truth).
        candidate_provider: Async function (query: str) -> list[(doc_id, snippet)].
        model: OpenAI model for relevance judgment.
        api_key: OpenAI API key (optional).
        only_missing: If True, only enrich queries with empty relevant_doc_ids.

    Returns:
        New Dataset with enriched ground truth. Queries skipped (e.g. already
        have ground truth when only_missing=True) keep original data.
    """
    enriched_queries: list[DatasetQuery] = []

    for q in dataset.queries:
        if only_missing and q.relevant_doc_ids:
            enriched_queries.append(q)
            continue

        try:
            candidates = await candidate_provider(q.query)
            if not candidates:
                enriched_queries.append(q)
                continue

            relevant_ids, relevance_scores = await generate_ground_truth_for_query(
                q.query,
                candidates,
                model=model,
                api_key=api_key,
            )

            new_query = DatasetQuery(
                query_id=q.query_id,
                query=q.query,
                relevant_doc_ids=relevant_ids if relevant_ids else q.relevant_doc_ids,
                relevance_scores=relevance_scores if relevance_scores else q.relevance_scores,
                category=q.category,
                metadata={**q.metadata, "llm_ground_truth": True},
            )
            enriched_queries.append(new_query)

        except Exception as e:
            logger.warning("Enrich failed for query_id=%s: %s", q.query_id, e)
            enriched_queries.append(q)

    return Dataset(
        name=f"{dataset.name}_llm_enriched",
        description=dataset.description + " (LLM-enriched ground truth)",
        queries=enriched_queries,
        metadata={**dataset.metadata, "llm_enriched": True},
    )
