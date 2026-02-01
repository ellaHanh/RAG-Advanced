"""
RAG Strategy Agents.

Strategy implementations for the orchestration layer.
Register strategies at app startup via register_all_strategies(pool, embed_query_fn).
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Coroutine

from orchestration.models import ResourceType, StrategyMetadata, StrategyType
from orchestration.registry import get_registry

from strategies.agents.agentic import make_agentic_strategy
from strategies.agents.multi_query import make_multi_query_strategy
from strategies.agents.query_expansion import make_query_expansion_strategy
from strategies.agents.reranking import make_reranking_strategy
from strategies.agents.self_reflective import make_self_reflective_strategy
from strategies.agents.standard import make_standard_strategy

logger = logging.getLogger(__name__)


def register_all_strategies(
    pool: Any,
    embed_query_fn: Callable[[str], Coroutine[Any, Any, list[float]]],
) -> None:
    """
    Register all RAG strategies with the global registry.

    Call this at app startup (pool may be None if DATABASE_URL is not set;
    strategies will raise a clear error when executed without a pool).

    Args:
        pool: asyncpg connection pool (with acquire() context manager), or None.
        embed_query_fn: Async function (query: str) -> list[float] for embeddings.
    """
    registry = get_registry()

    standard = make_standard_strategy(pool, embed_query_fn)
    registry.register(
        "standard",
        standard,
        StrategyMetadata(
            name="standard",
            description="Semantic vector search using query embedding and pgvector match_chunks",
            strategy_type=StrategyType.STANDARD,
            version="1.0.0",
            required_resources=[ResourceType.DATABASE, ResourceType.EMBEDDING_API],
            estimated_latency_ms=(100, 400),
            estimated_cost_per_query=0.0008,
            precision_rating=3,
            tags=["vector", "semantic", "fast"],
        ),
        allow_override=True,
    )

    reranking = make_reranking_strategy(pool, embed_query_fn)
    registry.register(
        "reranking",
        reranking,
        StrategyMetadata(
            name="reranking",
            description="Two-stage retrieval: vector search + cross-encoder reranking (ms-marco-MiniLM)",
            strategy_type=StrategyType.RERANKING,
            version="1.0.0",
            required_resources=[ResourceType.DATABASE, ResourceType.EMBEDDING_API, ResourceType.RERANKER],
            estimated_latency_ms=(200, 600),
            estimated_cost_per_query=0.0012,
            precision_rating=4,
            tags=["reranking", "cross-encoder", "precision"],
        ),
        allow_override=True,
    )

    multi_query = make_multi_query_strategy(pool, embed_query_fn)
    registry.register(
        "multi_query",
        multi_query,
        StrategyMetadata(
            name="multi_query",
            description="Query expansion via LLM + parallel vector search; deduplicated results",
            strategy_type=StrategyType.MULTI_QUERY,
            version="1.0.0",
            required_resources=[ResourceType.DATABASE, ResourceType.EMBEDDING_API, ResourceType.LLM_API],
            estimated_latency_ms=(300, 800),
            estimated_cost_per_query=0.0015,
            precision_rating=4,
            tags=["multi-query", "query-expansion", "recall"],
        ),
        allow_override=True,
    )

    query_expansion = make_query_expansion_strategy(pool, embed_query_fn)
    registry.register(
        "query_expansion",
        query_expansion,
        StrategyMetadata(
            name="query_expansion",
            description="Expand query via LLM then single vector search (cheaper than multi_query)",
            strategy_type=StrategyType.QUERY_EXPANSION,
            version="1.0.0",
            required_resources=[ResourceType.DATABASE, ResourceType.EMBEDDING_API, ResourceType.LLM_API],
            estimated_latency_ms=(250, 600),
            estimated_cost_per_query=0.0010,
            precision_rating=3,
            tags=["query-expansion", "single-search"],
        ),
        allow_override=True,
    )

    self_reflective = make_self_reflective_strategy(pool, embed_query_fn)
    registry.register(
        "self_reflective",
        self_reflective,
        StrategyMetadata(
            name="self_reflective",
            description="Search → grade relevance (LLM) → if low, refine query (LLM) → search again",
            strategy_type=StrategyType.SELF_REFLECTIVE,
            version="1.0.0",
            required_resources=[ResourceType.DATABASE, ResourceType.EMBEDDING_API, ResourceType.LLM_API],
            estimated_latency_ms=(500, 1200),
            estimated_cost_per_query=0.0025,
            precision_rating=5,
            tags=["self-reflective", "grade", "refine"],
        ),
        allow_override=True,
    )

    agentic = make_agentic_strategy(pool, embed_query_fn)
    registry.register(
        "agentic",
        agentic,
        StrategyMetadata(
            name="agentic",
            description="Vector search + full document retrieval for top result (chunks + full doc)",
            strategy_type=StrategyType.AGENTIC,
            version="1.0.0",
            required_resources=[ResourceType.DATABASE, ResourceType.EMBEDDING_API],
            estimated_latency_ms=(150, 500),
            estimated_cost_per_query=0.0009,
            precision_rating=4,
            tags=["agentic", "full-document", "chunks"],
        ),
        allow_override=True,
    )

    # Contextual retrieval: same as standard; for use when ingestion used --contextual
    registry.register(
        "contextual_retrieval",
        standard,
        StrategyMetadata(
            name="contextual_retrieval",
            description="Vector search over contextually enriched chunks (use with ingestion --contextual)",
            strategy_type=StrategyType.CONTEXTUAL_RETRIEVAL,
            version="1.0.0",
            required_resources=[ResourceType.DATABASE, ResourceType.EMBEDDING_API],
            estimated_latency_ms=(100, 400),
            estimated_cost_per_query=0.0008,
            precision_rating=4,
            tags=["contextual", "ingestion-enriched"],
        ),
        allow_override=True,
    )

    # Context-aware chunking: same as standard; chunking is applied at ingestion (Docling)
    registry.register(
        "context_aware_chunking",
        standard,
        StrategyMetadata(
            name="context_aware_chunking",
            description="Vector search; chunking is context-aware at ingestion (Docling HybridChunker)",
            strategy_type=StrategyType.STANDARD,
            version="1.0.0",
            required_resources=[ResourceType.DATABASE, ResourceType.EMBEDDING_API],
            estimated_latency_ms=(100, 400),
            estimated_cost_per_query=0.0008,
            precision_rating=3,
            tags=["context-aware-chunking", "ingestion"],
        ),
        allow_override=True,
    )

    logger.info(
        "Registered RAG strategies: standard, reranking, multi_query, query_expansion, "
        "self_reflective, agentic, contextual_retrieval, context_aware_chunking"
    )
