"""
Strategy Utilities.

Database and model utilities for RAG strategies.

Components:
    - db_utils: PostgreSQL + pgvector connection management
    - models: Pydantic models for documents, chunks, configs
    - providers: LLM and embedding provider wrappers
    - embedding_cache: Thread-safe LRU cache for embeddings

Exports:
    - EmbeddingCache: Thread-safe LRU cache for embeddings
    - CacheConfig: Configuration for embedding cache
    - CacheStats: Cache statistics
    - get_embedding_cache: Get global cache instance
"""

__version__ = "0.1.0"

from strategies.utils.embedding_cache import (
    CacheConfig,
    CacheStats,
    EmbeddingCache,
    generate_batch_cache_keys,
    generate_cache_key,
    get_embedding_cache,
    reset_embedding_cache,
)

from strategies.utils.result_cache import (
    ResultCache,
    ResultCacheConfig,
    ResultCacheStats,
    StrategyTTLConfig,
    generate_result_cache_key,
    get_result_cache,
    reset_result_cache,
)

__all__ = [
    "__version__",
    # Embedding Cache
    "CacheConfig",
    "CacheStats",
    "EmbeddingCache",
    "generate_batch_cache_keys",
    "generate_cache_key",
    "get_embedding_cache",
    "reset_embedding_cache",
    # Result Cache
    "ResultCache",
    "ResultCacheConfig",
    "ResultCacheStats",
    "StrategyTTLConfig",
    "generate_result_cache_key",
    "get_result_cache",
    "reset_result_cache",
]
