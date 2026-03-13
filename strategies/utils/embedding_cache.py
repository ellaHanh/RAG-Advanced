"""
RAG-Advanced Embedding Cache.

Thread-safe LRU cache for embedding vectors to reduce API costs.
Uses SHA256 hashes as keys for efficient lookup.

Usage:
    from strategies.utils.embedding_cache import EmbeddingCache

    cache = EmbeddingCache(max_size=10000)
    
    # Check cache first
    embedding = cache.get("What is machine learning?")
    if embedding is None:
        embedding = await generate_embedding("What is machine learning?")
        cache.set("What is machine learning?", embedding)
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


class CacheConfig(BaseModel):
    """
    Configuration for embedding cache.

    Attributes:
        max_size: Maximum number of entries.
        ttl_seconds: Time-to-live for entries (0 = no expiration).
        eviction_batch_size: Number of entries to evict when full.
    """

    model_config = ConfigDict(frozen=True)

    max_size: int = Field(default=10000, ge=5, description="Max entries")
    ttl_seconds: int = Field(default=0, ge=0, description="TTL (0=forever)")
    eviction_batch_size: int = Field(default=100, ge=1, description="Eviction batch")


# =============================================================================
# Cache Entry
# =============================================================================


@dataclass
class CacheEntry:
    """
    A single cache entry.

    Attributes:
        embedding: The embedding vector.
        created_at: Unix timestamp when entry was created.
        access_count: Number of times entry was accessed.
        last_accessed: Unix timestamp of last access.
    """

    embedding: list[float]
    created_at: float = field(default_factory=time.time)
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)

    def touch(self) -> None:
        """Update access metadata."""
        self.access_count += 1
        self.last_accessed = time.time()

    def is_expired(self, ttl_seconds: int) -> bool:
        """Check if entry has expired."""
        if ttl_seconds <= 0:
            return False
        return (time.time() - self.created_at) > ttl_seconds


# =============================================================================
# Cache Statistics
# =============================================================================


@dataclass
class CacheStats:
    """
    Cache statistics.

    Attributes:
        hits: Number of cache hits.
        misses: Number of cache misses.
        evictions: Number of evictions.
        current_size: Current number of entries.
        max_size: Maximum capacity.
    """

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    current_size: int = 0
    max_size: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "current_size": self.current_size,
            "max_size": self.max_size,
            "hit_rate": self.hit_rate,
        }


# =============================================================================
# Embedding Cache
# =============================================================================


class EmbeddingCache:
    """
    Thread-safe LRU cache for embedding vectors.

    Uses SHA256 hashes of text as keys for efficient lookup.
    Automatically evicts least-recently-used entries when full.

    Example:
        >>> cache = EmbeddingCache(max_size=10000)
        >>> cache.set("hello world", [0.1, 0.2, 0.3])
        >>> embedding = cache.get("hello world")
        >>> print(cache.stats.hit_rate)
    """

    def __init__(self, config: CacheConfig | None = None) -> None:
        """
        Initialize the cache.

        Args:
            config: Optional cache configuration.
        """
        self.config = config or CacheConfig()
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._stats = CacheStats(max_size=self.config.max_size)

    @property
    def stats(self) -> CacheStats:
        """Get cache statistics."""
        with self._lock:
            self._stats.current_size = len(self._cache)
            return self._stats

    def get(self, text: str) -> list[float] | None:
        """
        Get embedding from cache.

        Args:
            text: The text to look up.

        Returns:
            Embedding vector if found, None otherwise.
        """
        key = self._hash_text(text)

        with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                self._stats.misses += 1
                return None

            # Check expiration
            if entry.is_expired(self.config.ttl_seconds):
                del self._cache[key]
                self._stats.misses += 1
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            entry.touch()
            self._stats.hits += 1

            return entry.embedding

    def set(self, text: str, embedding: list[float]) -> None:
        """
        Store embedding in cache.

        Args:
            text: The text key.
            embedding: The embedding vector.
        """
        key = self._hash_text(text)

        with self._lock:
            # If key exists, update and move to end
            if key in self._cache:
                self._cache[key] = CacheEntry(embedding=embedding)
                self._cache.move_to_end(key)
                return

            # Evict if at capacity
            if len(self._cache) >= self.config.max_size:
                self._evict()

            # Add new entry
            self._cache[key] = CacheEntry(embedding=embedding)

    def get_or_compute(
        self,
        text: str,
        compute_fn: Any,
    ) -> list[float]:
        """
        Get from cache or compute and store.

        Args:
            text: The text key.
            compute_fn: Function to compute embedding (sync or async).

        Returns:
            Embedding vector.
        """
        embedding = self.get(text)
        if embedding is not None:
            return embedding

        # Compute new embedding
        embedding = compute_fn(text)
        self.set(text, embedding)
        return embedding

    async def get_or_compute_async(
        self,
        text: str,
        compute_fn: Any,
    ) -> list[float]:
        """
        Async version of get_or_compute.

        Args:
            text: The text key.
            compute_fn: Async function to compute embedding.

        Returns:
            Embedding vector.
        """
        embedding = self.get(text)
        if embedding is not None:
            return embedding

        # Compute new embedding
        embedding = await compute_fn(text)
        self.set(text, embedding)
        return embedding

    def delete(self, text: str) -> bool:
        """
        Delete entry from cache.

        Args:
            text: The text key.

        Returns:
            True if entry was deleted.
        """
        key = self._hash_text(text)

        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear(self) -> None:
        """Clear all entries from cache."""
        with self._lock:
            self._cache.clear()
            self._stats.hits = 0
            self._stats.misses = 0
            self._stats.evictions = 0

    def contains(self, text: str) -> bool:
        """
        Check if text is in cache.

        Args:
            text: The text key.

        Returns:
            True if text is cached.
        """
        key = self._hash_text(text)

        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return False
            if entry.is_expired(self.config.ttl_seconds):
                del self._cache[key]
                return False
            return True

    def get_batch(self, texts: list[str]) -> dict[str, list[float] | None]:
        """
        Get multiple embeddings from cache.

        Args:
            texts: List of texts to look up.

        Returns:
            Dictionary of text -> embedding (None if not found).
        """
        results = {}
        for text in texts:
            results[text] = self.get(text)
        return results

    def set_batch(self, embeddings: dict[str, list[float]]) -> None:
        """
        Store multiple embeddings in cache.

        Args:
            embeddings: Dictionary of text -> embedding.
        """
        for text, embedding in embeddings.items():
            self.set(text, embedding)

    # =========================================================================
    # Private Methods
    # =========================================================================

    def _hash_text(self, text: str) -> str:
        """Hash text to create cache key."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _evict(self) -> None:
        """Evict least-recently-used entries."""
        evict_count = min(
            self.config.eviction_batch_size,
            len(self._cache) - self.config.max_size + 1,
        )

        for _ in range(evict_count):
            if self._cache:
                # Pop oldest (first) entry
                self._cache.popitem(last=False)
                self._stats.evictions += 1


# =============================================================================
# Global Cache Instance
# =============================================================================

_global_cache: EmbeddingCache | None = None
_global_lock = threading.Lock()


def get_embedding_cache(config: CacheConfig | None = None) -> EmbeddingCache:
    """
    Get or create the global embedding cache.

    Args:
        config: Optional configuration (only used on first call).

    Returns:
        Global EmbeddingCache instance.
    """
    global _global_cache

    if _global_cache is None:
        with _global_lock:
            if _global_cache is None:
                _global_cache = EmbeddingCache(config)

    return _global_cache


def reset_embedding_cache() -> None:
    """Reset the global embedding cache."""
    global _global_cache

    with _global_lock:
        _global_cache = None


# =============================================================================
# Cache Key Generation
# =============================================================================


def generate_cache_key(
    text: str,
    model: str | None = None,
    prefix: str | None = None,
) -> str:
    """
    Generate a deterministic cache key from text and optional model name.

    Uses SHA256 hashing to create a unique, collision-resistant key.

    Args:
        text: The text to hash.
        model: Optional model name to include in hash.
        prefix: Optional prefix to prepend to key.

    Returns:
        Deterministic cache key string.

    Example:
        >>> key = generate_cache_key("hello world", model="text-embedding-3-small")
        >>> # Returns: "emb:abc123..." where abc123 is the hash
    """
    # Combine text and model for hashing
    if model:
        content = f"{model}:{text}"
    else:
        content = text

    # Generate SHA256 hash
    hash_value = hashlib.sha256(content.encode("utf-8")).hexdigest()

    # Apply prefix
    if prefix:
        return f"{prefix}{hash_value}"
    return hash_value


def generate_batch_cache_keys(
    texts: list[str],
    model: str | None = None,
    prefix: str | None = None,
) -> dict[str, str]:
    """
    Generate cache keys for multiple texts.

    Args:
        texts: List of texts to hash.
        model: Optional model name.
        prefix: Optional key prefix.

    Returns:
        Dictionary mapping text -> cache key.
    """
    return {
        text: generate_cache_key(text, model, prefix)
        for text in texts
    }
