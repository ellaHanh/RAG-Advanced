"""
RAG-Advanced Query Result TTL Cache.

Time-based cache for strategy execution results with per-strategy TTL configuration.
Automatically expires old entries based on configurable time-to-live.

Usage:
    from strategies.utils.result_cache import ResultCache, ResultCacheConfig

    cache = ResultCache()
    
    # Store result with strategy-specific TTL
    cache.set("query hash", result, strategy="reranking")
    
    # Retrieve if not expired
    result = cache.get("query hash", strategy="reranking")
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from orchestration.models import ExecutionResult


logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


class StrategyTTLConfig(BaseModel):
    """
    TTL configuration for a specific strategy.

    Attributes:
        ttl_seconds: Time-to-live in seconds.
        max_entries: Maximum cached entries for this strategy.
    """

    model_config = ConfigDict(frozen=True)

    ttl_seconds: int = Field(default=300, ge=0, description="TTL (seconds)")
    max_entries: int = Field(default=1000, ge=10, description="Max entries")


class ResultCacheConfig(BaseModel):
    """
    Configuration for result cache.

    Attributes:
        default_ttl_seconds: Default TTL for strategies without specific config.
        default_max_entries: Default max entries per strategy.
        strategy_configs: Per-strategy TTL configurations.
        cleanup_interval_seconds: How often to run cleanup.
    """

    model_config = ConfigDict(frozen=True)

    default_ttl_seconds: int = Field(default=300, ge=0, description="Default TTL")
    default_max_entries: int = Field(default=1000, ge=10, description="Default max entries")
    strategy_configs: dict[str, StrategyTTLConfig] = Field(
        default_factory=dict,
        description="Per-strategy configs",
    )
    cleanup_interval_seconds: int = Field(default=60, ge=10, description="Cleanup interval")


# =============================================================================
# Cache Entry
# =============================================================================


@dataclass
class ResultCacheEntry:
    """
    A single cached result entry.

    Attributes:
        result: The cached execution result.
        strategy: Strategy name.
        created_at: Unix timestamp when entry was created.
        ttl_seconds: Time-to-live for this entry.
        access_count: Number of times entry was accessed.
    """

    result: ExecutionResult
    strategy: str
    created_at: float = field(default_factory=time.time)
    ttl_seconds: int = 300
    access_count: int = 0

    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.ttl_seconds <= 0:
            return False
        return (time.time() - self.created_at) > self.ttl_seconds

    @property
    def expires_at(self) -> float:
        """Get expiration timestamp."""
        if self.ttl_seconds <= 0:
            return float("inf")
        return self.created_at + self.ttl_seconds

    @property
    def remaining_ttl(self) -> float:
        """Get remaining time-to-live in seconds."""
        if self.ttl_seconds <= 0:
            return float("inf")
        remaining = self.expires_at - time.time()
        return max(0, remaining)


# =============================================================================
# Cache Statistics
# =============================================================================


@dataclass
class ResultCacheStats:
    """
    Result cache statistics.

    Attributes:
        hits: Number of cache hits.
        misses: Number of cache misses.
        expirations: Number of expired entries removed.
        evictions: Number of LRU evictions.
        entries_by_strategy: Entry count per strategy.
    """

    hits: int = 0
    misses: int = 0
    expirations: int = 0
    evictions: int = 0
    entries_by_strategy: dict[str, int] = field(default_factory=dict)

    @property
    def hit_rate(self) -> float:
        """Calculate hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def total_entries(self) -> int:
        """Get total entry count."""
        return sum(self.entries_by_strategy.values())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "expirations": self.expirations,
            "evictions": self.evictions,
            "hit_rate": self.hit_rate,
            "total_entries": self.total_entries,
            "entries_by_strategy": self.entries_by_strategy,
        }


# =============================================================================
# Result Cache
# =============================================================================


class ResultCache:
    """
    TTL-based cache for strategy execution results.

    Supports per-strategy TTL configuration and automatic expiration.
    Thread-safe for concurrent access.

    Example:
        >>> cache = ResultCache()
        >>> cache.set(query_hash, result, strategy="standard")
        >>> cached_result = cache.get(query_hash, strategy="standard")
    """

    def __init__(self, config: ResultCacheConfig | None = None) -> None:
        """
        Initialize the result cache.

        Args:
            config: Optional cache configuration.
        """
        self.config = config or ResultCacheConfig()
        self._caches: dict[str, OrderedDict[str, ResultCacheEntry]] = {}
        self._lock = threading.RLock()
        self._stats = ResultCacheStats()
        self._last_cleanup = time.time()

    @property
    def stats(self) -> ResultCacheStats:
        """Get cache statistics."""
        with self._lock:
            self._update_stats()
            return self._stats

    def get(
        self,
        key: str,
        strategy: str,
    ) -> ExecutionResult | None:
        """
        Get cached result.

        Args:
            key: Cache key (query hash).
            strategy: Strategy name.

        Returns:
            ExecutionResult if found and not expired, None otherwise.
        """
        with self._lock:
            self._maybe_cleanup()

            cache = self._caches.get(strategy)
            if cache is None:
                self._stats.misses += 1
                return None

            entry = cache.get(key)
            if entry is None:
                self._stats.misses += 1
                return None

            if entry.is_expired():
                del cache[key]
                self._stats.expirations += 1
                self._stats.misses += 1
                return None

            # Move to end (most recently used)
            cache.move_to_end(key)
            entry.access_count += 1
            self._stats.hits += 1

            return entry.result

    def set(
        self,
        key: str,
        result: ExecutionResult,
        strategy: str,
        ttl_seconds: int | None = None,
    ) -> None:
        """
        Store result in cache.

        Args:
            key: Cache key (query hash).
            result: Execution result to cache.
            strategy: Strategy name.
            ttl_seconds: Optional TTL override.
        """
        with self._lock:
            self._maybe_cleanup()

            # Get or create strategy cache
            if strategy not in self._caches:
                self._caches[strategy] = OrderedDict()
            cache = self._caches[strategy]

            # Determine TTL
            if ttl_seconds is None:
                ttl_seconds = self._get_strategy_ttl(strategy)

            # Evict if at capacity
            max_entries = self._get_strategy_max_entries(strategy)
            while len(cache) >= max_entries:
                cache.popitem(last=False)
                self._stats.evictions += 1

            # Store entry
            cache[key] = ResultCacheEntry(
                result=result,
                strategy=strategy,
                ttl_seconds=ttl_seconds,
            )

    def delete(
        self,
        key: str,
        strategy: str,
    ) -> bool:
        """
        Delete cached result.

        Args:
            key: Cache key.
            strategy: Strategy name.

        Returns:
            True if entry was deleted.
        """
        with self._lock:
            cache = self._caches.get(strategy)
            if cache is None:
                return False

            if key in cache:
                del cache[key]
                return True
            return False

    def clear(self, strategy: str | None = None) -> None:
        """
        Clear cache entries.

        Args:
            strategy: Strategy to clear (None = all strategies).
        """
        with self._lock:
            if strategy:
                if strategy in self._caches:
                    self._caches[strategy].clear()
            else:
                self._caches.clear()
                self._stats = ResultCacheStats()

    def get_entry_info(
        self,
        key: str,
        strategy: str,
    ) -> dict[str, Any] | None:
        """
        Get information about a cached entry.

        Args:
            key: Cache key.
            strategy: Strategy name.

        Returns:
            Entry information dict or None.
        """
        with self._lock:
            cache = self._caches.get(strategy)
            if cache is None:
                return None

            entry = cache.get(key)
            if entry is None or entry.is_expired():
                return None

            return {
                "strategy": entry.strategy,
                "created_at": entry.created_at,
                "expires_at": entry.expires_at,
                "remaining_ttl": entry.remaining_ttl,
                "access_count": entry.access_count,
            }

    # =========================================================================
    # Private Methods
    # =========================================================================

    def _get_strategy_ttl(self, strategy: str) -> int:
        """Get TTL for a strategy."""
        if strategy in self.config.strategy_configs:
            return self.config.strategy_configs[strategy].ttl_seconds
        return self.config.default_ttl_seconds

    def _get_strategy_max_entries(self, strategy: str) -> int:
        """Get max entries for a strategy."""
        if strategy in self.config.strategy_configs:
            return self.config.strategy_configs[strategy].max_entries
        return self.config.default_max_entries

    def _maybe_cleanup(self) -> None:
        """Run cleanup if interval has passed."""
        now = time.time()
        if now - self._last_cleanup >= self.config.cleanup_interval_seconds:
            self._cleanup_expired()
            self._last_cleanup = now

    def _cleanup_expired(self) -> None:
        """Remove all expired entries."""
        for cache in self._caches.values():
            expired_keys = [
                key for key, entry in cache.items()
                if entry.is_expired()
            ]
            for key in expired_keys:
                del cache[key]
                self._stats.expirations += 1

    def _update_stats(self) -> None:
        """Update statistics counts."""
        self._stats.entries_by_strategy = {
            strategy: len(cache)
            for strategy, cache in self._caches.items()
        }


# =============================================================================
# Cache Key Generation
# =============================================================================


def generate_result_cache_key(
    query: str,
    strategy: str,
    config_hash: str | None = None,
) -> str:
    """
    Generate cache key for a query result.

    Args:
        query: The query text.
        strategy: Strategy name.
        config_hash: Optional configuration hash.

    Returns:
        Cache key string.
    """
    content = f"{strategy}:{query}"
    if config_hash:
        content = f"{config_hash}:{content}"

    return hashlib.sha256(content.encode("utf-8")).hexdigest()


# =============================================================================
# Global Instance
# =============================================================================

_global_result_cache: ResultCache | None = None
_global_lock = threading.Lock()


def get_result_cache(config: ResultCacheConfig | None = None) -> ResultCache:
    """
    Get or create the global result cache.

    Args:
        config: Optional configuration (only used on first call).

    Returns:
        Global ResultCache instance.
    """
    global _global_result_cache

    if _global_result_cache is None:
        with _global_lock:
            if _global_result_cache is None:
                _global_result_cache = ResultCache(config)

    return _global_result_cache


def reset_result_cache() -> None:
    """Reset the global result cache."""
    global _global_result_cache

    with _global_lock:
        _global_result_cache = None
