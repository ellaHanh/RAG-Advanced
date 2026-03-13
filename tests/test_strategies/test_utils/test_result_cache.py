"""
Unit tests for Query Result TTL Cache.

Tests cover:
- Basic cache operations
- TTL expiration
- Per-strategy configuration
- LRU eviction
- Statistics tracking
"""

from __future__ import annotations

import time

import pytest

from orchestration.models import Document, ExecutionResult
from strategies.utils.result_cache import (
    ResultCache,
    ResultCacheConfig,
    ResultCacheEntry,
    ResultCacheStats,
    StrategyTTLConfig,
    generate_result_cache_key,
    get_result_cache,
    reset_result_cache,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_result() -> ExecutionResult:
    """Create a sample execution result."""
    return ExecutionResult(
        documents=[
            Document(
                id="doc1",
                content="Test content",
                title="Test",
                source="test.md",
                similarity=0.9,
            )
        ],
        query="test query",
        strategy_name="standard",
        latency_ms=100,
        cost_usd=0.001,
    )


@pytest.fixture
def cache() -> ResultCache:
    """Create a test cache instance."""
    return ResultCache()


@pytest.fixture
def cache_with_strategy_config() -> ResultCache:
    """Create cache with per-strategy configuration."""
    config = ResultCacheConfig(
        default_ttl_seconds=300,
        strategy_configs={
            "reranking": StrategyTTLConfig(ttl_seconds=600, max_entries=500),
            "standard": StrategyTTLConfig(ttl_seconds=120, max_entries=100),
        },
    )
    return ResultCache(config)


@pytest.fixture(autouse=True)
def reset_global():
    """Reset global cache before each test."""
    reset_result_cache()
    yield
    reset_result_cache()


# =============================================================================
# Test: Configuration
# =============================================================================


class TestConfiguration:
    """Tests for configuration models."""

    def test_default_config(self):
        """Test default configuration."""
        config = ResultCacheConfig()

        assert config.default_ttl_seconds == 300
        assert config.default_max_entries == 1000

    def test_strategy_ttl_config(self):
        """Test strategy-specific configuration."""
        config = StrategyTTLConfig(ttl_seconds=600, max_entries=500)

        assert config.ttl_seconds == 600
        assert config.max_entries == 500


# =============================================================================
# Test: Cache Entry
# =============================================================================


class TestCacheEntry:
    """Tests for ResultCacheEntry."""

    def test_entry_creation(self, sample_result: ExecutionResult):
        """Test entry creation."""
        entry = ResultCacheEntry(
            result=sample_result,
            strategy="standard",
            ttl_seconds=300,
        )

        assert entry.result == sample_result
        assert entry.strategy == "standard"
        assert entry.access_count == 0

    def test_is_expired_within_ttl(self, sample_result: ExecutionResult):
        """Test not expired within TTL."""
        entry = ResultCacheEntry(
            result=sample_result,
            strategy="standard",
            ttl_seconds=300,
        )

        assert entry.is_expired() is False

    def test_is_expired_past_ttl(self, sample_result: ExecutionResult):
        """Test expired past TTL."""
        entry = ResultCacheEntry(
            result=sample_result,
            strategy="standard",
            ttl_seconds=1,
        )
        entry.created_at = time.time() - 10

        assert entry.is_expired() is True

    def test_remaining_ttl(self, sample_result: ExecutionResult):
        """Test remaining TTL calculation."""
        entry = ResultCacheEntry(
            result=sample_result,
            strategy="standard",
            ttl_seconds=300,
        )

        remaining = entry.remaining_ttl
        assert 295 < remaining <= 300

    def test_no_ttl_never_expires(self, sample_result: ExecutionResult):
        """Test entry with no TTL never expires."""
        entry = ResultCacheEntry(
            result=sample_result,
            strategy="standard",
            ttl_seconds=0,
        )
        entry.created_at = time.time() - 10000

        assert entry.is_expired() is False


# =============================================================================
# Test: Basic Operations
# =============================================================================


class TestBasicOperations:
    """Tests for basic cache operations."""

    def test_set_and_get(
        self,
        cache: ResultCache,
        sample_result: ExecutionResult,
    ):
        """Test basic set and get."""
        cache.set("key1", sample_result, strategy="standard")
        result = cache.get("key1", strategy="standard")

        assert result is not None
        assert result.query == sample_result.query

    def test_get_missing_key(self, cache: ResultCache):
        """Test get with missing key."""
        result = cache.get("nonexistent", strategy="standard")

        assert result is None

    def test_get_wrong_strategy(
        self,
        cache: ResultCache,
        sample_result: ExecutionResult,
    ):
        """Test get with wrong strategy."""
        cache.set("key1", sample_result, strategy="standard")
        result = cache.get("key1", strategy="reranking")

        assert result is None

    def test_delete(
        self,
        cache: ResultCache,
        sample_result: ExecutionResult,
    ):
        """Test delete operation."""
        cache.set("key1", sample_result, strategy="standard")
        deleted = cache.delete("key1", strategy="standard")

        assert deleted is True
        assert cache.get("key1", strategy="standard") is None

    def test_delete_missing(self, cache: ResultCache):
        """Test delete with missing key."""
        deleted = cache.delete("nonexistent", strategy="standard")

        assert deleted is False

    def test_clear_single_strategy(
        self,
        cache: ResultCache,
        sample_result: ExecutionResult,
    ):
        """Test clearing single strategy."""
        cache.set("key1", sample_result, strategy="standard")
        cache.set("key2", sample_result, strategy="reranking")

        cache.clear(strategy="standard")

        assert cache.get("key1", strategy="standard") is None
        assert cache.get("key2", strategy="reranking") is not None

    def test_clear_all(
        self,
        cache: ResultCache,
        sample_result: ExecutionResult,
    ):
        """Test clearing all strategies."""
        cache.set("key1", sample_result, strategy="standard")
        cache.set("key2", sample_result, strategy="reranking")

        cache.clear()

        assert cache.get("key1", strategy="standard") is None
        assert cache.get("key2", strategy="reranking") is None


# =============================================================================
# Test: TTL Expiration
# =============================================================================


class TestTTLExpiration:
    """Tests for TTL-based expiration."""

    def test_expired_entry_not_returned(
        self,
        cache: ResultCache,
        sample_result: ExecutionResult,
    ):
        """Test expired entries are not returned."""
        cache.set("key1", sample_result, strategy="standard", ttl_seconds=1)

        # Manually expire
        cache._caches["standard"]["key1"].created_at = time.time() - 10

        result = cache.get("key1", strategy="standard")

        assert result is None

    def test_expired_entry_removed(
        self,
        cache: ResultCache,
        sample_result: ExecutionResult,
    ):
        """Test expired entries are removed on access."""
        cache.set("key1", sample_result, strategy="standard", ttl_seconds=1)
        cache._caches["standard"]["key1"].created_at = time.time() - 10

        cache.get("key1", strategy="standard")

        assert cache.stats.expirations >= 1


# =============================================================================
# Test: Per-Strategy Configuration
# =============================================================================


class TestPerStrategyConfig:
    """Tests for per-strategy TTL configuration."""

    def test_strategy_specific_ttl(
        self,
        cache_with_strategy_config: ResultCache,
        sample_result: ExecutionResult,
    ):
        """Test strategy-specific TTL is applied."""
        cache = cache_with_strategy_config
        cache.set("key1", sample_result, strategy="reranking")

        entry = cache._caches["reranking"]["key1"]
        assert entry.ttl_seconds == 600

    def test_default_ttl_for_unknown_strategy(
        self,
        cache_with_strategy_config: ResultCache,
        sample_result: ExecutionResult,
    ):
        """Test default TTL for unconfigured strategy."""
        cache = cache_with_strategy_config
        cache.set("key1", sample_result, strategy="multi_query")

        entry = cache._caches["multi_query"]["key1"]
        assert entry.ttl_seconds == 300  # Default


# =============================================================================
# Test: Statistics
# =============================================================================


class TestStatistics:
    """Tests for cache statistics."""

    def test_hits_counted(
        self,
        cache: ResultCache,
        sample_result: ExecutionResult,
    ):
        """Test hits are counted."""
        cache.set("key1", sample_result, strategy="standard")

        cache.get("key1", strategy="standard")
        cache.get("key1", strategy="standard")

        assert cache.stats.hits == 2

    def test_misses_counted(self, cache: ResultCache):
        """Test misses are counted."""
        cache.get("nonexistent", strategy="standard")
        cache.get("also_missing", strategy="standard")

        assert cache.stats.misses == 2

    def test_hit_rate_calculation(
        self,
        cache: ResultCache,
        sample_result: ExecutionResult,
    ):
        """Test hit rate calculation."""
        cache.set("key1", sample_result, strategy="standard")

        # 2 hits + 2 misses = 50% hit rate
        cache.get("key1", strategy="standard")
        cache.get("key1", strategy="standard")
        cache.get("missing1", strategy="standard")
        cache.get("missing2", strategy="standard")

        assert cache.stats.hit_rate == pytest.approx(0.5)

    def test_entries_by_strategy(
        self,
        cache: ResultCache,
        sample_result: ExecutionResult,
    ):
        """Test entries counted by strategy."""
        cache.set("key1", sample_result, strategy="standard")
        cache.set("key2", sample_result, strategy="standard")
        cache.set("key3", sample_result, strategy="reranking")

        stats = cache.stats

        assert stats.entries_by_strategy["standard"] == 2
        assert stats.entries_by_strategy["reranking"] == 1


# =============================================================================
# Test: Entry Info
# =============================================================================


class TestEntryInfo:
    """Tests for get_entry_info."""

    def test_get_entry_info(
        self,
        cache: ResultCache,
        sample_result: ExecutionResult,
    ):
        """Test getting entry information."""
        cache.set("key1", sample_result, strategy="standard")

        info = cache.get_entry_info("key1", strategy="standard")

        assert info is not None
        assert info["strategy"] == "standard"
        assert "remaining_ttl" in info
        assert "access_count" in info

    def test_get_entry_info_missing(self, cache: ResultCache):
        """Test entry info for missing key."""
        info = cache.get_entry_info("nonexistent", strategy="standard")

        assert info is None


# =============================================================================
# Test: Key Generation
# =============================================================================


class TestKeyGeneration:
    """Tests for cache key generation."""

    def test_generate_result_cache_key(self):
        """Test key generation."""
        key = generate_result_cache_key("test query", "standard")

        assert len(key) == 64  # SHA256 hex
        assert key.isalnum()

    def test_key_deterministic(self):
        """Test key generation is deterministic."""
        key1 = generate_result_cache_key("test", "standard")
        key2 = generate_result_cache_key("test", "standard")

        assert key1 == key2

    def test_different_strategy_different_key(self):
        """Test different strategy produces different key."""
        key1 = generate_result_cache_key("test", "standard")
        key2 = generate_result_cache_key("test", "reranking")

        assert key1 != key2

    def test_key_with_config_hash(self):
        """Test key includes config hash."""
        key1 = generate_result_cache_key("test", "standard", config_hash="abc")
        key2 = generate_result_cache_key("test", "standard", config_hash="def")

        assert key1 != key2


# =============================================================================
# Test: Global Instance
# =============================================================================


class TestGlobalInstance:
    """Tests for global cache instance."""

    def test_get_result_cache_singleton(self):
        """Test global cache is singleton."""
        cache1 = get_result_cache()
        cache2 = get_result_cache()

        assert cache1 is cache2

    def test_reset_result_cache(self, sample_result: ExecutionResult):
        """Test resetting global cache."""
        cache1 = get_result_cache()
        cache1.set("test", sample_result, strategy="standard")

        reset_result_cache()

        cache2 = get_result_cache()
        assert cache2.get("test", strategy="standard") is None
