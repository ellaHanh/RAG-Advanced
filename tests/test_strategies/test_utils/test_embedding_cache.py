"""
Unit tests for Embedding Cache.

Tests cover:
- Cache operations (get, set, delete)
- LRU eviction
- TTL expiration
- Thread safety
- Batch operations
- Statistics
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from strategies.utils.embedding_cache import (
    CacheConfig,
    CacheEntry,
    CacheStats,
    EmbeddingCache,
    generate_batch_cache_keys,
    generate_cache_key,
    get_embedding_cache,
    reset_embedding_cache,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def cache() -> EmbeddingCache:
    """Create a test cache instance."""
    return EmbeddingCache(CacheConfig(max_size=100))


@pytest.fixture
def small_cache() -> EmbeddingCache:
    """Create a small cache for eviction tests."""
    return EmbeddingCache(CacheConfig(max_size=5, eviction_batch_size=2))


@pytest.fixture(autouse=True)
def reset_global_cache():
    """Reset global cache before each test."""
    reset_embedding_cache()
    yield
    reset_embedding_cache()


# =============================================================================
# Test: CacheConfig
# =============================================================================


class TestCacheConfig:
    """Tests for CacheConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = CacheConfig()

        assert config.max_size == 10000
        assert config.ttl_seconds == 0
        assert config.eviction_batch_size == 100

    def test_custom_config(self):
        """Test custom configuration."""
        config = CacheConfig(
            max_size=5000,
            ttl_seconds=3600,
            eviction_batch_size=50,
        )

        assert config.max_size == 5000
        assert config.ttl_seconds == 3600


# =============================================================================
# Test: CacheEntry
# =============================================================================


class TestCacheEntry:
    """Tests for CacheEntry."""

    def test_entry_creation(self):
        """Test entry creation."""
        embedding = [0.1, 0.2, 0.3]
        entry = CacheEntry(embedding=embedding)

        assert entry.embedding == embedding
        assert entry.access_count == 0
        assert entry.created_at > 0

    def test_touch_updates_metadata(self):
        """Test touch updates access metadata."""
        entry = CacheEntry(embedding=[0.1])
        original_time = entry.last_accessed

        time.sleep(0.01)
        entry.touch()

        assert entry.access_count == 1
        assert entry.last_accessed > original_time

    def test_is_expired_no_ttl(self):
        """Test expiration with no TTL."""
        entry = CacheEntry(embedding=[0.1])

        assert entry.is_expired(0) is False

    def test_is_expired_within_ttl(self):
        """Test not expired within TTL."""
        entry = CacheEntry(embedding=[0.1])

        assert entry.is_expired(3600) is False

    def test_is_expired_past_ttl(self):
        """Test expired past TTL."""
        entry = CacheEntry(embedding=[0.1])
        entry.created_at = time.time() - 100  # 100 seconds ago

        assert entry.is_expired(60) is True


# =============================================================================
# Test: CacheStats
# =============================================================================


class TestCacheStats:
    """Tests for CacheStats."""

    def test_hit_rate_calculation(self):
        """Test hit rate calculation."""
        stats = CacheStats(hits=80, misses=20)

        assert stats.hit_rate == pytest.approx(0.8)

    def test_hit_rate_zero_requests(self):
        """Test hit rate with no requests."""
        stats = CacheStats()

        assert stats.hit_rate == 0.0

    def test_to_dict(self):
        """Test serialization."""
        stats = CacheStats(hits=10, misses=5, evictions=2, current_size=100)
        d = stats.to_dict()

        assert d["hits"] == 10
        assert d["misses"] == 5
        assert d["evictions"] == 2


# =============================================================================
# Test: Basic Cache Operations
# =============================================================================


class TestBasicCacheOperations:
    """Tests for basic cache operations."""

    def test_set_and_get(self, cache: EmbeddingCache):
        """Test basic set and get."""
        embedding = [0.1, 0.2, 0.3]
        cache.set("hello", embedding)

        result = cache.get("hello")

        assert result == embedding

    def test_get_missing_key(self, cache: EmbeddingCache):
        """Test get with missing key."""
        result = cache.get("nonexistent")

        assert result is None

    def test_delete(self, cache: EmbeddingCache):
        """Test delete operation."""
        cache.set("hello", [0.1])

        deleted = cache.delete("hello")

        assert deleted is True
        assert cache.get("hello") is None

    def test_delete_missing_key(self, cache: EmbeddingCache):
        """Test delete with missing key."""
        deleted = cache.delete("nonexistent")

        assert deleted is False

    def test_contains(self, cache: EmbeddingCache):
        """Test contains check."""
        cache.set("hello", [0.1])

        assert cache.contains("hello") is True
        assert cache.contains("nonexistent") is False

    def test_clear(self, cache: EmbeddingCache):
        """Test clear operation."""
        cache.set("a", [0.1])
        cache.set("b", [0.2])

        cache.clear()

        assert cache.get("a") is None
        assert cache.get("b") is None
        assert cache.stats.current_size == 0


# =============================================================================
# Test: LRU Eviction
# =============================================================================


class TestLRUEviction:
    """Tests for LRU eviction behavior."""

    def test_evicts_when_full(self, small_cache: EmbeddingCache):
        """Test eviction when cache is full."""
        # Fill cache
        for i in range(5):
            small_cache.set(f"key{i}", [float(i)])

        # Add one more (should trigger eviction)
        small_cache.set("new_key", [99.0])

        # Cache should not exceed max_size
        assert small_cache.stats.current_size <= 5

    def test_evicts_least_recently_used(self, small_cache: EmbeddingCache):
        """Test LRU eviction order."""
        # Fill cache
        for i in range(5):
            small_cache.set(f"key{i}", [float(i)])

        # Access key0 to make it recently used
        small_cache.get("key0")

        # Add new keys (should evict key1, key2 as they're oldest)
        small_cache.set("new1", [10.0])
        small_cache.set("new2", [11.0])

        # key0 should still exist (was accessed)
        assert small_cache.get("key0") is not None

    def test_eviction_count_tracked(self, small_cache: EmbeddingCache):
        """Test eviction count is tracked."""
        # Fill cache
        for i in range(10):
            small_cache.set(f"key{i}", [float(i)])

        # Should have evicted some entries
        assert small_cache.stats.evictions > 0


# =============================================================================
# Test: TTL Expiration
# =============================================================================


class TestTTLExpiration:
    """Tests for TTL-based expiration."""

    def test_expired_entry_not_returned(self):
        """Test expired entries are not returned."""
        cache = EmbeddingCache(CacheConfig(max_size=100, ttl_seconds=1))
        cache.set("test", [0.1])

        # Manually expire the entry
        key = cache._hash_text("test")
        cache._cache[key].created_at = time.time() - 10

        result = cache.get("test")

        assert result is None

    def test_expired_entry_removed_on_access(self):
        """Test expired entries are removed on access."""
        cache = EmbeddingCache(CacheConfig(max_size=100, ttl_seconds=1))
        cache.set("test", [0.1])

        # Manually expire
        key = cache._hash_text("test")
        cache._cache[key].created_at = time.time() - 10

        cache.get("test")

        assert cache.contains("test") is False


# =============================================================================
# Test: Batch Operations
# =============================================================================


class TestBatchOperations:
    """Tests for batch cache operations."""

    def test_get_batch(self, cache: EmbeddingCache):
        """Test batch get."""
        cache.set("a", [0.1])
        cache.set("b", [0.2])

        results = cache.get_batch(["a", "b", "c"])

        assert results["a"] == [0.1]
        assert results["b"] == [0.2]
        assert results["c"] is None

    def test_set_batch(self, cache: EmbeddingCache):
        """Test batch set."""
        embeddings = {
            "a": [0.1],
            "b": [0.2],
            "c": [0.3],
        }

        cache.set_batch(embeddings)

        assert cache.get("a") == [0.1]
        assert cache.get("b") == [0.2]
        assert cache.get("c") == [0.3]


# =============================================================================
# Test: Statistics
# =============================================================================


class TestStatistics:
    """Tests for cache statistics."""

    def test_hits_counted(self, cache: EmbeddingCache):
        """Test cache hits are counted."""
        cache.set("test", [0.1])

        cache.get("test")
        cache.get("test")

        assert cache.stats.hits == 2

    def test_misses_counted(self, cache: EmbeddingCache):
        """Test cache misses are counted."""
        cache.get("nonexistent")
        cache.get("also_missing")

        assert cache.stats.misses == 2

    def test_current_size_accurate(self, cache: EmbeddingCache):
        """Test current size is accurate."""
        cache.set("a", [0.1])
        cache.set("b", [0.2])

        assert cache.stats.current_size == 2


# =============================================================================
# Test: Thread Safety
# =============================================================================


class TestThreadSafety:
    """Tests for thread-safe operations."""

    def test_concurrent_writes(self, cache: EmbeddingCache):
        """Test concurrent write operations."""
        def writer(thread_id: int):
            for i in range(100):
                cache.set(f"thread{thread_id}_key{i}", [float(i)])

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(writer, i) for i in range(4)]
            for f in futures:
                f.result()

        # All entries should be present
        assert cache.stats.current_size <= 100  # Limited by max_size

    def test_concurrent_reads_and_writes(self, cache: EmbeddingCache):
        """Test concurrent read and write operations."""
        # Pre-populate
        for i in range(50):
            cache.set(f"key{i}", [float(i)])

        errors = []

        def reader():
            for i in range(50):
                try:
                    cache.get(f"key{i}")
                except Exception as e:
                    errors.append(e)

        def writer():
            for i in range(50, 100):
                try:
                    cache.set(f"key{i}", [float(i)])
                except Exception as e:
                    errors.append(e)

        threads = [
            threading.Thread(target=reader) for _ in range(2)
        ] + [
            threading.Thread(target=writer) for _ in range(2)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


# =============================================================================
# Test: Get or Compute
# =============================================================================


class TestGetOrCompute:
    """Tests for get_or_compute functionality."""

    def test_get_or_compute_cache_hit(self, cache: EmbeddingCache):
        """Test get_or_compute returns cached value."""
        cache.set("test", [0.1, 0.2])
        compute_called = False

        def compute(text: str) -> list[float]:
            nonlocal compute_called
            compute_called = True
            return [0.9, 0.9]

        result = cache.get_or_compute("test", compute)

        assert result == [0.1, 0.2]
        assert compute_called is False

    def test_get_or_compute_cache_miss(self, cache: EmbeddingCache):
        """Test get_or_compute computes on miss."""
        def compute(text: str) -> list[float]:
            return [0.5, 0.5]

        result = cache.get_or_compute("new_text", compute)

        assert result == [0.5, 0.5]
        assert cache.get("new_text") == [0.5, 0.5]

    @pytest.mark.asyncio
    async def test_get_or_compute_async(self, cache: EmbeddingCache):
        """Test async get_or_compute."""
        async def compute(text: str) -> list[float]:
            return [0.7, 0.7]

        result = await cache.get_or_compute_async("async_text", compute)

        assert result == [0.7, 0.7]


# =============================================================================
# Test: Global Cache
# =============================================================================


class TestGlobalCache:
    """Tests for global cache instance."""

    def test_get_embedding_cache_returns_same_instance(self):
        """Test global cache returns same instance."""
        cache1 = get_embedding_cache()
        cache2 = get_embedding_cache()

        assert cache1 is cache2

    def test_reset_embedding_cache(self):
        """Test resetting global cache."""
        cache1 = get_embedding_cache()
        cache1.set("test", [0.1])

        reset_embedding_cache()

        cache2 = get_embedding_cache()
        assert cache2.get("test") is None


# =============================================================================
# Test: Cache Key Generation
# =============================================================================


class TestCacheKeyGeneration:
    """Tests for cache key generation."""

    def test_generate_cache_key_basic(self):
        """Test basic key generation."""
        key = generate_cache_key("hello world")

        assert len(key) == 64  # SHA256 hex digest
        assert key.isalnum()

    def test_generate_cache_key_deterministic(self):
        """Test key generation is deterministic."""
        key1 = generate_cache_key("test text")
        key2 = generate_cache_key("test text")

        assert key1 == key2

    def test_generate_cache_key_different_text(self):
        """Test different text produces different keys."""
        key1 = generate_cache_key("text one")
        key2 = generate_cache_key("text two")

        assert key1 != key2

    def test_generate_cache_key_with_model(self):
        """Test key includes model name."""
        key1 = generate_cache_key("hello", model="model_a")
        key2 = generate_cache_key("hello", model="model_b")
        key3 = generate_cache_key("hello")

        assert key1 != key2
        assert key1 != key3

    def test_generate_cache_key_with_prefix(self):
        """Test key with prefix."""
        key = generate_cache_key("hello", prefix="emb:")

        assert key.startswith("emb:")

    def test_generate_batch_cache_keys(self):
        """Test batch key generation."""
        texts = ["hello", "world", "test"]
        keys = generate_batch_cache_keys(texts)

        assert len(keys) == 3
        assert all(text in keys for text in texts)
        assert len(set(keys.values())) == 3  # All unique

    def test_generate_batch_cache_keys_with_model(self):
        """Test batch keys with model."""
        texts = ["a", "b"]
        keys = generate_batch_cache_keys(texts, model="test_model")

        # Same texts with model should be consistent
        single_key = generate_cache_key("a", model="test_model")
        assert keys["a"] == single_key
