"""
Unit tests for Redis Sliding Window Rate Limiter.

Tests cover:
- Rate limit checking (sync and async)
- Request counting
- Window sliding behavior
- Retry-after calculation
- Rate limit headers
"""

from __future__ import annotations

import time

import pytest

from api.rate_limiter import (
    RateLimitConfig,
    RateLimiter,
    RateLimitResult,
)


# =============================================================================
# Test: RateLimitConfig
# =============================================================================


class TestRateLimitConfig:
    """Tests for RateLimitConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = RateLimitConfig()

        assert config.default_limit == 60
        assert config.default_window_seconds == 60
        assert config.key_prefix == "ratelimit:"

    def test_custom_config(self):
        """Test custom configuration."""
        config = RateLimitConfig(
            default_limit=100,
            default_window_seconds=30,
            key_prefix="myapp:",
        )

        assert config.default_limit == 100
        assert config.default_window_seconds == 30
        assert config.key_prefix == "myapp:"


# =============================================================================
# Test: RateLimitResult
# =============================================================================


class TestRateLimitResult:
    """Tests for RateLimitResult."""

    def test_allowed_result(self):
        """Test allowed result."""
        result = RateLimitResult(
            allowed=True,
            current_count=5,
            limit=60,
            remaining=55,
            reset_at=time.time() + 60,
        )

        assert result.allowed is True
        assert result.remaining == 55

    def test_denied_result(self):
        """Test denied result."""
        result = RateLimitResult(
            allowed=False,
            current_count=61,
            limit=60,
            remaining=0,
            reset_at=time.time() + 60,
            retry_after=30.5,
        )

        assert result.allowed is False
        assert result.retry_after == 30.5

    def test_headers(self):
        """Test HTTP headers generation."""
        reset_at = time.time() + 60
        result = RateLimitResult(
            allowed=True,
            current_count=10,
            limit=100,
            remaining=90,
            reset_at=reset_at,
        )

        headers = result.headers

        assert headers["X-RateLimit-Limit"] == "100"
        assert headers["X-RateLimit-Remaining"] == "90"
        assert headers["X-RateLimit-Reset"] == str(int(reset_at))


# =============================================================================
# Test: RateLimiter - Synchronous
# =============================================================================


class TestRateLimiterSync:
    """Tests for synchronous rate limiting."""

    @pytest.fixture
    def limiter(self) -> RateLimiter:
        """Create a rate limiter instance."""
        return RateLimiter()

    def test_first_request_allowed(self, limiter: RateLimiter):
        """Test first request is allowed."""
        result = limiter.check_sync("user1", limit=10)

        assert result.allowed is True
        assert result.current_count == 1
        assert result.remaining == 9

    def test_within_limit(self, limiter: RateLimiter):
        """Test requests within limit are allowed."""
        for i in range(5):
            result = limiter.check_sync("user1", limit=10)
            assert result.allowed is True
            assert result.current_count == i + 1

    def test_exceeds_limit(self, limiter: RateLimiter):
        """Test requests exceeding limit are denied."""
        # Make 10 requests (at limit)
        for _ in range(10):
            result = limiter.check_sync("user1", limit=10)

        # 11th request should be denied
        result = limiter.check_sync("user1", limit=10)

        assert result.allowed is False
        assert result.current_count == 11
        assert result.remaining == 0

    def test_different_identifiers(self, limiter: RateLimiter):
        """Test different users have separate limits."""
        for _ in range(5):
            limiter.check_sync("user1", limit=10)

        result = limiter.check_sync("user2", limit=10)

        assert result.allowed is True
        assert result.current_count == 1

    def test_retry_after_calculated(self, limiter: RateLimiter):
        """Test retry_after is calculated when rate limited."""
        # Exhaust limit
        for _ in range(10):
            limiter.check_sync("user1", limit=10, window_seconds=60)

        # Next request should have retry_after
        result = limiter.check_sync("user1", limit=10, window_seconds=60)

        assert result.allowed is False
        assert result.retry_after > 0
        assert result.retry_after <= 60

    def test_reset_clears_count(self, limiter: RateLimiter):
        """Test reset clears request count."""
        for _ in range(5):
            limiter.check_sync("user1", limit=10)

        limiter.reset_sync("user1")

        result = limiter.check_sync("user1", limit=10)
        assert result.current_count == 1


# =============================================================================
# Test: RateLimiter - Async
# =============================================================================


class TestRateLimiterAsync:
    """Tests for async rate limiting."""

    @pytest.mark.asyncio
    async def test_async_check_without_redis(self):
        """Test async check falls back to local storage."""
        limiter = RateLimiter()
        result = await limiter.check("user1", limit=10)

        assert result.allowed is True
        assert result.current_count == 1

    @pytest.mark.asyncio
    async def test_async_multiple_checks(self):
        """Test multiple async checks."""
        limiter = RateLimiter()

        for i in range(5):
            result = await limiter.check("user1", limit=10)
            assert result.current_count == i + 1

    @pytest.mark.asyncio
    async def test_async_get_current_count(self):
        """Test getting current count."""
        limiter = RateLimiter()

        for _ in range(3):
            await limiter.check("user1", limit=10)

        count = await limiter.get_current_count("user1")
        assert count == 3

    @pytest.mark.asyncio
    async def test_async_reset(self):
        """Test async reset."""
        limiter = RateLimiter()

        for _ in range(5):
            await limiter.check("user1", limit=10)

        success = await limiter.reset("user1")
        assert success is True

        count = await limiter.get_current_count("user1")
        assert count == 0


# =============================================================================
# Test: Window Sliding
# =============================================================================


class TestWindowSliding:
    """Tests for sliding window behavior."""

    def test_old_entries_removed(self):
        """Test old entries outside window are removed."""
        limiter = RateLimiter()

        # Manually add old entries
        old_time = time.time() - 100  # 100 seconds ago
        limiter._local_counts["user1"] = [old_time, old_time + 1]

        # Check with 60 second window - old entries should be removed
        result = limiter.check_sync("user1", limit=10, window_seconds=60)

        # Only the new request should be counted
        assert result.current_count == 1

    def test_recent_entries_kept(self):
        """Test recent entries within window are kept."""
        limiter = RateLimiter()

        # Add recent entries
        recent_time = time.time() - 30  # 30 seconds ago
        limiter._local_counts["user1"] = [recent_time, recent_time + 1]

        # Check with 60 second window - recent entries should be kept
        result = limiter.check_sync("user1", limit=10, window_seconds=60)

        # Previous entries + new request
        assert result.current_count == 3


# =============================================================================
# Test: Configuration Integration
# =============================================================================


class TestConfigurationIntegration:
    """Tests for configuration integration."""

    def test_uses_default_limit(self):
        """Test limiter uses config default limit."""
        config = RateLimitConfig(default_limit=5)
        limiter = RateLimiter(config=config)

        for _ in range(5):
            result = limiter.check_sync("user1")
            assert result.allowed is True

        result = limiter.check_sync("user1")
        assert result.allowed is False

    def test_uses_default_window(self):
        """Test limiter uses config default window."""
        config = RateLimitConfig(default_window_seconds=10)
        limiter = RateLimiter(config=config)

        result = limiter.check_sync("user1")

        # Reset should be ~10 seconds from now
        expected_reset = time.time() + 10
        assert abs(result.reset_at - expected_reset) < 1

    def test_override_defaults(self):
        """Test method params override defaults."""
        config = RateLimitConfig(default_limit=100)
        limiter = RateLimiter(config=config)

        # Override with limit=3
        for _ in range(3):
            result = limiter.check_sync("user1", limit=3)
            assert result.allowed is True

        result = limiter.check_sync("user1", limit=3)
        assert result.allowed is False
