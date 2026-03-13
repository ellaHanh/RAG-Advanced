"""
RAG-Advanced Redis Sliding Window Rate Limiter.

Implements sliding window rate limiting using Redis sorted sets.
Provides accurate per-user rate limiting with automatic cleanup.

Usage:
    from api.rate_limiter import RateLimiter, RateLimitConfig

    limiter = RateLimiter(redis_client)
    result = await limiter.check("user_123", limit=60, window_seconds=60)
    if not result.allowed:
        print(f"Rate limited. Retry after {result.retry_after} seconds")
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


class RateLimitConfig(BaseModel):
    """
    Configuration for rate limiting.

    Attributes:
        default_limit: Default requests per window.
        default_window_seconds: Default window size in seconds.
        key_prefix: Prefix for Redis keys.
        cleanup_probability: Probability of cleanup on each check.
    """

    model_config = ConfigDict(frozen=True)

    default_limit: int = Field(default=60, ge=1, description="Default request limit")
    default_window_seconds: int = Field(default=60, ge=1, description="Window size (seconds)")
    key_prefix: str = Field(default="ratelimit:", description="Redis key prefix")
    cleanup_probability: float = Field(
        default=0.1, ge=0.0, le=1.0, description="Cleanup probability"
    )


# =============================================================================
# Result Models
# =============================================================================


@dataclass
class RateLimitResult:
    """
    Result of a rate limit check.

    Attributes:
        allowed: Whether the request is allowed.
        current_count: Current request count in window.
        limit: Maximum allowed requests.
        remaining: Remaining requests in window.
        reset_at: Unix timestamp when window resets.
        retry_after: Seconds until request would be allowed (if limited).
    """

    allowed: bool
    current_count: int
    limit: int
    remaining: int
    reset_at: float
    retry_after: float = 0.0

    @property
    def headers(self) -> dict[str, str]:
        """Get rate limit headers for HTTP response."""
        return {
            "X-RateLimit-Limit": str(self.limit),
            "X-RateLimit-Remaining": str(max(0, self.remaining)),
            "X-RateLimit-Reset": str(int(self.reset_at)),
        }


# =============================================================================
# Rate Limiter
# =============================================================================


class RateLimiter:
    """
    Redis-based sliding window rate limiter.

    Uses Redis sorted sets to track request timestamps within a sliding
    window. Old entries are automatically cleaned up.

    Example:
        >>> limiter = RateLimiter(redis)
        >>> result = await limiter.check("user_123", limit=100)
        >>> if result.allowed:
        ...     process_request()
        >>> else:
        ...     return_429(result.retry_after)
    """

    def __init__(
        self,
        redis_client: Any | None = None,
        config: RateLimitConfig | None = None,
    ) -> None:
        """
        Initialize the rate limiter.

        Args:
            redis_client: Redis client (async redis-py).
            config: Optional configuration.
        """
        self._redis = redis_client
        self.config = config or RateLimitConfig()
        self._local_counts: dict[str, list[float]] = {}  # For testing without Redis

    async def check(
        self,
        identifier: str,
        limit: int | None = None,
        window_seconds: int | None = None,
    ) -> RateLimitResult:
        """
        Check if request is allowed and record it.

        Args:
            identifier: Unique identifier (user ID, API key, IP).
            limit: Maximum requests per window.
            window_seconds: Window size in seconds.

        Returns:
            RateLimitResult with allow/deny decision.
        """
        limit = limit or self.config.default_limit
        window_seconds = window_seconds or self.config.default_window_seconds

        now = time.time()
        window_start = now - window_seconds
        reset_at = now + window_seconds

        if self._redis:
            return await self._check_redis(identifier, limit, window_seconds, now, window_start, reset_at)
        else:
            return self._check_local(identifier, limit, window_seconds, now, window_start, reset_at)

    async def _check_redis(
        self,
        identifier: str,
        limit: int,
        window_seconds: int,
        now: float,
        window_start: float,
        reset_at: float,
    ) -> RateLimitResult:
        """Check rate limit using Redis sorted set."""
        key = f"{self.config.key_prefix}{identifier}"

        try:
            # Use Redis pipeline for atomic operations
            pipe = self._redis.pipeline()

            # Remove entries outside the window
            pipe.zremrangebyscore(key, "-inf", window_start)

            # Add current request timestamp
            pipe.zadd(key, {str(now): now})

            # Count requests in window
            pipe.zcard(key)

            # Set key expiration
            pipe.expire(key, window_seconds + 1)

            # Execute pipeline
            results = await pipe.execute()
            current_count = results[2]  # ZCARD result

            allowed = current_count <= limit
            remaining = max(0, limit - current_count)

            # If not allowed, calculate retry after
            retry_after = 0.0
            if not allowed:
                # Get oldest entry to calculate when it will expire
                oldest = await self._redis.zrange(key, 0, 0, withscores=True)
                if oldest:
                    oldest_time = oldest[0][1]
                    retry_after = max(0, (oldest_time + window_seconds) - now)

            return RateLimitResult(
                allowed=allowed,
                current_count=current_count,
                limit=limit,
                remaining=remaining,
                reset_at=reset_at,
                retry_after=retry_after,
            )

        except Exception as e:
            logger.exception(f"Redis rate limit check failed: {e}")
            # Fail open - allow request if Redis is unavailable
            return RateLimitResult(
                allowed=True,
                current_count=0,
                limit=limit,
                remaining=limit,
                reset_at=reset_at,
            )

    def _check_local(
        self,
        identifier: str,
        limit: int,
        window_seconds: int,
        now: float,
        window_start: float,
        reset_at: float,
    ) -> RateLimitResult:
        """Check rate limit using local memory (for testing)."""
        if identifier not in self._local_counts:
            self._local_counts[identifier] = []

        # Remove old entries
        self._local_counts[identifier] = [
            ts for ts in self._local_counts[identifier]
            if ts > window_start
        ]

        # Add current request
        self._local_counts[identifier].append(now)

        current_count = len(self._local_counts[identifier])
        allowed = current_count <= limit
        remaining = max(0, limit - current_count)

        # Calculate retry after
        retry_after = 0.0
        if not allowed and self._local_counts[identifier]:
            oldest_time = min(self._local_counts[identifier])
            retry_after = max(0, (oldest_time + window_seconds) - now)

        return RateLimitResult(
            allowed=allowed,
            current_count=current_count,
            limit=limit,
            remaining=remaining,
            reset_at=reset_at,
            retry_after=retry_after,
        )

    def check_sync(
        self,
        identifier: str,
        limit: int | None = None,
        window_seconds: int | None = None,
    ) -> RateLimitResult:
        """
        Synchronous rate limit check (uses local storage).

        Args:
            identifier: Unique identifier.
            limit: Maximum requests per window.
            window_seconds: Window size in seconds.

        Returns:
            RateLimitResult.
        """
        limit = limit or self.config.default_limit
        window_seconds = window_seconds or self.config.default_window_seconds

        now = time.time()
        window_start = now - window_seconds
        reset_at = now + window_seconds

        return self._check_local(identifier, limit, window_seconds, now, window_start, reset_at)

    async def get_current_count(self, identifier: str) -> int:
        """
        Get current request count for an identifier.

        Args:
            identifier: Unique identifier.

        Returns:
            Current request count in window.
        """
        if self._redis:
            key = f"{self.config.key_prefix}{identifier}"
            now = time.time()
            window_start = now - self.config.default_window_seconds

            try:
                # Remove old entries first
                await self._redis.zremrangebyscore(key, "-inf", window_start)
                return await self._redis.zcard(key)
            except Exception:
                return 0
        else:
            if identifier not in self._local_counts:
                return 0
            now = time.time()
            window_start = now - self.config.default_window_seconds
            return len([ts for ts in self._local_counts[identifier] if ts > window_start])

    async def reset(self, identifier: str) -> bool:
        """
        Reset rate limit for an identifier.

        Args:
            identifier: Unique identifier.

        Returns:
            True if reset was successful.
        """
        if self._redis:
            key = f"{self.config.key_prefix}{identifier}"
            try:
                await self._redis.delete(key)
                return True
            except Exception as e:
                logger.warning(f"Failed to reset rate limit: {e}")
                return False
        else:
            if identifier in self._local_counts:
                del self._local_counts[identifier]
            return True

    def reset_sync(self, identifier: str) -> bool:
        """
        Synchronous reset (local storage only).

        Args:
            identifier: Unique identifier.

        Returns:
            True if reset was successful.
        """
        if identifier in self._local_counts:
            del self._local_counts[identifier]
        return True


# =============================================================================
# Convenience Functions
# =============================================================================


async def check_rate_limit(
    redis_client: Any,
    identifier: str,
    limit: int = 60,
    window_seconds: int = 60,
) -> RateLimitResult:
    """
    Check rate limit for an identifier.

    Args:
        redis_client: Redis client.
        identifier: Unique identifier.
        limit: Maximum requests per window.
        window_seconds: Window size in seconds.

    Returns:
        RateLimitResult.
    """
    limiter = RateLimiter(redis_client)
    return await limiter.check(identifier, limit, window_seconds)
