"""
RAG-Advanced Resource Semaphore Manager.

Limit concurrent access to shared resources using asyncio semaphores.
Manages database connections, API calls, and other rate-limited resources.

Usage:
    from orchestration.resource_manager import ResourceManager

    manager = ResourceManager()
    
    async with manager.acquire("database"):
        # Execute database operation
        await db.query(...)

    async with manager.acquire("openai_api"):
        # Call API with rate limiting
        await client.chat(...)
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator

from pydantic import BaseModel, ConfigDict, Field


logger = logging.getLogger(__name__)


# =============================================================================
# Resource Types
# =============================================================================


class ResourceType(str, Enum):
    """Types of resources that can be managed."""

    DATABASE = "database"
    OPENAI_API = "openai_api"
    ANTHROPIC_API = "anthropic_api"
    EMBEDDING_API = "embedding_api"
    RERANKER = "reranker"
    FILE_IO = "file_io"
    CUSTOM = "custom"


# =============================================================================
# Configuration
# =============================================================================


class ResourceLimitConfig(BaseModel):
    """
    Configuration for a single resource limit.

    Attributes:
        max_concurrent: Maximum concurrent accesses.
        timeout_seconds: Timeout for acquiring resource.
        priority: Priority for scheduling (higher = higher priority).
    """

    model_config = ConfigDict(frozen=True)

    max_concurrent: int = Field(default=10, ge=1, description="Max concurrent")
    timeout_seconds: float = Field(default=30.0, ge=0.0, description="Acquire timeout")
    priority: int = Field(default=0, ge=0, description="Scheduling priority")


class ResourceManagerConfig(BaseModel):
    """
    Configuration for the resource manager.

    Attributes:
        default_limit: Default concurrent access limit.
        default_timeout: Default timeout for acquiring.
        resource_limits: Per-resource limit configurations.
    """

    model_config = ConfigDict(frozen=True)

    default_limit: int = Field(default=10, ge=1, description="Default limit")
    default_timeout: float = Field(default=30.0, ge=0.0, description="Default timeout")
    resource_limits: dict[str, ResourceLimitConfig] = Field(
        default_factory=lambda: {
            "database": ResourceLimitConfig(max_concurrent=20),
            "openai_api": ResourceLimitConfig(max_concurrent=50),
            "anthropic_api": ResourceLimitConfig(max_concurrent=20),
            "embedding_api": ResourceLimitConfig(max_concurrent=100),
            "reranker": ResourceLimitConfig(max_concurrent=5),
            "file_io": ResourceLimitConfig(max_concurrent=10),
        },
        description="Per-resource limits",
    )


# =============================================================================
# Statistics
# =============================================================================


@dataclass
class ResourceStats:
    """
    Statistics for a single resource.

    Attributes:
        resource_name: Name of the resource.
        max_concurrent: Maximum allowed concurrent.
        current_usage: Current usage count.
        total_acquisitions: Total successful acquisitions.
        total_releases: Total releases.
        timeouts: Number of acquisition timeouts.
        wait_time_total_ms: Total time spent waiting.
    """

    resource_name: str
    max_concurrent: int
    current_usage: int = 0
    total_acquisitions: int = 0
    total_releases: int = 0
    timeouts: int = 0
    wait_time_total_ms: float = 0.0

    @property
    def utilization(self) -> float:
        """Get current utilization (0.0 to 1.0)."""
        if self.max_concurrent == 0:
            return 0.0
        return self.current_usage / self.max_concurrent

    @property
    def avg_wait_time_ms(self) -> float:
        """Get average wait time."""
        if self.total_acquisitions == 0:
            return 0.0
        return self.wait_time_total_ms / self.total_acquisitions


@dataclass
class ManagerStats:
    """
    Statistics for the resource manager.

    Attributes:
        resources: Statistics per resource.
    """

    resources: dict[str, ResourceStats] = field(default_factory=dict)

    @property
    def total_acquisitions(self) -> int:
        """Get total acquisitions across all resources."""
        return sum(r.total_acquisitions for r in self.resources.values())

    @property
    def total_timeouts(self) -> int:
        """Get total timeouts across all resources."""
        return sum(r.timeouts for r in self.resources.values())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_acquisitions": self.total_acquisitions,
            "total_timeouts": self.total_timeouts,
            "resources": {
                name: {
                    "max_concurrent": stats.max_concurrent,
                    "current_usage": stats.current_usage,
                    "utilization": stats.utilization,
                    "total_acquisitions": stats.total_acquisitions,
                    "avg_wait_time_ms": stats.avg_wait_time_ms,
                }
                for name, stats in self.resources.items()
            },
        }


# =============================================================================
# Resource Manager
# =============================================================================


class ResourceAcquisitionTimeout(Exception):
    """Raised when resource acquisition times out."""

    def __init__(self, resource: str, timeout: float) -> None:
        self.resource = resource
        self.timeout = timeout
        super().__init__(f"Timeout acquiring resource '{resource}' after {timeout}s")


class ResourceManager:
    """
    Manage concurrent access to shared resources using semaphores.

    Thread-safe manager for controlling access to databases, APIs,
    and other rate-limited resources.

    Example:
        >>> manager = ResourceManager()
        >>> async with manager.acquire("database"):
        ...     await db.execute("SELECT ...")
    """

    def __init__(self, config: ResourceManagerConfig | None = None) -> None:
        """
        Initialize the resource manager.

        Args:
            config: Optional configuration.
        """
        self._config = config or ResourceManagerConfig()
        self._semaphores: dict[str, asyncio.Semaphore] = {}
        self._stats: dict[str, ResourceStats] = {}
        self._lock = asyncio.Lock()

    async def _get_or_create_semaphore(
        self,
        resource: str,
    ) -> tuple[asyncio.Semaphore, int]:
        """Get or create semaphore for resource."""
        async with self._lock:
            if resource not in self._semaphores:
                # Get limit from config
                if resource in self._config.resource_limits:
                    limit = self._config.resource_limits[resource].max_concurrent
                else:
                    limit = self._config.default_limit

                self._semaphores[resource] = asyncio.Semaphore(limit)
                self._stats[resource] = ResourceStats(
                    resource_name=resource,
                    max_concurrent=limit,
                )

            return self._semaphores[resource], self._stats[resource].max_concurrent

    def _get_timeout(self, resource: str) -> float:
        """Get timeout for resource."""
        if resource in self._config.resource_limits:
            return self._config.resource_limits[resource].timeout_seconds
        return self._config.default_timeout

    @asynccontextmanager
    async def acquire(
        self,
        resource: str,
        timeout: float | None = None,
    ) -> AsyncIterator[None]:
        """
        Acquire access to a resource.

        Args:
            resource: Resource name to acquire.
            timeout: Optional timeout override.

        Yields:
            None when resource is acquired.

        Raises:
            ResourceAcquisitionTimeout: If timeout is exceeded.
        """
        semaphore, _ = await self._get_or_create_semaphore(resource)
        timeout = timeout if timeout is not None else self._get_timeout(resource)

        import time
        start_time = time.time()

        try:
            acquired = await asyncio.wait_for(
                semaphore.acquire(),
                timeout=timeout if timeout > 0 else None,
            )

            if not acquired:
                raise ResourceAcquisitionTimeout(resource, timeout)

        except asyncio.TimeoutError:
            async with self._lock:
                self._stats[resource].timeouts += 1
            raise ResourceAcquisitionTimeout(resource, timeout)

        # Track acquisition
        async with self._lock:
            wait_time = (time.time() - start_time) * 1000
            self._stats[resource].current_usage += 1
            self._stats[resource].total_acquisitions += 1
            self._stats[resource].wait_time_total_ms += wait_time

        try:
            yield
        finally:
            # Release resource
            semaphore.release()
            async with self._lock:
                self._stats[resource].current_usage -= 1
                self._stats[resource].total_releases += 1

    async def acquire_multiple(
        self,
        resources: list[str],
    ) -> asyncio.Task[None]:
        """
        Acquire multiple resources atomically.

        Args:
            resources: List of resource names.

        Returns:
            Task that must be awaited and cancelled to release.

        Note:
            Use with AsyncExitStack for proper cleanup:
            
            async with AsyncExitStack() as stack:
                for resource in resources:
                    await stack.enter_async_context(manager.acquire(resource))
                # All resources acquired
        """
        # This is a placeholder - real implementation should use AsyncExitStack
        raise NotImplementedError("Use AsyncExitStack for multiple resource acquisition")

    def get_stats(self, resource: str | None = None) -> ManagerStats | ResourceStats | None:
        """
        Get statistics.

        Args:
            resource: Specific resource name (None = all).

        Returns:
            Statistics for resource or manager.
        """
        if resource:
            return self._stats.get(resource)
        return ManagerStats(resources=self._stats.copy())

    def reset_stats(self) -> None:
        """Reset all statistics."""
        for stats in self._stats.values():
            stats.total_acquisitions = 0
            stats.total_releases = 0
            stats.timeouts = 0
            stats.wait_time_total_ms = 0.0

    async def get_utilization(self, resource: str) -> float:
        """
        Get current utilization for a resource.

        Args:
            resource: Resource name.

        Returns:
            Utilization between 0.0 and 1.0.
        """
        stats = self._stats.get(resource)
        if stats:
            return stats.utilization
        return 0.0

    async def get_available_slots(self, resource: str) -> int:
        """
        Get number of available slots for a resource.

        Args:
            resource: Resource name.

        Returns:
            Number of available slots.
        """
        if resource not in self._semaphores:
            if resource in self._config.resource_limits:
                return self._config.resource_limits[resource].max_concurrent
            return self._config.default_limit

        stats = self._stats.get(resource)
        if stats:
            return stats.max_concurrent - stats.current_usage
        return 0


# =============================================================================
# Global Instance
# =============================================================================

_global_manager: ResourceManager | None = None
_global_lock = asyncio.Lock()


async def get_resource_manager(
    config: ResourceManagerConfig | None = None,
) -> ResourceManager:
    """
    Get or create the global resource manager.

    Args:
        config: Optional configuration (only used on first call).

    Returns:
        Global ResourceManager instance.
    """
    global _global_manager

    if _global_manager is None:
        async with _global_lock:
            if _global_manager is None:
                _global_manager = ResourceManager(config)

    return _global_manager


def reset_resource_manager() -> None:
    """Reset the global resource manager."""
    global _global_manager
    _global_manager = None
