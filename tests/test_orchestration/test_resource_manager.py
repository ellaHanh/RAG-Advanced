"""
Unit tests for Resource Semaphore Manager.

Tests cover:
- Resource acquisition and release
- Concurrent access limiting
- Timeout handling
- Statistics tracking
"""

from __future__ import annotations

import asyncio

import pytest

from orchestration.resource_manager import (
    ManagerStats,
    ResourceAcquisitionTimeout,
    ResourceLimitConfig,
    ResourceManager,
    ResourceManagerConfig,
    ResourceStats,
    get_resource_manager,
    reset_resource_manager,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def manager() -> ResourceManager:
    """Create a test resource manager."""
    config = ResourceManagerConfig(
        default_limit=5,
        resource_limits={
            "test_resource": ResourceLimitConfig(max_concurrent=2),
            "fast_resource": ResourceLimitConfig(max_concurrent=10),
        },
    )
    return ResourceManager(config)


@pytest.fixture(autouse=True)
def reset_global():
    """Reset global manager before each test."""
    reset_resource_manager()
    yield
    reset_resource_manager()


# =============================================================================
# Test: Configuration
# =============================================================================


class TestConfiguration:
    """Tests for configuration models."""

    def test_default_config(self):
        """Test default configuration."""
        config = ResourceManagerConfig()

        assert config.default_limit == 10
        assert config.default_timeout == 30.0

    def test_resource_limit_config(self):
        """Test resource limit configuration."""
        config = ResourceLimitConfig(max_concurrent=5, timeout_seconds=10.0)

        assert config.max_concurrent == 5
        assert config.timeout_seconds == 10.0


# =============================================================================
# Test: Resource Acquisition
# =============================================================================


class TestResourceAcquisition:
    """Tests for resource acquisition."""

    @pytest.mark.asyncio
    async def test_acquire_and_release(self, manager: ResourceManager):
        """Test basic acquire and release."""
        async with manager.acquire("test_resource"):
            stats = manager.get_stats("test_resource")
            assert stats is not None
            assert stats.current_usage == 1

        # After release
        stats = manager.get_stats("test_resource")
        assert stats.current_usage == 0

    @pytest.mark.asyncio
    async def test_acquire_tracks_statistics(self, manager: ResourceManager):
        """Test that acquisition tracks statistics."""
        async with manager.acquire("test_resource"):
            pass

        stats = manager.get_stats("test_resource")
        assert stats.total_acquisitions == 1
        assert stats.total_releases == 1

    @pytest.mark.asyncio
    async def test_multiple_acquisitions(self, manager: ResourceManager):
        """Test multiple acquisitions to same resource."""
        async with manager.acquire("test_resource"):
            async with manager.acquire("test_resource"):
                stats = manager.get_stats("test_resource")
                assert stats.current_usage == 2

    @pytest.mark.asyncio
    async def test_acquire_different_resources(self, manager: ResourceManager):
        """Test acquiring different resources."""
        async with manager.acquire("test_resource"):
            async with manager.acquire("fast_resource"):
                test_stats = manager.get_stats("test_resource")
                fast_stats = manager.get_stats("fast_resource")

                assert test_stats.current_usage == 1
                assert fast_stats.current_usage == 1


# =============================================================================
# Test: Concurrency Limiting
# =============================================================================


class TestConcurrencyLimiting:
    """Tests for concurrent access limiting."""

    @pytest.mark.asyncio
    async def test_limits_concurrent_access(self, manager: ResourceManager):
        """Test that semaphore limits concurrent access."""
        acquired_count = 0
        max_concurrent = 0

        async def worker():
            nonlocal acquired_count, max_concurrent
            async with manager.acquire("test_resource"):
                acquired_count += 1
                max_concurrent = max(max_concurrent, acquired_count)
                await asyncio.sleep(0.01)
                acquired_count -= 1

        # Start 5 workers with limit of 2
        tasks = [asyncio.create_task(worker()) for _ in range(5)]
        await asyncio.gather(*tasks)

        # Max concurrent should not exceed limit (2)
        assert max_concurrent <= 2

    @pytest.mark.asyncio
    async def test_uses_default_limit_for_unknown_resource(
        self, manager: ResourceManager
    ):
        """Test default limit for unconfigured resources."""
        # Default limit is 5
        async with manager.acquire("unknown_resource"):
            stats = manager.get_stats("unknown_resource")
            assert stats.max_concurrent == 5


# =============================================================================
# Test: Timeout Handling
# =============================================================================


class TestTimeoutHandling:
    """Tests for timeout handling."""

    @pytest.mark.asyncio
    async def test_timeout_raises_error(self):
        """Test that timeout raises appropriate error."""
        # Create manager with very low limit
        config = ResourceManagerConfig(
            resource_limits={
                "limited": ResourceLimitConfig(max_concurrent=1, timeout_seconds=0.1),
            }
        )
        manager = ResourceManager(config)

        # Acquire the only slot
        async with manager.acquire("limited"):
            # Try to acquire again (should timeout)
            with pytest.raises(ResourceAcquisitionTimeout) as exc_info:
                async with manager.acquire("limited"):
                    pass

            assert exc_info.value.resource == "limited"

    @pytest.mark.asyncio
    async def test_timeout_tracked_in_stats(self):
        """Test that timeouts are tracked in statistics."""
        config = ResourceManagerConfig(
            resource_limits={
                "limited": ResourceLimitConfig(max_concurrent=1, timeout_seconds=0.1),
            }
        )
        manager = ResourceManager(config)

        async with manager.acquire("limited"):
            try:
                async with manager.acquire("limited"):
                    pass
            except ResourceAcquisitionTimeout:
                pass

        stats = manager.get_stats("limited")
        assert stats.timeouts == 1


# =============================================================================
# Test: Statistics
# =============================================================================


class TestStatistics:
    """Tests for statistics tracking."""

    def test_resource_stats_utilization(self):
        """Test utilization calculation."""
        stats = ResourceStats(
            resource_name="test",
            max_concurrent=10,
            current_usage=5,
        )

        assert stats.utilization == 0.5

    def test_resource_stats_avg_wait_time(self):
        """Test average wait time calculation."""
        stats = ResourceStats(
            resource_name="test",
            max_concurrent=10,
            total_acquisitions=10,
            wait_time_total_ms=100.0,
        )

        assert stats.avg_wait_time_ms == 10.0

    def test_manager_stats_totals(self):
        """Test manager stats aggregation."""
        stats = ManagerStats(
            resources={
                "a": ResourceStats(
                    resource_name="a",
                    max_concurrent=10,
                    total_acquisitions=5,
                    timeouts=1,
                ),
                "b": ResourceStats(
                    resource_name="b",
                    max_concurrent=10,
                    total_acquisitions=3,
                    timeouts=2,
                ),
            }
        )

        assert stats.total_acquisitions == 8
        assert stats.total_timeouts == 3

    @pytest.mark.asyncio
    async def test_get_all_stats(self, manager: ResourceManager):
        """Test getting all statistics."""
        async with manager.acquire("test_resource"):
            pass
        async with manager.acquire("fast_resource"):
            pass

        all_stats = manager.get_stats()
        assert isinstance(all_stats, ManagerStats)
        assert "test_resource" in all_stats.resources
        assert "fast_resource" in all_stats.resources

    @pytest.mark.asyncio
    async def test_reset_stats(self, manager: ResourceManager):
        """Test resetting statistics."""
        async with manager.acquire("test_resource"):
            pass

        manager.reset_stats()

        stats = manager.get_stats("test_resource")
        assert stats.total_acquisitions == 0


# =============================================================================
# Test: Utility Methods
# =============================================================================


class TestUtilityMethods:
    """Tests for utility methods."""

    @pytest.mark.asyncio
    async def test_get_utilization(self, manager: ResourceManager):
        """Test getting current utilization."""
        async with manager.acquire("test_resource"):
            util = await manager.get_utilization("test_resource")
            assert util == 0.5  # 1 of 2 slots used

    @pytest.mark.asyncio
    async def test_get_available_slots(self, manager: ResourceManager):
        """Test getting available slots."""
        async with manager.acquire("test_resource"):
            available = await manager.get_available_slots("test_resource")
            assert available == 1  # 2 max, 1 used


# =============================================================================
# Test: Global Instance
# =============================================================================


class TestGlobalInstance:
    """Tests for global resource manager."""

    @pytest.mark.asyncio
    async def test_get_resource_manager_singleton(self):
        """Test global manager is singleton."""
        manager1 = await get_resource_manager()
        manager2 = await get_resource_manager()

        assert manager1 is manager2

    @pytest.mark.asyncio
    async def test_reset_resource_manager(self):
        """Test resetting global manager."""
        manager1 = await get_resource_manager()
        async with manager1.acquire("test"):
            pass

        reset_resource_manager()

        manager2 = await get_resource_manager()
        # New manager has no stats for "test"
        assert manager2.get_stats("test") is None
