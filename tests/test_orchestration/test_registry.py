"""
Unit tests for the Strategy Registry.

Tests cover:
- Strategy registration (decorator and manual)
- Strategy lookup
- Duplicate registration handling
- Validation of strategy functions
- Listing and filtering strategies
- Singleton pattern
"""

from __future__ import annotations

import pytest

from orchestration.errors import (
    InvalidStrategyError,
    StrategyAlreadyRegisteredError,
    StrategyNotFoundError,
)
from orchestration.models import (
    Document,
    ExecutionResult,
    ResourceType,
    StrategyConfig,
    StrategyMetadata,
    StrategyType,
)
from orchestration.registry import (
    StrategyRegistry,
    get_registry,
    get_strategy,
    get_strategy_metadata,
    list_strategies,
    list_strategy_names,
    register_strategy,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def reset_registry():
    """Reset the registry before and after each test."""
    StrategyRegistry.reset()
    yield
    StrategyRegistry.reset()


@pytest.fixture
def sample_metadata() -> StrategyMetadata:
    """Create sample strategy metadata."""
    return StrategyMetadata(
        name="test_strategy",
        description="A test strategy",
        strategy_type=StrategyType.STANDARD,
        version="1.0.0",
        required_resources=[ResourceType.DATABASE],
        estimated_latency_ms=(100, 300),
        estimated_cost_per_query=0.001,
        precision_rating=3,
        tags=["test", "sample"],
    )


@pytest.fixture
def sample_result() -> ExecutionResult:
    """Create a sample execution result."""
    return ExecutionResult(
        documents=[
            Document(id="doc_1", content="Test content", title="Test", source="test.md")
        ],
        query="test query",
        strategy_name="test_strategy",
        latency_ms=100,
        cost_usd=0.001,
    )


# =============================================================================
# Helper Functions
# =============================================================================


async def dummy_strategy(query: str, config: StrategyConfig | None = None) -> ExecutionResult:
    """A dummy async strategy function for testing."""
    return ExecutionResult(
        documents=[],
        query=query,
        strategy_name="dummy",
        latency_ms=10,
        cost_usd=0.0001,
    )


async def another_strategy(query: str) -> ExecutionResult:
    """Another dummy strategy with minimal signature."""
    return ExecutionResult(
        documents=[],
        query=query,
        strategy_name="another",
        latency_ms=5,
        cost_usd=0.00005,
    )


def sync_function(query: str) -> str:
    """A synchronous function (should fail validation)."""
    return query


# =============================================================================
# Test: Strategy Registration
# =============================================================================


class TestStrategyRegistration:
    """Tests for strategy registration functionality."""

    def test_register_strategy_manually(self, sample_metadata: StrategyMetadata):
        """Test manual strategy registration."""
        registry = StrategyRegistry()
        registry.register("test", dummy_strategy, sample_metadata)

        assert "test" in registry
        assert registry.is_registered("test")
        assert len(registry) == 1

    def test_register_strategy_without_metadata(self):
        """Test registration without explicit metadata creates default."""
        registry = StrategyRegistry()
        registry.register("test", dummy_strategy)

        assert "test" in registry
        metadata = registry.get_metadata("test")
        assert metadata.name == "test"
        assert metadata.strategy_type == StrategyType.STANDARD

    def test_register_strategy_with_decorator(self):
        """Test decorator-based registration."""

        @register_strategy(
            name="decorated",
            description="A decorated strategy",
            strategy_type=StrategyType.RERANKING,
            precision_rating=5,
            tags=["decorator", "test"],
        )
        async def decorated_strategy(query: str) -> ExecutionResult:
            return ExecutionResult(
                documents=[],
                query=query,
                strategy_name="decorated",
            )

        registry = get_registry()
        assert "decorated" in registry
        metadata = registry.get_metadata("decorated")
        assert metadata.description == "A decorated strategy"
        assert metadata.strategy_type == StrategyType.RERANKING
        assert metadata.precision_rating == 5
        assert "decorator" in metadata.tags

    def test_register_normalizes_name(self):
        """Test that strategy names are normalized (lowercase, trimmed)."""
        registry = StrategyRegistry()
        registry.register("  TEST_Strategy  ", dummy_strategy)

        assert "test_strategy" in registry
        assert registry.is_registered("TEST_STRATEGY")
        assert registry.is_registered("  test_strategy  ")

    def test_duplicate_registration_raises_error(self):
        """Test that duplicate registration raises error."""
        registry = StrategyRegistry()
        registry.register("test", dummy_strategy)

        with pytest.raises(StrategyAlreadyRegisteredError) as exc_info:
            registry.register("test", another_strategy)

        assert exc_info.value.strategy_name == "test"

    def test_duplicate_registration_with_override(self):
        """Test that duplicate registration works with allow_override=True."""
        registry = StrategyRegistry()
        registry.register("test", dummy_strategy)
        registry.register("test", another_strategy, allow_override=True)

        # Should have the new strategy
        strategy = registry.get("test")
        assert strategy == another_strategy


# =============================================================================
# Test: Strategy Validation
# =============================================================================


class TestStrategyValidation:
    """Tests for strategy function validation."""

    def test_validate_rejects_non_callable(self):
        """Test that non-callable objects are rejected."""
        registry = StrategyRegistry()

        with pytest.raises(InvalidStrategyError) as exc_info:
            registry.register("test", "not a function")  # type: ignore

        assert "must be callable" in exc_info.value.reason

    def test_validate_rejects_sync_function(self):
        """Test that synchronous functions are rejected."""
        registry = StrategyRegistry()

        with pytest.raises(InvalidStrategyError) as exc_info:
            registry.register("test", sync_function)

        assert "async function" in exc_info.value.reason

    def test_validate_accepts_async_function(self):
        """Test that async functions are accepted."""
        registry = StrategyRegistry()

        # Should not raise
        registry.register("test", dummy_strategy)
        assert "test" in registry


# =============================================================================
# Test: Strategy Lookup
# =============================================================================


class TestStrategyLookup:
    """Tests for strategy lookup functionality."""

    def test_get_registered_strategy(self):
        """Test getting a registered strategy."""
        registry = StrategyRegistry()
        registry.register("test", dummy_strategy)

        strategy = registry.get("test")
        assert strategy == dummy_strategy

    def test_get_unregistered_strategy_raises_error(self):
        """Test that getting unregistered strategy raises error."""
        registry = StrategyRegistry()

        with pytest.raises(StrategyNotFoundError) as exc_info:
            registry.get("nonexistent")

        assert exc_info.value.strategy_name == "nonexistent"
        assert exc_info.value.available_strategies == []

    def test_get_strategy_shows_available(self):
        """Test that error includes available strategies."""
        registry = StrategyRegistry()
        registry.register("one", dummy_strategy)
        registry.register("two", another_strategy)

        with pytest.raises(StrategyNotFoundError) as exc_info:
            registry.get("three")

        assert set(exc_info.value.available_strategies) == {"one", "two"}

    def test_get_metadata(self, sample_metadata: StrategyMetadata):
        """Test getting strategy metadata."""
        registry = StrategyRegistry()
        registry.register("test", dummy_strategy, sample_metadata)

        metadata = registry.get_metadata("test")
        assert metadata.description == sample_metadata.description
        assert metadata.strategy_type == sample_metadata.strategy_type

    def test_get_metadata_unregistered_raises_error(self):
        """Test that getting metadata for unregistered strategy raises error."""
        registry = StrategyRegistry()

        with pytest.raises(StrategyNotFoundError):
            registry.get_metadata("nonexistent")


# =============================================================================
# Test: Strategy Listing
# =============================================================================


class TestStrategyListing:
    """Tests for strategy listing functionality."""

    def test_list_strategies_empty(self):
        """Test listing with no registered strategies."""
        registry = StrategyRegistry()
        assert registry.list_strategies() == []
        assert registry.list_strategy_names() == []

    def test_list_strategies(self, sample_metadata: StrategyMetadata):
        """Test listing registered strategies."""
        registry = StrategyRegistry()
        registry.register("test", dummy_strategy, sample_metadata)
        registry.register("another", another_strategy)

        strategies = registry.list_strategies()
        assert len(strategies) == 2

        names = registry.list_strategy_names()
        assert set(names) == {"test", "another"}

    def test_filter_by_type(self):
        """Test filtering strategies by type."""
        registry = StrategyRegistry()

        registry.register(
            "standard1",
            dummy_strategy,
            StrategyMetadata(
                name="standard1",
                description="Standard 1",
                strategy_type=StrategyType.STANDARD,
            ),
        )
        registry.register(
            "reranking1",
            another_strategy,
            StrategyMetadata(
                name="reranking1",
                description="Reranking 1",
                strategy_type=StrategyType.RERANKING,
            ),
        )

        standard = registry.filter_by_type(StrategyType.STANDARD)
        assert len(standard) == 1
        assert standard[0].name == "standard1"

        reranking = registry.filter_by_type(StrategyType.RERANKING)
        assert len(reranking) == 1
        assert reranking[0].name == "reranking1"

    def test_filter_by_resource(self):
        """Test filtering strategies by required resource."""
        registry = StrategyRegistry()

        registry.register(
            "with_db",
            dummy_strategy,
            StrategyMetadata(
                name="with_db",
                description="Needs DB",
                strategy_type=StrategyType.STANDARD,
                required_resources=[ResourceType.DATABASE],
            ),
        )
        registry.register(
            "with_reranker",
            another_strategy,
            StrategyMetadata(
                name="with_reranker",
                description="Needs reranker",
                strategy_type=StrategyType.RERANKING,
                required_resources=[ResourceType.RERANKER, ResourceType.DATABASE],
            ),
        )

        db_strategies = registry.filter_by_resource(ResourceType.DATABASE)
        assert len(db_strategies) == 2

        reranker_strategies = registry.filter_by_resource(ResourceType.RERANKER)
        assert len(reranker_strategies) == 1
        assert reranker_strategies[0].name == "with_reranker"

    def test_filter_by_tag(self):
        """Test filtering strategies by tag."""
        registry = StrategyRegistry()

        registry.register(
            "tagged",
            dummy_strategy,
            StrategyMetadata(
                name="tagged",
                description="Tagged strategy",
                strategy_type=StrategyType.STANDARD,
                tags=["fast", "simple"],
            ),
        )
        registry.register(
            "untagged",
            another_strategy,
            StrategyMetadata(
                name="untagged",
                description="Untagged",
                strategy_type=StrategyType.STANDARD,
            ),
        )

        fast = registry.filter_by_tag("fast")
        assert len(fast) == 1
        assert fast[0].name == "tagged"

        # Case insensitive
        fast_upper = registry.filter_by_tag("FAST")
        assert len(fast_upper) == 1


# =============================================================================
# Test: Unregistration
# =============================================================================


class TestUnregistration:
    """Tests for strategy unregistration."""

    def test_unregister_existing(self):
        """Test unregistering an existing strategy."""
        registry = StrategyRegistry()
        registry.register("test", dummy_strategy)
        assert "test" in registry

        result = registry.unregister("test")
        assert result is True
        assert "test" not in registry

    def test_unregister_nonexistent(self):
        """Test unregistering a nonexistent strategy."""
        registry = StrategyRegistry()

        result = registry.unregister("nonexistent")
        assert result is False


# =============================================================================
# Test: Singleton Pattern
# =============================================================================


class TestSingletonPattern:
    """Tests for singleton behavior."""

    def test_singleton_returns_same_instance(self):
        """Test that StrategyRegistry returns the same instance."""
        registry1 = StrategyRegistry()
        registry2 = StrategyRegistry()

        assert registry1 is registry2

    def test_singleton_shares_state(self):
        """Test that singleton instances share state."""
        registry1 = StrategyRegistry()
        registry1.register("shared", dummy_strategy)

        registry2 = StrategyRegistry()
        assert "shared" in registry2

    def test_get_registry_returns_singleton(self):
        """Test that get_registry returns the singleton."""
        registry = StrategyRegistry()
        global_registry = get_registry()

        assert registry is global_registry

    def test_reset_clears_singleton(self):
        """Test that reset clears the singleton state."""
        registry = StrategyRegistry()
        registry.register("test", dummy_strategy)
        assert len(registry) == 1

        StrategyRegistry.reset()

        # New instance should be empty
        new_registry = StrategyRegistry()
        assert len(new_registry) == 0


# =============================================================================
# Test: Convenience Functions
# =============================================================================


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_get_strategy_function(self):
        """Test get_strategy convenience function."""
        registry = get_registry()
        registry.register("test", dummy_strategy)

        strategy = get_strategy("test")
        assert strategy == dummy_strategy

    def test_get_strategy_metadata_function(self, sample_metadata: StrategyMetadata):
        """Test get_strategy_metadata convenience function."""
        registry = get_registry()
        registry.register("test", dummy_strategy, sample_metadata)

        metadata = get_strategy_metadata("test")
        assert metadata.name == "test"

    def test_list_strategies_function(self):
        """Test list_strategies convenience function."""
        registry = get_registry()
        registry.register("test", dummy_strategy)

        strategies = list_strategies()
        assert len(strategies) == 1

    def test_list_strategy_names_function(self):
        """Test list_strategy_names convenience function."""
        registry = get_registry()
        registry.register("test", dummy_strategy)
        registry.register("another", another_strategy)

        names = list_strategy_names()
        assert set(names) == {"test", "another"}


# =============================================================================
# Test: Iterator Protocol
# =============================================================================


class TestIteratorProtocol:
    """Tests for iterator protocol support."""

    def test_iterate_over_registry(self):
        """Test iterating over registry."""
        registry = StrategyRegistry()
        registry.register("one", dummy_strategy)
        registry.register("two", another_strategy)

        names = list(registry)
        assert set(names) == {"one", "two"}

    def test_len_returns_count(self):
        """Test len() returns strategy count."""
        registry = StrategyRegistry()
        assert len(registry) == 0

        registry.register("one", dummy_strategy)
        assert len(registry) == 1

        registry.register("two", another_strategy)
        assert len(registry) == 2

    def test_in_operator(self):
        """Test 'in' operator for membership check."""
        registry = StrategyRegistry()
        registry.register("test", dummy_strategy)

        assert "test" in registry
        assert "other" not in registry


# =============================================================================
# Test: Async Execution
# =============================================================================


class TestAsyncExecution:
    """Tests for async strategy execution."""

    @pytest.mark.asyncio
    async def test_execute_registered_strategy(self):
        """Test executing a registered strategy."""
        registry = StrategyRegistry()
        registry.register("test", dummy_strategy)

        strategy = registry.get("test")
        result = await strategy("test query", StrategyConfig())

        assert result.query == "test query"
        assert result.strategy_name == "dummy"

    @pytest.mark.asyncio
    async def test_execute_decorated_strategy(self):
        """Test executing a decorated strategy."""

        @register_strategy(
            name="exec_test",
            description="Execution test",
            strategy_type=StrategyType.STANDARD,
        )
        async def exec_strategy(query: str) -> ExecutionResult:
            return ExecutionResult(
                documents=[Document(id="1", content="Result", title="Test", source="test")],
                query=query,
                strategy_name="exec_test",
                latency_ms=50,
            )

        strategy = get_strategy("exec_test")
        result = await strategy("my query")

        assert result.query == "my query"
        assert len(result.documents) == 1
        assert result.documents[0].content == "Result"
