"""
Unit tests for Single Strategy Executor.

Tests cover:
- Strategy execution
- Metrics tracking
- Timeout handling
- Retry logic
- Error handling
"""

from __future__ import annotations

import asyncio

import pytest

from orchestration.cost_tracker import CostTracker
from orchestration.errors import (
    StrategyExecutionError,
    StrategyNotFoundError,
    StrategyTimeoutError,
)
from orchestration.executor import (
    ExecutionContext,
    ExecutorConfig,
    ParallelExecutionResult,
    ParallelExecutor,
    StrategyExecutor,
    execute_strategies_parallel,
    execute_strategy,
)
from orchestration.models import Document, ExecutionResult, StrategyConfig
from orchestration.registry import StrategyRegistry


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="function")
def registry() -> StrategyRegistry:
    """Create a fresh test registry with mock strategies for each test."""
    # Reset singleton to get fresh registry
    StrategyRegistry.reset()
    registry = StrategyRegistry()

    # Fast strategy
    async def fast_strategy(ctx: ExecutionContext) -> list[Document]:
        return [
            Document(
                id="doc1",
                content="Test content",
                title="Test Doc",
                source="test.md",
                similarity=0.95,
            )
        ]

    # Slow strategy
    async def slow_strategy(ctx: ExecutionContext) -> list[Document]:
        await asyncio.sleep(5)
        return []

    # Failing strategy
    async def failing_strategy(ctx: ExecutionContext) -> list[Document]:
        raise ValueError("Strategy failed intentionally")

    # Configurable strategy
    async def configurable_strategy(ctx: ExecutionContext) -> list[Document]:
        limit = ctx.config.limit
        return [
            Document(
                id=f"doc{i}",
                content=f"Content {i}",
                title=f"Doc {i}",
                source="test.md",
                similarity=0.9 - i * 0.1,
            )
            for i in range(limit)
        ]

    # Strategy with costs
    async def costly_strategy(ctx: ExecutionContext) -> list[Document]:
        ctx.add_embedding_cost("text-embedding-3-small", 100)
        ctx.add_llm_cost("gpt-4o-mini", 500, 100)
        return [
            Document(
                id="doc1",
                content="Costly result",
                title="Costly",
                source="test.md",
                similarity=0.9,
            )
        ]

    registry.register("fast", fast_strategy)
    registry.register("slow", slow_strategy)
    registry.register("failing", failing_strategy)
    registry.register("configurable", configurable_strategy)
    registry.register("costly", costly_strategy)

    return registry


@pytest.fixture
def executor(registry: StrategyRegistry) -> StrategyExecutor:
    """Create a test executor."""
    return StrategyExecutor(registry=registry)


# =============================================================================
# Test: ExecutorConfig
# =============================================================================


class TestExecutorConfig:
    """Tests for ExecutorConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = ExecutorConfig()

        assert config.default_timeout_seconds == 60.0
        assert config.track_costs is True
        assert config.max_retries == 0

    def test_custom_config(self):
        """Test custom configuration."""
        config = ExecutorConfig(
            default_timeout_seconds=30.0,
            max_retries=2,
        )

        assert config.default_timeout_seconds == 30.0
        assert config.max_retries == 2


# =============================================================================
# Test: ExecutionContext
# =============================================================================


class TestExecutionContext:
    """Tests for ExecutionContext."""

    def test_context_creation(self):
        """Test context creation."""
        config = StrategyConfig(limit=5)
        cost_tracker = CostTracker()

        ctx = ExecutionContext(
            query="test query",
            config=config,
            cost_tracker=cost_tracker,
        )

        assert ctx.query == "test query"
        assert ctx.config.limit == 5

    def test_add_embedding_cost(self):
        """Test adding embedding cost."""
        ctx = ExecutionContext(
            query="test",
            config=StrategyConfig(),
            cost_tracker=CostTracker(),
        )

        cost = ctx.add_embedding_cost("text-embedding-3-small", 100)

        assert cost > 0
        assert ctx.cost_tracker.get_summary().total_cost > 0

    def test_add_llm_cost(self):
        """Test adding LLM cost."""
        ctx = ExecutionContext(
            query="test",
            config=StrategyConfig(),
            cost_tracker=CostTracker(),
        )

        cost = ctx.add_llm_cost("gpt-4o-mini", 500, 100)

        assert cost > 0


# =============================================================================
# Test: Basic Execution
# =============================================================================


class TestBasicExecution:
    """Tests for basic strategy execution."""

    @pytest.mark.asyncio
    async def test_execute_fast_strategy(
        self,
        executor: StrategyExecutor,
    ):
        """Test executing a fast strategy."""
        result = await executor.execute("fast", "test query")

        assert isinstance(result, ExecutionResult)
        assert len(result.documents) == 1
        assert result.strategy_name == "fast"
        assert result.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_execute_returns_correct_documents(
        self,
        executor: StrategyExecutor,
    ):
        """Test that execution returns correct documents."""
        result = await executor.execute("fast", "test")

        assert result.documents[0].id == "doc1"
        assert result.documents[0].content == "Test content"

    @pytest.mark.asyncio
    async def test_execute_with_config(
        self,
        executor: StrategyExecutor,
    ):
        """Test execution with configuration."""
        config = StrategyConfig(limit=3)
        result = await executor.execute("configurable", "test", config)

        assert len(result.documents) == 3

    @pytest.mark.asyncio
    async def test_execute_tracks_query(
        self,
        executor: StrategyExecutor,
    ):
        """Test that query is tracked in result."""
        result = await executor.execute("fast", "What is AI?")

        assert result.query == "What is AI?"


# =============================================================================
# Test: Cost Tracking
# =============================================================================


class TestCostTracking:
    """Tests for cost tracking during execution."""

    @pytest.mark.asyncio
    async def test_costs_tracked(
        self,
        executor: StrategyExecutor,
    ):
        """Test that costs are tracked."""
        result = await executor.execute("costly", "test")

        assert result.cost_usd > 0

    @pytest.mark.asyncio
    async def test_token_counts_tracked(
        self,
        executor: StrategyExecutor,
    ):
        """Test that token counts are tracked."""
        result = await executor.execute("costly", "test")

        assert result.token_counts is not None


# =============================================================================
# Test: Timeout Handling
# =============================================================================


class TestTimeoutHandling:
    """Tests for timeout handling."""

    @pytest.mark.asyncio
    async def test_timeout_raises_error(
        self,
        executor: StrategyExecutor,
    ):
        """Test that timeout raises appropriate error."""
        with pytest.raises(StrategyTimeoutError) as exc_info:
            await executor.execute("slow", "test", timeout=0.1)

        assert exc_info.value.strategy_name == "slow"

    @pytest.mark.asyncio
    async def test_fast_strategy_does_not_timeout(
        self,
        executor: StrategyExecutor,
    ):
        """Test that fast strategy completes before timeout."""
        result = await executor.execute("fast", "test", timeout=10.0)

        assert len(result.documents) == 1


# =============================================================================
# Test: Error Handling
# =============================================================================


class TestErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_unknown_strategy_raises_error(
        self,
        executor: StrategyExecutor,
    ):
        """Test that unknown strategy raises error."""
        with pytest.raises(StrategyNotFoundError) as exc_info:
            await executor.execute("nonexistent", "test")

        assert exc_info.value.strategy_name == "nonexistent"

    @pytest.mark.asyncio
    async def test_failing_strategy_raises_error(
        self,
        executor: StrategyExecutor,
    ):
        """Test that strategy failure raises error."""
        with pytest.raises(StrategyExecutionError) as exc_info:
            await executor.execute("failing", "test")

        assert exc_info.value.strategy_name == "failing"


# =============================================================================
# Test: Retry Logic
# =============================================================================


class TestRetryLogic:
    """Tests for retry logic."""

    @pytest.mark.asyncio
    async def test_retry_on_failure(self):
        """Test retry on strategy failure."""
        # Create registry with initially failing then succeeding strategy
        registry = StrategyRegistry()
        call_count = 0

        async def flaky_strategy(ctx: ExecutionContext) -> list[Document]:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary failure")
            return [
                Document(
                    id="doc1",
                    content="Success",
                    title="Test",
                    source="test.md",
                    similarity=0.9,
                )
            ]

        registry.register("flaky", flaky_strategy)

        config = ExecutorConfig(max_retries=3, retry_delay_seconds=0.01)
        executor = StrategyExecutor(registry=registry, config=config)

        result = await executor.execute_with_retry("flaky", "test")

        assert len(result.documents) == 1
        assert call_count == 3  # Failed twice, succeeded on third

    @pytest.mark.asyncio
    async def test_retry_exhausted_raises_error(self):
        """Test that exhausted retries raise error."""
        registry = StrategyRegistry()

        async def always_failing(ctx: ExecutionContext) -> list[Document]:
            raise ValueError("Always fails")

        registry.register("always_failing", always_failing)

        config = ExecutorConfig(max_retries=2, retry_delay_seconds=0.01)
        executor = StrategyExecutor(registry=registry, config=config)

        with pytest.raises(StrategyExecutionError):
            await executor.execute_with_retry("always_failing", "test")


# =============================================================================
# Test: Synchronous Execution
# =============================================================================


class TestSynchronousExecution:
    """Tests for synchronous execution wrapper."""

    def test_execute_sync(
        self,
        executor: StrategyExecutor,
    ):
        """Test synchronous execution."""
        result = executor.execute_sync("fast", "test query")

        assert isinstance(result, ExecutionResult)
        assert len(result.documents) == 1


# =============================================================================
# Test: Convenience Functions
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    @pytest.mark.asyncio
    async def test_execute_strategy_function(
        self,
        registry: StrategyRegistry,
    ):
        """Test execute_strategy convenience function."""
        # The registry fixture already reset and populated the singleton
        result = await execute_strategy("fast", "test")
        assert len(result.documents) == 1


# =============================================================================
# Test: Metadata
# =============================================================================


class TestMetadata:
    """Tests for execution metadata."""

    @pytest.mark.asyncio
    async def test_metadata_included_in_result(
        self,
        executor: StrategyExecutor,
    ):
        """Test that metadata is included in result."""
        result = await executor.execute(
            "fast",
            "test",
            metadata={"source": "api", "user_id": "123"},
        )

        assert result.metadata["source"] == "api"
        assert result.metadata["user_id"] == "123"

    @pytest.mark.asyncio
    async def test_config_in_metadata(
        self,
        executor: StrategyExecutor,
    ):
        """Test that config is included in metadata."""
        config = StrategyConfig(limit=5)
        result = await executor.execute("fast", "test", config)

        assert "config" in result.metadata
        assert result.metadata["config"]["limit"] == 5


# =============================================================================
# Test: ParallelExecutionResult
# =============================================================================


class TestParallelExecutionResult:
    """Tests for ParallelExecutionResult."""

    def test_successful_count(self):
        """Test successful count calculation."""
        result = ParallelExecutionResult()
        result.results["a"] = ExecutionResult(
            documents=[],
            query="test",
            strategy_name="a",
            latency_ms=100,
            cost_usd=0.01,
        )
        result.results["b"] = ExecutionResult(
            documents=[],
            query="test",
            strategy_name="b",
            latency_ms=100,
            cost_usd=0.01,
        )

        assert result.successful_count == 2

    def test_failed_count(self):
        """Test failed count calculation."""
        result = ParallelExecutionResult()
        result.errors["a"] = ValueError("Failed")
        result.errors["b"] = ValueError("Failed")

        assert result.failed_count == 2

    def test_all_succeeded(self):
        """Test all_succeeded property."""
        result = ParallelExecutionResult()
        result.results["a"] = ExecutionResult(
            documents=[],
            query="test",
            strategy_name="a",
            latency_ms=100,
            cost_usd=0.01,
        )

        assert result.all_succeeded is True

        result.errors["b"] = ValueError("Failed")
        assert result.all_succeeded is False


# =============================================================================
# Test: ParallelExecutor
# =============================================================================


class TestParallelExecutor:
    """Tests for ParallelExecutor."""

    @pytest.mark.asyncio
    async def test_execute_all_strategies(
        self,
        executor: StrategyExecutor,
    ):
        """Test executing all strategies concurrently."""
        parallel = ParallelExecutor(executor=executor)

        result = await parallel.execute_all(
            strategies=["fast", "configurable"],
            query="test query",
        )

        assert result.successful_count == 2
        assert "fast" in result.results
        assert "configurable" in result.results
        assert result.all_succeeded

    @pytest.mark.asyncio
    async def test_execute_all_with_failure(
        self,
        executor: StrategyExecutor,
    ):
        """Test execution with some failures."""
        parallel = ParallelExecutor(executor=executor)

        result = await parallel.execute_all(
            strategies=["fast", "failing"],
            query="test",
        )

        assert result.successful_count == 1
        assert result.failed_count == 1
        assert "fast" in result.results
        assert "failing" in result.errors

    @pytest.mark.asyncio
    async def test_total_cost_aggregation(
        self,
        executor: StrategyExecutor,
    ):
        """Test total cost aggregation."""
        parallel = ParallelExecutor(executor=executor)

        result = await parallel.execute_all(
            strategies=["costly", "costly"],
            query="test",
        )

        # Both costly strategies should contribute to total
        assert result.total_cost_usd > 0

    @pytest.mark.asyncio
    async def test_execute_with_concurrency_limit(
        self,
        executor: StrategyExecutor,
    ):
        """Test execution with concurrency limit."""
        parallel = ParallelExecutor(executor=executor, max_concurrency=2)

        # Use different strategies since results are keyed by strategy name
        result = await parallel.execute_all(
            strategies=["fast", "configurable", "costly"],
            query="test",
        )

        assert result.successful_count == 3

    @pytest.mark.asyncio
    async def test_execute_first_success(
        self,
        executor: StrategyExecutor,
    ):
        """Test getting first successful result."""
        parallel = ParallelExecutor(executor=executor)

        result = await parallel.execute_first_success(
            strategies=["fast", "slow"],
            query="test",
            timeout=10.0,
        )

        assert result is not None
        assert result.strategy_name == "fast"


# =============================================================================
# Test: Convenience Function
# =============================================================================


class TestParallelConvenience:
    """Tests for parallel execution convenience function."""

    @pytest.mark.asyncio
    async def test_execute_strategies_parallel(
        self,
        registry: StrategyRegistry,
    ):
        """Test parallel execution convenience function."""
        result = await execute_strategies_parallel(
            strategies=["fast", "configurable"],
            query="test",
        )

        assert result.successful_count == 2
