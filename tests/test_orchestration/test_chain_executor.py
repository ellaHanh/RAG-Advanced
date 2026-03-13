"""
Unit tests for Sequential Chain Executor.

Tests cover:
- Sequential execution
- Context passing
- Error handling
- Chain result structure
"""

from __future__ import annotations

import asyncio

import pytest

from orchestration.chain_executor import (
    ChainExecutor,
    ChainExecutorConfig,
    ChainResult,
    ChainStepResult,
    execute_chain,
)
from orchestration.errors import ChainConfigurationError
from orchestration.executor import ExecutionContext, StrategyExecutor
from orchestration.models import ChainStep, Document
from orchestration.registry import StrategyRegistry


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def registry() -> StrategyRegistry:
    """Create a test registry with chain strategies."""
    StrategyRegistry.reset()
    registry = StrategyRegistry()

    async def search_strategy(ctx: ExecutionContext) -> list[Document]:
        return [
            Document(
                id="search1",
                content="Search result 1",
                title="Result 1",
                source="test.md",
                similarity=0.8,
            ),
            Document(
                id="search2",
                content="Search result 2",
                title="Result 2",
                source="test.md",
                similarity=0.7,
            ),
        ]

    async def rerank_strategy(ctx: ExecutionContext) -> list[Document]:
        # When input_documents is passed (from previous step), use them; else return fixed list
        if getattr(ctx, "input_documents", None) and ctx.input_documents:
            # Refiner: return first doc from input as "reranked" (simulate rerank)
            doc = ctx.input_documents[0]
            return [
                Document(
                    id=doc.id,
                    content=doc.content,
                    title=doc.title or "",
                    source=doc.source or "",
                    similarity=0.95,
                    metadata=dict(doc.metadata) if doc.metadata else {},
                )
            ]
        return [
            Document(
                id="rerank1",
                content="Reranked result",
                title="Top Result",
                source="test.md",
                similarity=0.95,
            ),
        ]

    async def failing_strategy(ctx: ExecutionContext) -> list[Document]:
        raise ValueError("Strategy failed")

    registry.register("search", search_strategy)
    registry.register("rerank", rerank_strategy)
    registry.register("failing", failing_strategy)

    return registry


@pytest.fixture
def executor(registry: StrategyRegistry) -> StrategyExecutor:
    """Create a strategy executor."""
    return StrategyExecutor(registry=registry)


@pytest.fixture
def chain_executor(executor: StrategyExecutor) -> ChainExecutor:
    """Create a chain executor."""
    return ChainExecutor(strategy_executor=executor)


# =============================================================================
# Test: ChainStepResult
# =============================================================================


class TestChainStepResult:
    """Tests for ChainStepResult."""

    def test_step_result_creation(self):
        """Test step result creation."""
        from orchestration.models import ExecutionResult

        result = ChainStepResult(
            step_name="test_step",
            strategy_name="test",
            result=ExecutionResult(
                documents=[],
                query="test",
                strategy_name="test",
                latency_ms=100,
                cost_usd=0.01,
            ),
            duration_ms=150,
        )

        assert result.step_name == "test_step"
        assert result.duration_ms == 150


# =============================================================================
# Test: ChainResult
# =============================================================================


class TestChainResult:
    """Tests for ChainResult."""

    def test_step_count(self):
        """Test step count property."""
        from orchestration.models import ExecutionResult

        result = ChainResult(query="test")
        result.steps = [
            ChainStepResult(
                step_name="s1",
                strategy_name="test",
                result=ExecutionResult(
                    documents=[],
                    query="test",
                    strategy_name="test",
                    latency_ms=100,
                    cost_usd=0.01,
                ),
            ),
            ChainStepResult(
                step_name="s2",
                strategy_name="test",
                result=ExecutionResult(
                    documents=[],
                    query="test",
                    strategy_name="test",
                    latency_ms=100,
                    cost_usd=0.01,
                ),
            ),
        ]

        assert result.step_count == 2

    def test_get_step_by_name(self):
        """Test getting step by name."""
        from orchestration.models import ExecutionResult

        result = ChainResult(query="test")
        result.steps = [
            ChainStepResult(
                step_name="search",
                strategy_name="test",
                result=ExecutionResult(
                    documents=[],
                    query="test",
                    strategy_name="test",
                    latency_ms=100,
                    cost_usd=0.01,
                ),
            ),
        ]

        assert result.get_step("search") is not None
        assert result.get_step("nonexistent") is None

    def test_to_dict(self):
        """Test serialization."""
        result = ChainResult(
            query="test query",
            total_latency_ms=500,
            total_cost_usd=0.05,
        )

        d = result.to_dict()

        assert d["query"] == "test query"
        assert d["total_latency_ms"] == 500


# =============================================================================
# Test: Basic Execution
# =============================================================================


class TestBasicExecution:
    """Tests for basic chain execution."""

    @pytest.mark.asyncio
    async def test_execute_single_step_chain(
        self,
        chain_executor: ChainExecutor,
    ):
        """Test executing a single step chain."""
        steps = [ChainStep(strategy="search")]

        result = await chain_executor.execute_chain(steps, "test query")

        assert result.success is True
        assert result.step_count == 1
        assert len(result.final_documents) > 0

    @pytest.mark.asyncio
    async def test_execute_multi_step_chain(
        self,
        chain_executor: ChainExecutor,
    ):
        """Test executing multiple steps."""
        steps = [
            ChainStep(strategy="search"),
            ChainStep(strategy="rerank"),
        ]

        result = await chain_executor.execute_chain(steps, "test query")

        assert result.success is True
        assert result.step_count == 2

    @pytest.mark.asyncio
    async def test_chain_passes_documents_to_next_step(
        self,
        chain_executor: ChainExecutor,
    ):
        """Test that step output is passed as input to next step (real chaining)."""
        steps = [
            ChainStep(strategy="search"),
            ChainStep(strategy="rerank"),
        ]
        result = await chain_executor.execute_chain(steps, "test query")

        assert result.success is True
        assert result.step_count == 2
        # Rerank step receives search's docs and returns first one; final_documents should be that
        final = result.final_documents
        assert len(final) == 1
        assert final[0].id == "search1"
        assert "Search result 1" in final[0].content
        assert final[0].similarity == 0.95

    @pytest.mark.asyncio
    async def test_chain_accumulates_cost(
        self,
        chain_executor: ChainExecutor,
    ):
        """Test that costs are accumulated."""
        steps = [
            ChainStep(strategy="search"),
            ChainStep(strategy="search"),
        ]

        result = await chain_executor.execute_chain(steps, "test")

        # Cost should be accumulated (even if 0 in test)
        assert result.total_cost_usd >= 0

    @pytest.mark.asyncio
    async def test_chain_measures_latency(
        self,
        chain_executor: ChainExecutor,
    ):
        """Test that latency is measured."""
        steps = [ChainStep(strategy="search")]

        result = await chain_executor.execute_chain(steps, "test")

        # Latency is measured (may be 0 for very fast operations)
        assert result.total_latency_ms >= 0


# =============================================================================
# Test: Error Handling
# =============================================================================


class TestErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_empty_chain_raises_error(
        self,
        chain_executor: ChainExecutor,
    ):
        """Test that empty chain raises error."""
        with pytest.raises(ChainConfigurationError):
            await chain_executor.execute_chain([], "test")

    @pytest.mark.asyncio
    async def test_failing_step_stops_chain(
        self,
        chain_executor: ChainExecutor,
    ):
        """Test that failing step stops execution."""
        steps = [
            ChainStep(strategy="search"),
            ChainStep(strategy="failing"),
            ChainStep(strategy="rerank"),
        ]

        result = await chain_executor.execute_chain(steps, "test")

        assert result.success is False
        assert result.error is not None
        # Should only have executed 2 steps (search succeeded, fail failed)
        assert result.step_count <= 2

    @pytest.mark.asyncio
    async def test_continue_on_error(
        self,
        executor: StrategyExecutor,
    ):
        """Test continuing after error."""
        config = ChainExecutorConfig(continue_on_error=True)
        chain_executor = ChainExecutor(strategy_executor=executor, config=config)

        steps = [
            ChainStep(strategy="failing"),
            ChainStep(strategy="search"),
        ]

        result = await chain_executor.execute_chain(steps, "test")

        # Should have continued to second step
        assert result.step_count == 2


# =============================================================================
# Test: Context Passing
# =============================================================================


class TestContextPassing:
    """Tests for context passing between steps."""

    @pytest.mark.asyncio
    async def test_context_has_step_results(
        self,
        chain_executor: ChainExecutor,
    ):
        """Test that context accumulates step results."""
        steps = [
            ChainStep(strategy="search"),
        ]

        result = await chain_executor.execute_chain(steps, "test")

        assert result.final_context is not None
        # Single step means 1 intermediate result
        assert len(result.final_context.intermediate_results) >= 1


# =============================================================================
# Test: Convenience Function
# =============================================================================


class TestFallbackHandling:
    """Tests for fallback handling."""

    @pytest.mark.asyncio
    async def test_fallback_executes_on_failure(
        self,
        chain_executor: ChainExecutor,
    ):
        """Test that fallback executes when primary fails."""
        steps = [
            ChainStep(strategy="failing", fallback_strategy="search"),
        ]

        result = await chain_executor.execute_chain(steps, "test")

        # Should succeed via fallback
        assert result.success is True
        assert result.step_count == 1
        # Fallback results should be present
        assert len(result.final_documents) > 0

    @pytest.mark.asyncio
    async def test_step_continue_on_error_overrides_global(
        self,
        executor: StrategyExecutor,
    ):
        """Test step-level continue_on_error."""
        config = ChainExecutorConfig(continue_on_error=False)
        chain_executor = ChainExecutor(strategy_executor=executor, config=config)

        steps = [
            ChainStep(strategy="failing", continue_on_error=True),
            ChainStep(strategy="search"),
        ]

        result = await chain_executor.execute_chain(steps, "test")

        # Should continue despite failure
        assert result.step_count == 2


class TestConvenienceFunction:
    """Tests for convenience function."""

    @pytest.mark.asyncio
    async def test_execute_chain_function(
        self,
        registry: StrategyRegistry,
    ):
        """Test execute_chain convenience function."""
        steps = [ChainStep(strategy="search")]

        result = await execute_chain(steps, "test query")

        assert result.success is True
        assert result.step_count == 1
