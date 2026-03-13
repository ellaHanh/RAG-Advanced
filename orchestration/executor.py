"""
RAG-Advanced Single Strategy Executor.

Execute individual RAG strategies with configuration, metrics tracking,
and cost calculation.

Usage:
    from orchestration.executor import StrategyExecutor

    executor = StrategyExecutor()
    result = await executor.execute(
        strategy_name="standard",
        query="What is machine learning?",
        config=StrategyConfig(limit=5),
    )

    print(f"Found {result.document_count} documents")
    print(f"Cost: ${result.cost_usd:.4f}")
    print(f"Latency: {result.latency_ms}ms")
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Callable, Coroutine

from pydantic import BaseModel, ConfigDict, Field

from orchestration.cost_tracker import CostTracker
from orchestration.errors import (
    StrategyExecutionError,
    StrategyNotFoundError,
    StrategyTimeoutError,
)
from orchestration.models import (
    Document,
    ExecutionResult,
    StrategyConfig,
    TokenCounts,
)
from orchestration.registry import StrategyRegistry, get_registry


logger = logging.getLogger(__name__)


# =============================================================================
# Type Aliases
# =============================================================================

StrategyFunction = Callable[..., Coroutine[Any, Any, list[Document]]]


# =============================================================================
# Configuration
# =============================================================================


class ExecutorConfig(BaseModel):
    """
    Configuration for strategy executor.

    Attributes:
        default_timeout_seconds: Default timeout for strategy execution.
        track_costs: Whether to track API costs.
        enable_caching: Whether to enable result caching.
        max_retries: Maximum retry attempts on failure.
        retry_delay_seconds: Delay between retries.
    """

    model_config = ConfigDict(frozen=True)

    default_timeout_seconds: float = Field(default=60.0, ge=1.0, description="Default timeout")
    track_costs: bool = Field(default=True, description="Track API costs")
    enable_caching: bool = Field(default=True, description="Enable result caching")
    max_retries: int = Field(default=0, ge=0, le=3, description="Max retries")
    retry_delay_seconds: float = Field(default=1.0, ge=0.0, description="Retry delay")


# =============================================================================
# Execution Context
# =============================================================================


@dataclass
class ExecutionContext:
    """
    Context passed to strategy functions during execution.

    Attributes:
        query: The search query.
        config: Strategy configuration.
        cost_tracker: Cost tracker instance.
        metadata: Additional execution metadata.
        input_documents: When set (e.g. in a chain), strategies that support it use
            these as input instead of retrieving (e.g. reranking rerank-only mode).
        original_query: When set (e.g. in a chain), used for scoring (e.g. rerank
            with original user query).
    """

    query: str
    config: StrategyConfig
    cost_tracker: CostTracker
    metadata: dict[str, Any] = field(default_factory=dict)
    input_documents: list[Document] | None = None
    original_query: str | None = None

    def add_embedding_cost(
        self,
        model: str,
        token_count: int,
    ) -> float:
        """Track embedding cost."""
        return self.cost_tracker.add_embedding_cost(model, token_count)

    def add_llm_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """Track LLM cost."""
        return self.cost_tracker.add_llm_cost(model, input_tokens, output_tokens)


# =============================================================================
# Strategy Executor
# =============================================================================


class StrategyExecutor:
    """
    Execute individual RAG strategies with metrics tracking.

    Handles strategy lookup, execution timing, cost tracking,
    and result packaging.

    Example:
        >>> executor = StrategyExecutor()
        >>> result = await executor.execute("standard", "What is AI?")
        >>> print(f"Cost: ${result.cost_usd}")
    """

    def __init__(
        self,
        registry: StrategyRegistry | None = None,
        config: ExecutorConfig | None = None,
    ) -> None:
        """
        Initialize the executor.

        Args:
            registry: Optional strategy registry (uses global if None).
            config: Optional executor configuration.
        """
        self._registry = registry
        self._config = config or ExecutorConfig()

    @property
    def registry(self) -> StrategyRegistry:
        """Get the strategy registry."""
        if self._registry is None:
            self._registry = get_registry()
        return self._registry

    async def execute(
        self,
        strategy_name: str,
        query: str,
        config: StrategyConfig | None = None,
        timeout: float | None = None,
        metadata: dict[str, Any] | None = None,
        input_documents: list[Document] | None = None,
        original_query: str | None = None,
    ) -> ExecutionResult:
        """
        Execute a single strategy.

        Args:
            strategy_name: Name of the strategy to execute.
            query: Search query.
            config: Optional strategy configuration.
            timeout: Optional timeout override.
            metadata: Optional execution metadata.
            input_documents: Optional documents from previous chain step (for refiners).
            original_query: Optional original user query (e.g. for reranking in chain).

        Returns:
            ExecutionResult with documents and metrics.

        Raises:
            StrategyNotFoundError: If strategy is not registered.
            StrategyTimeoutError: If execution times out.
            StrategyExecutionError: If execution fails.
        """
        # Get strategy from registry
        strategy_func = self.registry.get(strategy_name)
        if strategy_func is None:
            raise StrategyNotFoundError(strategy_name)

        # Initialize
        config = config or StrategyConfig()
        timeout = timeout or self._config.default_timeout_seconds
        cost_tracker = CostTracker()

        # Create execution context
        context = ExecutionContext(
            query=query,
            config=config,
            cost_tracker=cost_tracker,
            metadata=metadata or {},
            input_documents=input_documents,
            original_query=original_query,
        )

        # Execute with timing and error handling
        start_time = time.time()

        try:
            documents = await self._execute_with_timeout(
                strategy_func,
                context,
                timeout,
            )
            success = True
            error = None

        except (asyncio.TimeoutError, TimeoutError):
            raise StrategyTimeoutError(
                strategy_name,
                timeout,
            )

        except Exception as e:
            logger.exception(f"Strategy execution failed: {strategy_name}")
            raise StrategyExecutionError(
                strategy_name,
                str(e),
                {"error_type": type(e).__name__},
            ) from e

        finally:
            end_time = time.time()
            latency_ms = int((end_time - start_time) * 1000)

        # Build result
        cost_summary = cost_tracker.get_summary()

        return ExecutionResult(
            documents=documents,
            query=query,
            strategy_name=strategy_name,
            latency_ms=latency_ms,
            cost_usd=cost_summary.total_cost,
            token_counts=TokenCounts(
                embedding_tokens=cost_summary.total_input_tokens,
                llm_input_tokens=cost_summary.by_category.get("llm_input", 0),
                llm_output_tokens=cost_summary.total_output_tokens,
            ),
            cached=False,
            metadata={
                "config": config.model_dump() if config else {},
                **(metadata or {}),
            },
        )

    async def execute_with_retry(
        self,
        strategy_name: str,
        query: str,
        config: StrategyConfig | None = None,
        max_retries: int | None = None,
    ) -> ExecutionResult:
        """
        Execute strategy with automatic retry on failure.

        Args:
            strategy_name: Strategy name.
            query: Search query.
            config: Strategy configuration.
            max_retries: Override max retry count.

        Returns:
            ExecutionResult on success.

        Raises:
            StrategyExecutionError: If all retries fail.
        """
        max_retries = max_retries if max_retries is not None else self._config.max_retries
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                return await self.execute(strategy_name, query, config)

            except StrategyExecutionError as e:
                last_error = e
                if attempt < max_retries:
                    logger.warning(
                        f"Retry {attempt + 1}/{max_retries} for {strategy_name}: {e}"
                    )
                    await asyncio.sleep(self._config.retry_delay_seconds)

        raise StrategyExecutionError(
            strategy_name,
            f"All {max_retries + 1} attempts failed",
            {"last_error": str(last_error)},
        )

    def execute_sync(
        self,
        strategy_name: str,
        query: str,
        config: StrategyConfig | None = None,
    ) -> ExecutionResult:
        """
        Synchronous execution wrapper.

        Args:
            strategy_name: Strategy name.
            query: Search query.
            config: Strategy configuration.

        Returns:
            ExecutionResult.
        """
        return asyncio.run(self.execute(strategy_name, query, config))

    async def _execute_with_timeout(
        self,
        strategy_func: StrategyFunction,
        context: ExecutionContext,
        timeout: float,
    ) -> list[Document]:
        """Execute strategy function with timeout."""
        return await asyncio.wait_for(
            strategy_func(context),
            timeout=timeout,
        )


# =============================================================================
# Convenience Functions
# =============================================================================


async def execute_strategy(
    strategy_name: str,
    query: str,
    config: StrategyConfig | None = None,
    timeout: float | None = None,
) -> ExecutionResult:
    """
    Execute a strategy (convenience function).

    Args:
        strategy_name: Strategy name.
        query: Search query.
        config: Strategy configuration.
        timeout: Timeout in seconds.

    Returns:
        ExecutionResult.
    """
    executor = StrategyExecutor()
    return await executor.execute(strategy_name, query, config, timeout)


def execute_strategy_sync(
    strategy_name: str,
    query: str,
    config: StrategyConfig | None = None,
) -> ExecutionResult:
    """
    Execute a strategy synchronously.

    Args:
        strategy_name: Strategy name.
        query: Search query.
        config: Strategy configuration.

    Returns:
        ExecutionResult.
    """
    executor = StrategyExecutor()
    return executor.execute_sync(strategy_name, query, config)


# =============================================================================
# Parallel Execution
# =============================================================================


@dataclass
class ParallelExecutionResult:
    """
    Result of parallel strategy execution.

    Attributes:
        results: Dictionary mapping strategy names to their results.
        errors: Dictionary mapping strategy names to their errors.
        total_latency_ms: Total wall-clock time for all strategies.
        total_cost_usd: Total cost across all strategies.
    """

    results: dict[str, ExecutionResult] = field(default_factory=dict)
    errors: dict[str, Exception] = field(default_factory=dict)
    total_latency_ms: int = 0
    total_cost_usd: float = 0.0

    @property
    def successful_count(self) -> int:
        """Get count of successful executions."""
        return len(self.results)

    @property
    def failed_count(self) -> int:
        """Get count of failed executions."""
        return len(self.errors)

    @property
    def all_succeeded(self) -> bool:
        """Check if all strategies succeeded."""
        return len(self.errors) == 0

    def get_best_result(self) -> ExecutionResult | None:
        """Get result with highest document relevance."""
        if not self.results:
            return None

        best = None
        best_score = -1.0

        for result in self.results.values():
            if result.documents:
                score = result.documents[0].similarity or 0.0
                if score > best_score:
                    best_score = score
                    best = result

        return best


class ParallelExecutor:
    """
    Execute multiple strategies concurrently using asyncio.gather.

    Supports error isolation, timeout handling, and result aggregation.

    Example:
        >>> executor = ParallelExecutor()
        >>> result = await executor.execute_all(
        ...     strategies=["standard", "reranking"],
        ...     query="What is AI?",
        ... )
        >>> print(f"Successful: {result.successful_count}")
    """

    def __init__(
        self,
        executor: StrategyExecutor | None = None,
        max_concurrency: int | None = None,
    ) -> None:
        """
        Initialize parallel executor.

        Args:
            executor: Strategy executor instance.
            max_concurrency: Maximum concurrent executions (None = unlimited).
        """
        self._executor = executor or StrategyExecutor()
        self._max_concurrency = max_concurrency
        self._semaphore: asyncio.Semaphore | None = None

    async def execute_all(
        self,
        strategies: list[str],
        query: str,
        config: StrategyConfig | None = None,
        timeout: float | None = None,
        return_exceptions: bool = True,
    ) -> ParallelExecutionResult:
        """
        Execute multiple strategies concurrently.

        Args:
            strategies: List of strategy names to execute.
            query: Search query.
            config: Optional strategy configuration (same for all).
            timeout: Optional timeout per strategy.
            return_exceptions: If True, capture exceptions instead of raising.

        Returns:
            ParallelExecutionResult with all results and errors.
        """
        start_time = time.time()

        # Initialize semaphore for concurrency control
        if self._max_concurrency:
            self._semaphore = asyncio.Semaphore(self._max_concurrency)

        # Create tasks for all strategies
        tasks = [
            self._execute_with_semaphore(strategy, query, config, timeout)
            for strategy in strategies
        ]

        # Execute all concurrently
        task_results = await asyncio.gather(*tasks, return_exceptions=return_exceptions)

        # Aggregate results
        result = ParallelExecutionResult()

        for strategy, task_result in zip(strategies, task_results):
            if isinstance(task_result, Exception):
                result.errors[strategy] = task_result
            else:
                result.results[strategy] = task_result
                result.total_cost_usd += task_result.cost_usd

        result.total_latency_ms = int((time.time() - start_time) * 1000)

        return result

    async def execute_first_success(
        self,
        strategies: list[str],
        query: str,
        config: StrategyConfig | None = None,
        timeout: float | None = None,
    ) -> ExecutionResult | None:
        """
        Execute strategies and return first successful result.

        Cancels remaining strategies once one succeeds.

        Args:
            strategies: List of strategy names.
            query: Search query.
            config: Strategy configuration.
            timeout: Timeout per strategy.

        Returns:
            First successful ExecutionResult or None.
        """
        tasks = []

        for strategy in strategies:
            task = asyncio.create_task(
                self._executor.execute(strategy, query, config, timeout)
            )
            tasks.append((strategy, task))

        # Wait for first completion
        pending = {task for _, task in tasks}

        while pending:
            done, pending = await asyncio.wait(
                pending,
                return_when=asyncio.FIRST_COMPLETED,
            )

            for task in done:
                try:
                    result = task.result()
                    # Cancel remaining tasks
                    for p in pending:
                        p.cancel()
                    return result
                except Exception:
                    # Continue waiting for others
                    pass

        return None

    async def _execute_with_semaphore(
        self,
        strategy: str,
        query: str,
        config: StrategyConfig | None,
        timeout: float | None,
    ) -> ExecutionResult:
        """Execute with optional semaphore for concurrency control."""
        if self._semaphore:
            async with self._semaphore:
                return await self._executor.execute(strategy, query, config, timeout)
        return await self._executor.execute(strategy, query, config, timeout)


async def execute_strategies_parallel(
    strategies: list[str],
    query: str,
    config: StrategyConfig | None = None,
    timeout: float | None = None,
) -> ParallelExecutionResult:
    """
    Execute multiple strategies in parallel (convenience function).

    Args:
        strategies: List of strategy names.
        query: Search query.
        config: Strategy configuration.
        timeout: Timeout per strategy.

    Returns:
        ParallelExecutionResult.
    """
    executor = ParallelExecutor()
    return await executor.execute_all(strategies, query, config, timeout)
