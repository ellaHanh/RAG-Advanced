"""
RAG-Advanced Benchmark Runner.

Execute benchmarks across multiple strategies with statistical analysis.
Provides p50/p95/p99 latency metrics, cost tracking, and strategy rankings.

Usage:
    from evaluation.benchmarks import BenchmarkRunner, BenchmarkConfig

    runner = BenchmarkRunner()
    config = BenchmarkConfig(
        strategies=["standard", "reranking", "multi_query"],
        iterations=3,
    )
    report = await runner.run(queries, config)

    print(report.summary())
    print(report.rankings)
"""

from __future__ import annotations

import asyncio
import logging
import statistics
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Callable, Coroutine
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from evaluation.metrics import EvaluationMetrics, calculate_metrics


logger = logging.getLogger(__name__)


# =============================================================================
# Type Aliases
# =============================================================================

# Strategy execution function type
StrategyExecutor = Callable[[str, dict[str, Any]], Coroutine[Any, Any, "StrategyResult"]]


# =============================================================================
# Configuration Models
# =============================================================================


class BenchmarkConfig(BaseModel):
    """
    Configuration for benchmark execution.

    Attributes:
        strategies: List of strategy names to benchmark.
        iterations: Number of iterations per query for stability.
        k_values: K values for IR metrics calculation.
        timeout_seconds: Timeout per strategy execution.
        max_concurrent: Maximum concurrent executions.
        warmup_iterations: Iterations to discard for warmup.
    """

    model_config = ConfigDict(frozen=True)

    strategies: list[str] = Field(..., min_length=1, description="Strategies to benchmark")
    iterations: int = Field(default=3, ge=1, le=10, description="Iterations per query")
    k_values: list[int] = Field(default=[3, 5, 10], description="K values for metrics")
    timeout_seconds: float = Field(default=60.0, ge=1.0, description="Timeout per execution")
    max_concurrent: int = Field(default=5, ge=1, le=20, description="Max concurrent executions")
    warmup_iterations: int = Field(default=0, ge=0, le=2, description="Warmup iterations")


class BenchmarkQuery(BaseModel):
    """
    A single query in the benchmark dataset.

    Attributes:
        query_id: Unique identifier for the query.
        query: The query text.
        ground_truth_ids: List of relevant document IDs.
        relevance_scores: Optional graded relevance scores.
    """

    model_config = ConfigDict(frozen=True)

    query_id: str = Field(..., description="Query identifier")
    query: str = Field(..., description="Query text")
    ground_truth_ids: list[str] = Field(default_factory=list, description="Relevant doc IDs")
    relevance_scores: dict[str, int] | None = Field(default=None, description="Graded relevance")


# =============================================================================
# Result Models
# =============================================================================


@dataclass
class StrategyResult:
    """
    Result from a single strategy execution.

    Attributes:
        strategy_name: Name of the executed strategy.
        query_id: Query that was executed.
        retrieved_ids: List of retrieved document IDs.
        latency_ms: Execution time in milliseconds.
        cost_usd: Estimated cost in USD.
        success: Whether execution succeeded.
        error: Error message if failed.
    """

    strategy_name: str
    query_id: str
    retrieved_ids: list[str] = field(default_factory=list)
    latency_ms: int = 0
    cost_usd: float = 0.0
    success: bool = True
    error: str | None = None


@dataclass
class StrategyStatistics:
    """
    Statistical summary for a strategy.

    Attributes:
        strategy_name: Name of the strategy.
        query_count: Number of queries executed.
        iteration_count: Number of iterations.
        latency_p50: 50th percentile latency in ms.
        latency_p95: 95th percentile latency in ms.
        latency_p99: 99th percentile latency in ms.
        latency_mean: Mean latency in ms.
        latency_std: Standard deviation of latency.
        total_cost: Total cost in USD.
        avg_cost_per_query: Average cost per query.
        success_rate: Percentage of successful executions.
        avg_precision: Average precision at each k.
        avg_recall: Average recall at each k.
        avg_mrr: Average MRR.
        avg_ndcg: Average NDCG at each k.
    """

    strategy_name: str
    query_count: int = 0
    iteration_count: int = 0
    latency_p50: float = 0.0
    latency_p95: float = 0.0
    latency_p99: float = 0.0
    latency_mean: float = 0.0
    latency_std: float = 0.0
    total_cost: float = 0.0
    avg_cost_per_query: float = 0.0
    success_rate: float = 0.0
    avg_precision: dict[int, float] = field(default_factory=dict)
    avg_recall: dict[int, float] = field(default_factory=dict)
    avg_mrr: float = 0.0
    avg_ndcg: dict[int, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "strategy_name": self.strategy_name,
            "query_count": self.query_count,
            "iteration_count": self.iteration_count,
            "latency_p50": self.latency_p50,
            "latency_p95": self.latency_p95,
            "latency_p99": self.latency_p99,
            "latency_mean": self.latency_mean,
            "latency_std": self.latency_std,
            "total_cost": self.total_cost,
            "avg_cost_per_query": self.avg_cost_per_query,
            "success_rate": self.success_rate,
            "avg_precision": self.avg_precision,
            "avg_recall": self.avg_recall,
            "avg_mrr": self.avg_mrr,
            "avg_ndcg": self.avg_ndcg,
        }


class BenchmarkReport(BaseModel):
    """
    Complete benchmark report.

    Attributes:
        benchmark_id: Unique identifier for this benchmark run.
        config: Configuration used for the benchmark.
        statistics: Per-strategy statistics.
        rankings: Strategy rankings by various metrics.
        started_at: When the benchmark started.
        completed_at: When the benchmark completed.
        duration_seconds: Total benchmark duration.
        total_queries: Total number of queries.
        total_executions: Total number of strategy executions.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    benchmark_id: str = Field(default_factory=lambda: str(uuid4()))
    config: BenchmarkConfig
    statistics: dict[str, StrategyStatistics] = Field(default_factory=dict)
    rankings: dict[str, list[str]] = Field(default_factory=dict)
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None
    duration_seconds: float = 0.0
    total_queries: int = 0
    total_executions: int = 0

    def summary(self) -> str:
        """Generate a text summary of the benchmark."""
        lines = [
            f"Benchmark Report: {self.benchmark_id}",
            f"Duration: {self.duration_seconds:.2f}s",
            f"Queries: {self.total_queries}, Executions: {self.total_executions}",
            "",
            "Strategy Performance:",
            "-" * 60,
        ]

        for name, stats in self.statistics.items():
            lines.append(
                f"{name}: "
                f"P50={stats.latency_p50:.0f}ms, "
                f"P95={stats.latency_p95:.0f}ms, "
                f"MRR={stats.avg_mrr:.3f}, "
                f"Cost=${stats.total_cost:.4f}"
            )

        lines.extend(["", "Rankings:", "-" * 60])
        for metric, ranking in self.rankings.items():
            lines.append(f"{metric}: {' > '.join(ranking)}")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "benchmark_id": self.benchmark_id,
            "config": self.config.model_dump(),
            "statistics": {k: v.to_dict() for k, v in self.statistics.items()},
            "rankings": self.rankings,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
            "total_queries": self.total_queries,
            "total_executions": self.total_executions,
        }


# =============================================================================
# Statistical Analysis
# =============================================================================


def _calculate_percentile(values: list[float], percentile: float) -> float:
    """
    Calculate percentile of a list of values.

    Args:
        values: List of numeric values.
        percentile: Percentile to calculate (0-100).

    Returns:
        The percentile value.
    """
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = (percentile / 100) * (len(sorted_values) - 1)
    lower = int(index)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = index - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def _calculate_statistics(
    results: list[StrategyResult],
    metrics_list: list[EvaluationMetrics],
    k_values: list[int],
) -> StrategyStatistics:
    """
    Calculate statistics from a list of results.

    Args:
        results: List of strategy results.
        metrics_list: List of evaluation metrics.
        k_values: K values used for metrics.

    Returns:
        StrategyStatistics with aggregated data.
    """
    if not results:
        return StrategyStatistics(strategy_name="unknown")

    strategy_name = results[0].strategy_name

    # Latency statistics
    latencies = [r.latency_ms for r in results if r.success]
    if latencies:
        latency_p50 = _calculate_percentile(latencies, 50)
        latency_p95 = _calculate_percentile(latencies, 95)
        latency_p99 = _calculate_percentile(latencies, 99)
        latency_mean = statistics.mean(latencies)
        latency_std = statistics.stdev(latencies) if len(latencies) > 1 else 0.0
    else:
        latency_p50 = latency_p95 = latency_p99 = latency_mean = latency_std = 0.0

    # Cost statistics
    total_cost = sum(r.cost_usd for r in results)
    successful_count = sum(1 for r in results if r.success)
    avg_cost_per_query = total_cost / successful_count if successful_count > 0 else 0.0

    # Success rate
    success_rate = successful_count / len(results) if results else 0.0

    # Metrics averages
    avg_precision: dict[int, float] = {}
    avg_recall: dict[int, float] = {}
    avg_ndcg: dict[int, float] = {}

    if metrics_list:
        for k in k_values:
            precisions = [m.precision.get(k, 0.0) for m in metrics_list]
            recalls = [m.recall.get(k, 0.0) for m in metrics_list]
            ndcgs = [m.ndcg.get(k, 0.0) for m in metrics_list]

            avg_precision[k] = statistics.mean(precisions) if precisions else 0.0
            avg_recall[k] = statistics.mean(recalls) if recalls else 0.0
            avg_ndcg[k] = statistics.mean(ndcgs) if ndcgs else 0.0

        mrrs = [m.mrr for m in metrics_list]
        avg_mrr = statistics.mean(mrrs) if mrrs else 0.0
    else:
        avg_mrr = 0.0

    return StrategyStatistics(
        strategy_name=strategy_name,
        query_count=len(set(r.query_id for r in results)),
        iteration_count=len(results),
        latency_p50=latency_p50,
        latency_p95=latency_p95,
        latency_p99=latency_p99,
        latency_mean=latency_mean,
        latency_std=latency_std,
        total_cost=total_cost,
        avg_cost_per_query=avg_cost_per_query,
        success_rate=success_rate,
        avg_precision=avg_precision,
        avg_recall=avg_recall,
        avg_mrr=avg_mrr,
        avg_ndcg=avg_ndcg,
    )


def _calculate_rankings(
    statistics: dict[str, StrategyStatistics],
    k_values: list[int],
) -> dict[str, list[str]]:
    """
    Calculate strategy rankings by various metrics.

    Args:
        statistics: Per-strategy statistics.
        k_values: K values for metrics.

    Returns:
        Dictionary of metric -> ranked strategy list.
    """
    rankings: dict[str, list[str]] = {}

    if not statistics:
        return rankings

    strategies = list(statistics.keys())

    # Rank by latency (lower is better)
    rankings["latency_p50"] = sorted(
        strategies,
        key=lambda s: statistics[s].latency_p50,
    )

    rankings["latency_p95"] = sorted(
        strategies,
        key=lambda s: statistics[s].latency_p95,
    )

    # Rank by cost (lower is better)
    rankings["cost"] = sorted(
        strategies,
        key=lambda s: statistics[s].avg_cost_per_query,
    )

    # Rank by MRR (higher is better)
    rankings["mrr"] = sorted(
        strategies,
        key=lambda s: statistics[s].avg_mrr,
        reverse=True,
    )

    # Rank by precision and NDCG at different k values
    for k in k_values:
        rankings[f"precision@{k}"] = sorted(
            strategies,
            key=lambda s, k=k: statistics[s].avg_precision.get(k, 0.0),
            reverse=True,
        )
        rankings[f"ndcg@{k}"] = sorted(
            strategies,
            key=lambda s, k=k: statistics[s].avg_ndcg.get(k, 0.0),
            reverse=True,
        )

    return rankings


# =============================================================================
# Benchmark Runner
# =============================================================================


class BenchmarkRunner:
    """
    Execute benchmarks across strategies with statistical analysis.

    The runner executes multiple queries against multiple strategies,
    collects results, calculates metrics, and generates rankings.

    Example:
        >>> runner = BenchmarkRunner()
        >>> config = BenchmarkConfig(strategies=["standard", "reranking"])
        >>> report = await runner.run(queries, config)
    """

    def __init__(
        self,
        executor: StrategyExecutor | None = None,
    ) -> None:
        """
        Initialize the benchmark runner.

        Args:
            executor: Optional custom strategy executor function.
                     If not provided, uses a mock executor for testing.
        """
        self._executor = executor or self._default_executor
        self._semaphore: asyncio.Semaphore | None = None

    async def _default_executor(
        self,
        strategy_name: str,
        query_data: dict[str, Any],
    ) -> StrategyResult:
        """
        Default mock executor for testing.

        In production, this should be replaced with actual strategy execution.
        """
        # Simulate execution time
        await asyncio.sleep(0.01)

        return StrategyResult(
            strategy_name=strategy_name,
            query_id=query_data.get("query_id", "unknown"),
            retrieved_ids=query_data.get("mock_retrieved", []),
            latency_ms=int(50 + hash(strategy_name) % 100),
            cost_usd=0.001 * (1 + hash(strategy_name) % 3),
            success=True,
        )

    async def run(
        self,
        queries: list[BenchmarkQuery] | list[dict[str, Any]],
        config: BenchmarkConfig,
    ) -> BenchmarkReport:
        """
        Run the benchmark.

        Args:
            queries: List of benchmark queries.
            config: Benchmark configuration.

        Returns:
            BenchmarkReport with all results and statistics.
        """
        start_time = time.time()
        started_at = datetime.now(UTC)

        # Convert dict queries to BenchmarkQuery objects
        benchmark_queries = []
        for q in queries:
            if isinstance(q, dict):
                benchmark_queries.append(BenchmarkQuery(**q))
            else:
                benchmark_queries.append(q)

        # Initialize semaphore for concurrency control
        self._semaphore = asyncio.Semaphore(config.max_concurrent)

        # Collect all results
        all_results: dict[str, list[StrategyResult]] = {s: [] for s in config.strategies}
        all_metrics: dict[str, list[EvaluationMetrics]] = {s: [] for s in config.strategies}

        total_executions = 0

        # Run benchmarks for each strategy
        for strategy in config.strategies:
            logger.info(f"Benchmarking strategy: {strategy}")

            for iteration in range(config.iterations):
                # Skip warmup iterations in results
                is_warmup = iteration < config.warmup_iterations

                tasks = []
                for query in benchmark_queries:
                    task = self._execute_with_timeout(
                        strategy,
                        query,
                        config.timeout_seconds,
                    )
                    tasks.append((query, task))

                # Execute all queries for this iteration
                for query, task in tasks:
                    try:
                        result = await task
                        total_executions += 1

                        if not is_warmup:
                            all_results[strategy].append(result)

                            # Calculate metrics if we have ground truth
                            if result.success and query.ground_truth_ids:
                                metrics = calculate_metrics(
                                    retrieved_ids=result.retrieved_ids,
                                    ground_truth_ids=query.ground_truth_ids,
                                    relevance_scores=query.relevance_scores,
                                    k_values=config.k_values,
                                )
                                all_metrics[strategy].append(metrics)

                    except asyncio.TimeoutError:
                        total_executions += 1
                        if not is_warmup:
                            all_results[strategy].append(
                                StrategyResult(
                                    strategy_name=strategy,
                                    query_id=query.query_id,
                                    success=False,
                                    error="Timeout",
                                )
                            )
                    except Exception as e:
                        total_executions += 1
                        logger.exception(f"Error executing {strategy} for {query.query_id}")
                        if not is_warmup:
                            all_results[strategy].append(
                                StrategyResult(
                                    strategy_name=strategy,
                                    query_id=query.query_id,
                                    success=False,
                                    error=str(e),
                                )
                            )

        # Calculate statistics for each strategy
        statistics = {}
        for strategy in config.strategies:
            statistics[strategy] = _calculate_statistics(
                all_results[strategy],
                all_metrics[strategy],
                config.k_values,
            )

        # Calculate rankings
        rankings = _calculate_rankings(statistics, config.k_values)

        # Calculate duration
        duration = time.time() - start_time
        completed_at = datetime.now(UTC)

        return BenchmarkReport(
            config=config,
            statistics=statistics,
            rankings=rankings,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=duration,
            total_queries=len(benchmark_queries),
            total_executions=total_executions,
        )

    async def _execute_with_timeout(
        self,
        strategy: str,
        query: BenchmarkQuery,
        timeout: float,
    ) -> StrategyResult:
        """
        Execute a strategy with timeout and semaphore.

        Args:
            strategy: Strategy name.
            query: Benchmark query.
            timeout: Timeout in seconds.

        Returns:
            StrategyResult from execution.
        """
        assert self._semaphore is not None

        async with self._semaphore:
            start_time = time.time()

            try:
                result = await asyncio.wait_for(
                    self._executor(
                        strategy,
                        {
                            "query_id": query.query_id,
                            "query": query.query,
                            "ground_truth_ids": query.ground_truth_ids,
                        },
                    ),
                    timeout=timeout,
                )

                # Update latency if not set
                if result.latency_ms == 0:
                    result.latency_ms = int((time.time() - start_time) * 1000)

                return result

            except asyncio.TimeoutError:
                return StrategyResult(
                    strategy_name=strategy,
                    query_id=query.query_id,
                    latency_ms=int(timeout * 1000),
                    success=False,
                    error="Timeout",
                )

    async def run_single(
        self,
        query: BenchmarkQuery | dict[str, Any],
        config: BenchmarkConfig,
    ) -> dict[str, StrategyResult]:
        """
        Run benchmark for a single query across all strategies.

        Args:
            query: The query to benchmark.
            config: Benchmark configuration.

        Returns:
            Dictionary of strategy name to result.
        """
        if isinstance(query, dict):
            query = BenchmarkQuery(**query)

        self._semaphore = asyncio.Semaphore(config.max_concurrent)

        results = {}
        for strategy in config.strategies:
            result = await self._execute_with_timeout(
                strategy,
                query,
                config.timeout_seconds,
            )
            results[strategy] = result

        return results
