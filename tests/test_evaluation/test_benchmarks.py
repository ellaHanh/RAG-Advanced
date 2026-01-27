"""
Unit tests for Benchmark Runner.

Tests cover:
- BenchmarkRunner execution
- Statistical calculations (percentiles)
- Strategy rankings
- Configuration validation
- Error handling
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from evaluation.benchmarks import (
    BenchmarkConfig,
    BenchmarkQuery,
    BenchmarkReport,
    BenchmarkRunner,
    StrategyResult,
    StrategyStatistics,
    _calculate_percentile,
    _calculate_rankings,
    _calculate_statistics,
)
from evaluation.metrics import EvaluationMetrics


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_config() -> BenchmarkConfig:
    """Create a sample benchmark configuration."""
    return BenchmarkConfig(
        strategies=["standard", "reranking"],
        iterations=2,
        k_values=[3, 5],
        timeout_seconds=10.0,
        max_concurrent=3,
    )


@pytest.fixture
def sample_queries() -> list[BenchmarkQuery]:
    """Create sample benchmark queries."""
    return [
        BenchmarkQuery(
            query_id="q1",
            query="What is machine learning?",
            ground_truth_ids=["doc1", "doc2"],
        ),
        BenchmarkQuery(
            query_id="q2",
            query="How does RAG work?",
            ground_truth_ids=["doc3", "doc4", "doc5"],
        ),
    ]


@pytest.fixture
def sample_results() -> list[StrategyResult]:
    """Create sample strategy results."""
    return [
        StrategyResult(
            strategy_name="standard",
            query_id="q1",
            retrieved_ids=["doc1", "doc3", "doc2"],
            latency_ms=100,
            cost_usd=0.001,
            success=True,
        ),
        StrategyResult(
            strategy_name="standard",
            query_id="q2",
            retrieved_ids=["doc3", "doc5"],
            latency_ms=120,
            cost_usd=0.0012,
            success=True,
        ),
        StrategyResult(
            strategy_name="standard",
            query_id="q1",
            retrieved_ids=["doc1", "doc2"],
            latency_ms=110,
            cost_usd=0.001,
            success=True,
        ),
    ]


# =============================================================================
# Test: BenchmarkConfig
# =============================================================================


class TestBenchmarkConfig:
    """Tests for BenchmarkConfig model."""

    def test_valid_config(self, sample_config: BenchmarkConfig):
        """Test valid configuration creation."""
        assert sample_config.strategies == ["standard", "reranking"]
        assert sample_config.iterations == 2
        assert sample_config.k_values == [3, 5]

    def test_default_values(self):
        """Test default configuration values."""
        config = BenchmarkConfig(strategies=["standard"])

        assert config.iterations == 3
        assert config.k_values == [3, 5, 10]
        assert config.timeout_seconds == 60.0
        assert config.max_concurrent == 5

    def test_frozen_config(self, sample_config: BenchmarkConfig):
        """Test that config is frozen/immutable."""
        with pytest.raises(Exception):
            sample_config.iterations = 5  # type: ignore


# =============================================================================
# Test: BenchmarkQuery
# =============================================================================


class TestBenchmarkQuery:
    """Tests for BenchmarkQuery model."""

    def test_valid_query(self):
        """Test valid query creation."""
        query = BenchmarkQuery(
            query_id="q1",
            query="Test query",
            ground_truth_ids=["doc1", "doc2"],
        )

        assert query.query_id == "q1"
        assert query.query == "Test query"
        assert query.ground_truth_ids == ["doc1", "doc2"]

    def test_query_with_relevance_scores(self):
        """Test query with graded relevance."""
        query = BenchmarkQuery(
            query_id="q1",
            query="Test query",
            ground_truth_ids=["doc1", "doc2"],
            relevance_scores={"doc1": 2, "doc2": 1},
        )

        assert query.relevance_scores["doc1"] == 2


# =============================================================================
# Test: Percentile Calculation
# =============================================================================


class TestPercentileCalculation:
    """Tests for percentile calculation."""

    def test_p50_even_list(self):
        """Test p50 for even-length list."""
        values = [10, 20, 30, 40]
        p50 = _calculate_percentile(values, 50)
        assert p50 == pytest.approx(25.0)

    def test_p50_odd_list(self):
        """Test p50 for odd-length list."""
        values = [10, 20, 30, 40, 50]
        p50 = _calculate_percentile(values, 50)
        assert p50 == pytest.approx(30.0)

    def test_p95(self):
        """Test p95 calculation."""
        values = list(range(1, 101))  # 1 to 100
        p95 = _calculate_percentile(values, 95)
        assert 95 <= p95 <= 96

    def test_p99(self):
        """Test p99 calculation."""
        values = list(range(1, 101))
        p99 = _calculate_percentile(values, 99)
        assert 99 <= p99 <= 100

    def test_empty_list(self):
        """Test percentile of empty list."""
        assert _calculate_percentile([], 50) == 0.0

    def test_single_value(self):
        """Test percentile of single value."""
        assert _calculate_percentile([42], 50) == 42.0
        assert _calculate_percentile([42], 95) == 42.0


# =============================================================================
# Test: Statistics Calculation
# =============================================================================


class TestStatisticsCalculation:
    """Tests for statistics calculation."""

    def test_calculate_statistics(self, sample_results: list[StrategyResult]):
        """Test basic statistics calculation."""
        stats = _calculate_statistics(sample_results, [], [3, 5])

        assert stats.strategy_name == "standard"
        assert stats.query_count == 2  # q1 and q2
        assert stats.iteration_count == 3
        assert stats.total_cost == pytest.approx(0.0032)
        assert stats.success_rate == 1.0

    def test_latency_statistics(self, sample_results: list[StrategyResult]):
        """Test latency statistics."""
        stats = _calculate_statistics(sample_results, [], [3, 5])

        # Latencies: 100, 120, 110
        assert stats.latency_p50 == pytest.approx(110.0)
        assert stats.latency_mean == pytest.approx(110.0)
        assert stats.latency_std > 0

    def test_empty_results(self):
        """Test statistics with no results."""
        stats = _calculate_statistics([], [], [3, 5])

        assert stats.strategy_name == "unknown"
        assert stats.query_count == 0
        assert stats.latency_p50 == 0.0

    def test_with_failures(self):
        """Test statistics with failed results."""
        results = [
            StrategyResult(
                strategy_name="test",
                query_id="q1",
                latency_ms=100,
                success=True,
            ),
            StrategyResult(
                strategy_name="test",
                query_id="q2",
                latency_ms=0,
                success=False,
                error="Timeout",
            ),
        ]

        stats = _calculate_statistics(results, [], [3])

        assert stats.success_rate == 0.5
        assert stats.latency_p50 == 100.0  # Only successful result


# =============================================================================
# Test: Rankings
# =============================================================================


class TestRankings:
    """Tests for strategy rankings."""

    def test_calculate_rankings(self):
        """Test ranking calculation."""
        stats = {
            "fast": StrategyStatistics(
                strategy_name="fast",
                latency_p50=50.0,
                latency_p95=100.0,
                avg_cost_per_query=0.002,
                avg_mrr=0.8,
                avg_precision={3: 0.6},
                avg_ndcg={3: 0.7},
            ),
            "slow": StrategyStatistics(
                strategy_name="slow",
                latency_p50=200.0,
                latency_p95=400.0,
                avg_cost_per_query=0.001,
                avg_mrr=0.9,
                avg_precision={3: 0.8},
                avg_ndcg={3: 0.85},
            ),
        }

        rankings = _calculate_rankings(stats, [3])

        # Fast has lower latency
        assert rankings["latency_p50"] == ["fast", "slow"]
        assert rankings["latency_p95"] == ["fast", "slow"]

        # Slow has lower cost
        assert rankings["cost"] == ["slow", "fast"]

        # Slow has higher MRR
        assert rankings["mrr"] == ["slow", "fast"]

        # Slow has higher precision
        assert rankings["precision@3"] == ["slow", "fast"]

    def test_empty_rankings(self):
        """Test rankings with no statistics."""
        rankings = _calculate_rankings({}, [3])
        assert rankings == {}


# =============================================================================
# Test: BenchmarkRunner
# =============================================================================


class TestBenchmarkRunner:
    """Tests for BenchmarkRunner."""

    @pytest.mark.asyncio
    async def test_run_basic(
        self,
        sample_config: BenchmarkConfig,
        sample_queries: list[BenchmarkQuery],
    ):
        """Test basic benchmark run."""
        runner = BenchmarkRunner()
        report = await runner.run(sample_queries, sample_config)

        assert isinstance(report, BenchmarkReport)
        assert report.total_queries == 2
        assert len(report.statistics) == 2  # standard, reranking

    @pytest.mark.asyncio
    async def test_run_with_custom_executor(
        self,
        sample_config: BenchmarkConfig,
        sample_queries: list[BenchmarkQuery],
    ):
        """Test benchmark with custom executor."""

        async def custom_executor(
            strategy: str,
            query_data: dict[str, Any],
        ) -> StrategyResult:
            return StrategyResult(
                strategy_name=strategy,
                query_id=query_data["query_id"],
                retrieved_ids=["doc1", "doc2"] if strategy == "standard" else ["doc3"],
                latency_ms=100 if strategy == "standard" else 200,
                cost_usd=0.001,
                success=True,
            )

        runner = BenchmarkRunner(executor=custom_executor)
        report = await runner.run(sample_queries, sample_config)

        # Standard should have lower latency
        assert report.statistics["standard"].latency_p50 < report.statistics["reranking"].latency_p50

    @pytest.mark.asyncio
    async def test_run_with_dict_queries(self, sample_config: BenchmarkConfig):
        """Test benchmark with dict queries instead of BenchmarkQuery."""
        queries = [
            {
                "query_id": "q1",
                "query": "Test query",
                "ground_truth_ids": ["doc1"],
            }
        ]

        runner = BenchmarkRunner()
        report = await runner.run(queries, sample_config)

        assert report.total_queries == 1

    @pytest.mark.asyncio
    async def test_report_summary(
        self,
        sample_config: BenchmarkConfig,
        sample_queries: list[BenchmarkQuery],
    ):
        """Test report summary generation."""
        runner = BenchmarkRunner()
        report = await runner.run(sample_queries, sample_config)

        summary = report.summary()

        assert "Benchmark Report" in summary
        assert "standard" in summary
        assert "reranking" in summary
        assert "Rankings" in summary

    @pytest.mark.asyncio
    async def test_report_to_dict(
        self,
        sample_config: BenchmarkConfig,
        sample_queries: list[BenchmarkQuery],
    ):
        """Test report serialization."""
        runner = BenchmarkRunner()
        report = await runner.run(sample_queries, sample_config)

        data = report.to_dict()

        assert "benchmark_id" in data
        assert "statistics" in data
        assert "rankings" in data
        assert "duration_seconds" in data

    @pytest.mark.asyncio
    async def test_run_with_timeout(self):
        """Test benchmark handles timeouts."""

        async def slow_executor(
            strategy: str,
            query_data: dict[str, Any],
        ) -> StrategyResult:
            await asyncio.sleep(5.0)  # Simulate slow execution
            return StrategyResult(
                strategy_name=strategy,
                query_id=query_data["query_id"],
                success=True,
            )

        config = BenchmarkConfig(
            strategies=["slow"],
            iterations=1,
            timeout_seconds=1.0,  # Minimum allowed timeout
        )

        queries = [BenchmarkQuery(query_id="q1", query="Test")]

        runner = BenchmarkRunner(executor=slow_executor)
        report = await runner.run(queries, config)

        # Should have timeout result
        stats = report.statistics["slow"]
        assert stats.success_rate < 1.0

    @pytest.mark.asyncio
    async def test_run_with_errors(self):
        """Test benchmark handles executor errors."""

        async def failing_executor(
            strategy: str,
            query_data: dict[str, Any],
        ) -> StrategyResult:
            raise RuntimeError("Test error")

        config = BenchmarkConfig(
            strategies=["failing"],
            iterations=1,
        )

        queries = [BenchmarkQuery(query_id="q1", query="Test")]

        runner = BenchmarkRunner(executor=failing_executor)
        report = await runner.run(queries, config)

        stats = report.statistics["failing"]
        assert stats.success_rate == 0.0

    @pytest.mark.asyncio
    async def test_run_single(
        self,
        sample_config: BenchmarkConfig,
    ):
        """Test running single query benchmark."""
        query = BenchmarkQuery(
            query_id="q1",
            query="Single test",
            ground_truth_ids=["doc1"],
        )

        runner = BenchmarkRunner()
        results = await runner.run_single(query, sample_config)

        assert "standard" in results
        assert "reranking" in results
        assert results["standard"].query_id == "q1"


# =============================================================================
# Test: StrategyStatistics
# =============================================================================


class TestStrategyStatistics:
    """Tests for StrategyStatistics."""

    def test_to_dict(self):
        """Test serialization to dict."""
        stats = StrategyStatistics(
            strategy_name="test",
            query_count=10,
            latency_p50=100.0,
            avg_mrr=0.85,
        )

        data = stats.to_dict()

        assert data["strategy_name"] == "test"
        assert data["query_count"] == 10
        assert data["latency_p50"] == 100.0
        assert data["avg_mrr"] == 0.85


# =============================================================================
# Test: Concurrency Control
# =============================================================================


class TestConcurrencyControl:
    """Tests for concurrency control."""

    @pytest.mark.asyncio
    async def test_max_concurrent_respected(self):
        """Test that max_concurrent is respected."""
        concurrent_count = 0
        max_observed = 0

        async def counting_executor(
            strategy: str,
            query_data: dict[str, Any],
        ) -> StrategyResult:
            nonlocal concurrent_count, max_observed
            concurrent_count += 1
            max_observed = max(max_observed, concurrent_count)
            await asyncio.sleep(0.05)
            concurrent_count -= 1
            return StrategyResult(
                strategy_name=strategy,
                query_id=query_data["query_id"],
                success=True,
            )

        config = BenchmarkConfig(
            strategies=["test"],
            iterations=1,
            max_concurrent=2,
        )

        queries = [
            BenchmarkQuery(query_id=f"q{i}", query=f"Query {i}")
            for i in range(10)
        ]

        runner = BenchmarkRunner(executor=counting_executor)
        await runner.run(queries, config)

        # Max concurrent should not exceed configured value
        assert max_observed <= 2
