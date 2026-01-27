"""
Unit tests for Comparison Result Aggregator.

Tests cover:
- Metrics extraction
- Ranking calculations
- Best overall determination
- Comparison result methods
"""

from __future__ import annotations

import pytest

from orchestration.comparison import (
    AggregatorConfig,
    ComparisonAggregator,
    ComparisonResult,
    RankingCriteria,
    StrategyMetrics,
    StrategyRanking,
    compare_results,
)
from orchestration.executor import ParallelExecutionResult
from orchestration.models import Document, ExecutionResult


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_results() -> dict[str, ExecutionResult]:
    """Create sample execution results."""
    return {
        "fast": ExecutionResult(
            documents=[
                Document(
                    id="doc1",
                    content="Fast result",
                    title="Fast",
                    source="test.md",
                    similarity=0.8,
                )
            ],
            query="test",
            strategy_name="fast",
            latency_ms=50,
            cost_usd=0.001,
        ),
        "accurate": ExecutionResult(
            documents=[
                Document(
                    id="doc2",
                    content="Accurate result",
                    title="Accurate",
                    source="test.md",
                    similarity=0.95,
                ),
                Document(
                    id="doc3",
                    content="Another result",
                    title="Another",
                    source="test.md",
                    similarity=0.9,
                ),
            ],
            query="test",
            strategy_name="accurate",
            latency_ms=200,
            cost_usd=0.005,
        ),
        "cheap": ExecutionResult(
            documents=[
                Document(
                    id="doc4",
                    content="Cheap result",
                    title="Cheap",
                    source="test.md",
                    similarity=0.7,
                )
            ],
            query="test",
            strategy_name="cheap",
            latency_ms=100,
            cost_usd=0.0005,
        ),
    }


@pytest.fixture
def parallel_result(sample_results: dict[str, ExecutionResult]) -> ParallelExecutionResult:
    """Create a parallel execution result."""
    result = ParallelExecutionResult()
    result.results = sample_results
    result.total_latency_ms = 250
    result.total_cost_usd = sum(r.cost_usd for r in sample_results.values())
    return result


# =============================================================================
# Test: StrategyMetrics
# =============================================================================


class TestStrategyMetrics:
    """Tests for StrategyMetrics."""

    def test_from_result(self, sample_results: dict[str, ExecutionResult]):
        """Test creating metrics from result."""
        metrics = StrategyMetrics.from_result(sample_results["fast"])

        assert metrics.strategy_name == "fast"
        assert metrics.latency_ms == 50
        assert metrics.cost_usd == 0.001
        assert metrics.document_count == 1
        assert metrics.top_similarity == 0.8

    def test_avg_similarity_calculation(self, sample_results: dict[str, ExecutionResult]):
        """Test average similarity calculation."""
        metrics = StrategyMetrics.from_result(sample_results["accurate"])

        assert metrics.avg_similarity == pytest.approx(0.925)


# =============================================================================
# Test: StrategyRanking
# =============================================================================


class TestStrategyRanking:
    """Tests for StrategyRanking."""

    def test_ranking_creation(self):
        """Test ranking creation."""
        ranking = StrategyRanking(
            strategy_name="test",
            rank=1,
            score=0.95,
            relative_score=1.0,
        )

        assert ranking.strategy_name == "test"
        assert ranking.rank == 1
        assert ranking.score == 0.95


# =============================================================================
# Test: ComparisonResult
# =============================================================================


class TestComparisonResult:
    """Tests for ComparisonResult."""

    def test_best_by_criteria(self):
        """Test getting best by criteria."""
        result = ComparisonResult(
            query="test",
            rankings={
                RankingCriteria.LATENCY: [
                    StrategyRanking("fast", 1, 50, 1.0),
                    StrategyRanking("slow", 2, 200, 0.25),
                ],
            },
        )

        assert result.best_by(RankingCriteria.LATENCY) == "fast"
        assert result.best_by("latency") == "fast"

    def test_best_by_returns_none_for_missing(self):
        """Test best_by returns None for missing criteria."""
        result = ComparisonResult(query="test")

        assert result.best_by(RankingCriteria.COST) is None

    def test_get_ranking(self):
        """Test getting ranking list."""
        rankings = [
            StrategyRanking("a", 1, 0.9, 1.0),
            StrategyRanking("b", 2, 0.8, 0.89),
        ]
        result = ComparisonResult(
            query="test",
            rankings={RankingCriteria.ACCURACY: rankings},
        )

        assert result.get_ranking(RankingCriteria.ACCURACY) == rankings

    def test_to_dict(self):
        """Test serialization."""
        result = ComparisonResult(
            query="test query",
            best_overall="fast",
            total_cost=0.01,
            total_latency_ms=100,
        )

        d = result.to_dict()

        assert d["query"] == "test query"
        assert d["best_overall"] == "fast"


# =============================================================================
# Test: ComparisonAggregator
# =============================================================================


class TestComparisonAggregator:
    """Tests for ComparisonAggregator."""

    def test_aggregate_parallel_result(
        self,
        parallel_result: ParallelExecutionResult,
    ):
        """Test aggregating parallel results."""
        aggregator = ComparisonAggregator()
        result = aggregator.aggregate(parallel_result, query="test")

        assert result.query == "test"
        assert len(result.metrics) == 3
        assert "fast" in result.metrics
        assert "accurate" in result.metrics

    def test_aggregate_calculates_rankings(
        self,
        parallel_result: ParallelExecutionResult,
    ):
        """Test that aggregation calculates rankings."""
        aggregator = ComparisonAggregator()
        result = aggregator.aggregate(parallel_result)

        # Should have rankings for all criteria
        assert RankingCriteria.LATENCY in result.rankings
        assert RankingCriteria.COST in result.rankings
        assert RankingCriteria.ACCURACY in result.rankings

    def test_latency_ranking_correct(
        self,
        parallel_result: ParallelExecutionResult,
    ):
        """Test latency ranking is correct (lowest = best)."""
        aggregator = ComparisonAggregator()
        result = aggregator.aggregate(parallel_result)

        ranking = result.get_ranking(RankingCriteria.LATENCY)
        # Fast (50ms) should be #1
        assert ranking[0].strategy_name == "fast"

    def test_cost_ranking_correct(
        self,
        parallel_result: ParallelExecutionResult,
    ):
        """Test cost ranking is correct (lowest = best)."""
        aggregator = ComparisonAggregator()
        result = aggregator.aggregate(parallel_result)

        ranking = result.get_ranking(RankingCriteria.COST)
        # Cheap (0.0005) should be #1
        assert ranking[0].strategy_name == "cheap"

    def test_accuracy_ranking_correct(
        self,
        parallel_result: ParallelExecutionResult,
    ):
        """Test accuracy ranking is correct (highest = best)."""
        aggregator = ComparisonAggregator()
        result = aggregator.aggregate(parallel_result)

        ranking = result.get_ranking(RankingCriteria.ACCURACY)
        # Accurate (0.95) should be #1
        assert ranking[0].strategy_name == "accurate"

    def test_best_overall_calculated(
        self,
        parallel_result: ParallelExecutionResult,
    ):
        """Test best overall is calculated."""
        aggregator = ComparisonAggregator()
        result = aggregator.aggregate(parallel_result)

        assert result.best_overall is not None
        assert result.best_overall in ["fast", "accurate", "cheap"]

    def test_aggregate_from_results(
        self,
        sample_results: dict[str, ExecutionResult],
    ):
        """Test aggregating from results dictionary."""
        aggregator = ComparisonAggregator()
        result = aggregator.aggregate_from_results(sample_results, query="test")

        assert len(result.metrics) == 3
        assert result.best_overall is not None

    def test_handles_errors(self):
        """Test handling of execution errors."""
        parallel_result = ParallelExecutionResult()
        parallel_result.results = {"success": ExecutionResult(
            documents=[],
            query="test",
            strategy_name="success",
            latency_ms=100,
            cost_usd=0.01,
        )}
        parallel_result.errors = {"failed": ValueError("Test error")}

        aggregator = ComparisonAggregator()
        result = aggregator.aggregate(parallel_result)

        assert "success" in result.metrics
        assert "failed" in result.errors


# =============================================================================
# Test: Configuration
# =============================================================================


class TestConfiguration:
    """Tests for aggregator configuration."""

    def test_custom_weights(
        self,
        parallel_result: ParallelExecutionResult,
    ):
        """Test custom weight configuration."""
        # Weight accuracy heavily
        config = AggregatorConfig(
            weight_latency=0.0,
            weight_cost=0.0,
            weight_accuracy=1.0,
        )
        aggregator = ComparisonAggregator(config)
        result = aggregator.aggregate(parallel_result)

        # With accuracy weighted 100%, "accurate" should be best
        assert result.best_overall == "accurate"


# =============================================================================
# Test: Convenience Functions
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_compare_results_function(
        self,
        parallel_result: ParallelExecutionResult,
    ):
        """Test compare_results convenience function."""
        result = compare_results(parallel_result, query="test query")

        assert result.query == "test query"
        assert len(result.metrics) == 3
