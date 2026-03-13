"""
RAG-Advanced Comparison Result Aggregator.

Aggregate results from parallel strategy execution for comparison.
Calculate relative metrics and rank strategies by latency, cost, and accuracy.

Usage:
    from orchestration.comparison import ComparisonAggregator

    aggregator = ComparisonAggregator()
    comparison = aggregator.aggregate(parallel_result)
    
    # Get rankings
    best_by_latency = comparison.best_by("latency")
    best_by_cost = comparison.best_by("cost")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from orchestration.executor import ParallelExecutionResult
from orchestration.models import ExecutionResult


logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================


class RankingCriteria(str, Enum):
    """Criteria for ranking strategies."""

    LATENCY = "latency"
    COST = "cost"
    ACCURACY = "accuracy"
    DOCUMENT_COUNT = "document_count"
    TOP_SIMILARITY = "top_similarity"


# =============================================================================
# Models
# =============================================================================


@dataclass
class StrategyMetrics:
    """
    Metrics for a single strategy result.

    Attributes:
        strategy_name: Name of the strategy.
        latency_ms: Execution latency in milliseconds.
        cost_usd: Execution cost in USD.
        document_count: Number of documents returned.
        top_similarity: Similarity score of top document.
        avg_similarity: Average similarity of documents.
    """

    strategy_name: str
    latency_ms: int = 0
    cost_usd: float = 0.0
    document_count: int = 0
    top_similarity: float = 0.0
    avg_similarity: float = 0.0

    @classmethod
    def from_result(cls, result: ExecutionResult) -> "StrategyMetrics":
        """Create metrics from execution result."""
        docs = result.documents
        similarities = [d.similarity or 0.0 for d in docs]

        return cls(
            strategy_name=result.strategy_name,
            latency_ms=result.latency_ms,
            cost_usd=result.cost_usd,
            document_count=len(docs),
            top_similarity=max(similarities) if similarities else 0.0,
            avg_similarity=sum(similarities) / len(similarities) if similarities else 0.0,
        )


@dataclass
class StrategyRanking:
    """
    Ranking information for a strategy.

    Attributes:
        strategy_name: Name of the strategy.
        rank: Rank position (1 = best).
        score: Score used for ranking.
        relative_score: Score relative to best (0-1).
    """

    strategy_name: str
    rank: int
    score: float
    relative_score: float = 1.0


@dataclass
class ComparisonResult:
    """
    Aggregated comparison of multiple strategy results.

    Attributes:
        query: The query that was executed.
        metrics: Metrics per strategy.
        rankings: Rankings per criteria.
        best_overall: Best strategy considering all criteria.
        total_cost: Total cost of all strategies.
        total_latency_ms: Total latency (wall clock).
    """

    query: str
    metrics: dict[str, StrategyMetrics] = field(default_factory=dict)
    rankings: dict[RankingCriteria, list[StrategyRanking]] = field(default_factory=dict)
    best_overall: str | None = None
    total_cost: float = 0.0
    total_latency_ms: int = 0
    errors: dict[str, str] = field(default_factory=dict)

    def best_by(self, criteria: RankingCriteria | str) -> str | None:
        """
        Get best strategy by criteria.

        Args:
            criteria: Ranking criteria.

        Returns:
            Name of best strategy or None.
        """
        if isinstance(criteria, str):
            criteria = RankingCriteria(criteria)

        rankings = self.rankings.get(criteria)
        if rankings and len(rankings) > 0:
            return rankings[0].strategy_name
        return None

    def get_ranking(
        self,
        criteria: RankingCriteria | str,
    ) -> list[StrategyRanking]:
        """
        Get ranking list for criteria.

        Args:
            criteria: Ranking criteria.

        Returns:
            List of StrategyRanking.
        """
        if isinstance(criteria, str):
            criteria = RankingCriteria(criteria)
        return self.rankings.get(criteria, [])

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "query": self.query,
            "best_overall": self.best_overall,
            "total_cost": self.total_cost,
            "total_latency_ms": self.total_latency_ms,
            "metrics": {
                name: {
                    "latency_ms": m.latency_ms,
                    "cost_usd": m.cost_usd,
                    "document_count": m.document_count,
                    "top_similarity": m.top_similarity,
                }
                for name, m in self.metrics.items()
            },
            "rankings": {
                criteria.value: [
                    {"strategy": r.strategy_name, "rank": r.rank, "score": r.score}
                    for r in rankings
                ]
                for criteria, rankings in self.rankings.items()
            },
            "errors": self.errors,
        }


# =============================================================================
# Configuration
# =============================================================================


class AggregatorConfig(BaseModel):
    """
    Configuration for comparison aggregator.

    Attributes:
        ranking_criteria: Criteria to use for rankings.
        weight_latency: Weight for latency in overall score.
        weight_cost: Weight for cost in overall score.
        weight_accuracy: Weight for accuracy in overall score.
    """

    model_config = ConfigDict(frozen=True)

    ranking_criteria: list[RankingCriteria] = Field(
        default=[
            RankingCriteria.LATENCY,
            RankingCriteria.COST,
            RankingCriteria.ACCURACY,
            RankingCriteria.DOCUMENT_COUNT,
            RankingCriteria.TOP_SIMILARITY,
        ],
        description="Criteria to rank by",
    )
    weight_latency: float = Field(default=0.3, ge=0.0, le=1.0)
    weight_cost: float = Field(default=0.2, ge=0.0, le=1.0)
    weight_accuracy: float = Field(default=0.5, ge=0.0, le=1.0)


# =============================================================================
# Aggregator
# =============================================================================


class ComparisonAggregator:
    """
    Aggregate parallel execution results for comparison.

    Calculates metrics, creates rankings, and determines best strategy.

    Example:
        >>> aggregator = ComparisonAggregator()
        >>> result = aggregator.aggregate(parallel_result)
        >>> print(f"Best by latency: {result.best_by('latency')}")
    """

    def __init__(self, config: AggregatorConfig | None = None) -> None:
        """
        Initialize aggregator.

        Args:
            config: Optional configuration.
        """
        self._config = config or AggregatorConfig()

    def aggregate(
        self,
        parallel_result: ParallelExecutionResult,
        query: str = "",
    ) -> ComparisonResult:
        """
        Aggregate parallel execution results.

        Args:
            parallel_result: Results from parallel execution.
            query: The query that was executed.

        Returns:
            ComparisonResult with metrics and rankings.
        """
        comparison = ComparisonResult(
            query=query,
            total_cost=parallel_result.total_cost_usd,
            total_latency_ms=parallel_result.total_latency_ms,
        )

        # Extract metrics from results
        for name, result in parallel_result.results.items():
            comparison.metrics[name] = StrategyMetrics.from_result(result)

        # Record errors
        for name, error in parallel_result.errors.items():
            comparison.errors[name] = str(error)

        # Calculate rankings for each criteria
        for criteria in self._config.ranking_criteria:
            comparison.rankings[criteria] = self._rank_by_criteria(
                comparison.metrics,
                criteria,
            )

        # Determine best overall
        comparison.best_overall = self._calculate_best_overall(comparison)

        return comparison

    def aggregate_from_results(
        self,
        results: dict[str, ExecutionResult],
        query: str = "",
    ) -> ComparisonResult:
        """
        Aggregate from individual results dictionary.

        Args:
            results: Dictionary of strategy name to ExecutionResult.
            query: The query that was executed.

        Returns:
            ComparisonResult.
        """
        comparison = ComparisonResult(query=query)

        for name, result in results.items():
            comparison.metrics[name] = StrategyMetrics.from_result(result)
            comparison.total_cost += result.cost_usd

        # Calculate latency as max (parallel execution)
        if results:
            comparison.total_latency_ms = max(r.latency_ms for r in results.values())

        # Calculate rankings
        for criteria in self._config.ranking_criteria:
            comparison.rankings[criteria] = self._rank_by_criteria(
                comparison.metrics,
                criteria,
            )

        comparison.best_overall = self._calculate_best_overall(comparison)

        return comparison

    def _rank_by_criteria(
        self,
        metrics: dict[str, StrategyMetrics],
        criteria: RankingCriteria,
    ) -> list[StrategyRanking]:
        """Create ranking for a specific criteria."""
        if not metrics:
            return []

        # Extract scores based on criteria
        scores: list[tuple[str, float]] = []

        for name, m in metrics.items():
            if criteria == RankingCriteria.LATENCY:
                # Lower is better (negative for sorting)
                scores.append((name, -m.latency_ms))
            elif criteria == RankingCriteria.COST:
                # Lower is better
                scores.append((name, -m.cost_usd))
            elif criteria == RankingCriteria.ACCURACY:
                # Higher is better
                scores.append((name, m.top_similarity))
            elif criteria == RankingCriteria.DOCUMENT_COUNT:
                # Higher is better
                scores.append((name, float(m.document_count)))
            elif criteria == RankingCriteria.TOP_SIMILARITY:
                # Higher is better
                scores.append((name, m.top_similarity))

        # Sort by score (higher is better after transformation)
        scores.sort(key=lambda x: x[1], reverse=True)

        # Create rankings
        rankings = []
        best_score = scores[0][1] if scores else 0

        for rank, (name, score) in enumerate(scores, 1):
            relative = (score / best_score) if best_score != 0 else 1.0
            rankings.append(
                StrategyRanking(
                    strategy_name=name,
                    rank=rank,
                    score=abs(score),  # Use absolute score
                    relative_score=abs(relative) if relative >= 0 else 0,
                )
            )

        return rankings

    def _calculate_best_overall(
        self,
        comparison: ComparisonResult,
    ) -> str | None:
        """Calculate best overall strategy using weighted scoring."""
        if not comparison.metrics:
            return None

        # Normalize scores for each metric
        strategies = list(comparison.metrics.keys())

        # Get position scores (1/rank) for each criteria
        position_scores: dict[str, float] = {s: 0.0 for s in strategies}

        if RankingCriteria.LATENCY in comparison.rankings:
            for r in comparison.rankings[RankingCriteria.LATENCY]:
                position_scores[r.strategy_name] += (
                    self._config.weight_latency * (1 / r.rank)
                )

        if RankingCriteria.COST in comparison.rankings:
            for r in comparison.rankings[RankingCriteria.COST]:
                position_scores[r.strategy_name] += (
                    self._config.weight_cost * (1 / r.rank)
                )

        if RankingCriteria.ACCURACY in comparison.rankings:
            for r in comparison.rankings[RankingCriteria.ACCURACY]:
                position_scores[r.strategy_name] += (
                    self._config.weight_accuracy * (1 / r.rank)
                )

        # Return strategy with highest weighted score
        best = max(position_scores.items(), key=lambda x: x[1])
        return best[0]


# =============================================================================
# Convenience Functions
# =============================================================================


def compare_results(
    parallel_result: ParallelExecutionResult,
    query: str = "",
) -> ComparisonResult:
    """
    Compare parallel execution results (convenience function).

    Args:
        parallel_result: Results from parallel execution.
        query: The query that was executed.

    Returns:
        ComparisonResult.
    """
    aggregator = ComparisonAggregator()
    return aggregator.aggregate(parallel_result, query)
