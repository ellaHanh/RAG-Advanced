"""
RAG-Advanced IR Metrics Calculation.

Calculate Information Retrieval metrics for evaluating RAG strategy performance.
Supports Precision@k, Recall@k, MRR (Mean Reciprocal Rank), and NDCG@k.

Uses the ir-measures library for standardized metric calculations.

Usage:
    from evaluation.metrics import calculate_metrics, EvaluationMetrics

    metrics = calculate_metrics(
        retrieved_ids=["doc1", "doc3", "doc5"],
        ground_truth_ids=["doc1", "doc2", "doc3"],
        k_values=[3, 5, 10],
    )

    print(f"Precision@3: {metrics.precision[3]}")
    print(f"Recall@5: {metrics.recall[5]}")
    print(f"MRR: {metrics.mrr}")
    print(f"NDCG@10: {metrics.ndcg[10]}")
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from orchestration.errors import InvalidInputError


logger = logging.getLogger(__name__)


# =============================================================================
# Default Configuration
# =============================================================================

DEFAULT_K_VALUES = [3, 5, 10]


# =============================================================================
# Models
# =============================================================================


class EvaluationMetrics(BaseModel):
    """
    Container for all calculated IR metrics.

    Attributes:
        precision: Precision@k values for each k.
        recall: Recall@k values for each k.
        mrr: Mean Reciprocal Rank.
        ndcg: NDCG@k values for each k.
        k_values: The k values used for calculation.
        retrieved_count: Number of retrieved documents.
        relevant_count: Number of relevant documents.
        warnings: Any warnings generated during calculation.
    """

    model_config = ConfigDict(frozen=True)

    precision: dict[int, float] = Field(default_factory=dict, description="Precision@k")
    recall: dict[int, float] = Field(default_factory=dict, description="Recall@k")
    mrr: float = Field(default=0.0, ge=0.0, le=1.0, description="Mean Reciprocal Rank")
    ndcg: dict[int, float] = Field(default_factory=dict, description="NDCG@k")
    k_values: list[int] = Field(default_factory=list, description="K values used")
    retrieved_count: int = Field(default=0, ge=0, description="Retrieved doc count")
    relevant_count: int = Field(default=0, ge=0, description="Relevant doc count")
    warnings: list[str] = Field(default_factory=list, description="Warnings")

    @property
    def has_warnings(self) -> bool:
        """Check if there are any warnings."""
        return len(self.warnings) > 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "precision": self.precision,
            "recall": self.recall,
            "mrr": self.mrr,
            "ndcg": self.ndcg,
            "k_values": self.k_values,
            "retrieved_count": self.retrieved_count,
            "relevant_count": self.relevant_count,
            "warnings": self.warnings,
        }


@dataclass
class MetricsBuilder:
    """Builder for accumulating metrics during calculation."""

    precision: dict[int, float] = field(default_factory=dict)
    recall: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    ndcg: dict[int, float] = field(default_factory=dict)
    k_values: list[int] = field(default_factory=list)
    retrieved_count: int = 0
    relevant_count: int = 0
    warnings: list[str] = field(default_factory=list)

    def add_warning(self, warning: str) -> None:
        """Add a warning message."""
        self.warnings.append(warning)
        logger.warning(warning)

    def build(self) -> EvaluationMetrics:
        """Build the final EvaluationMetrics object."""
        return EvaluationMetrics(
            precision=self.precision,
            recall=self.recall,
            mrr=self.mrr,
            ndcg=self.ndcg,
            k_values=self.k_values,
            retrieved_count=self.retrieved_count,
            relevant_count=self.relevant_count,
            warnings=self.warnings,
        )


# =============================================================================
# Validation
# =============================================================================


def _validate_inputs(
    retrieved_ids: list[str],
    ground_truth_ids: list[str],
    relevance_scores: dict[str, int] | None,
    k_values: list[int],
) -> list[str]:
    """
    Validate inputs and return list of warnings.

    Args:
        retrieved_ids: List of retrieved document IDs.
        ground_truth_ids: List of relevant document IDs.
        relevance_scores: Optional relevance scores (0, 1, or 2).
        k_values: K values for metrics calculation.

    Returns:
        List of warning messages.

    Raises:
        InvalidInputError: If inputs are invalid.
    """
    warnings = []

    # Validate retrieved_ids
    if retrieved_ids is None:
        raise InvalidInputError("retrieved_ids", "Must not be None")
    if not isinstance(retrieved_ids, list):
        raise InvalidInputError("retrieved_ids", "Must be a list")
    for i, doc_id in enumerate(retrieved_ids):
        if not isinstance(doc_id, str):
            raise InvalidInputError("retrieved_ids", f"Element {i} must be a string, got {type(doc_id)}")

    # Validate ground_truth_ids
    if ground_truth_ids is None:
        raise InvalidInputError("ground_truth_ids", "Must not be None")
    if not isinstance(ground_truth_ids, list):
        raise InvalidInputError("ground_truth_ids", "Must be a list")
    for i, doc_id in enumerate(ground_truth_ids):
        if not isinstance(doc_id, str):
            raise InvalidInputError("ground_truth_ids", f"Element {i} must be a string, got {type(doc_id)}")

    # Validate k_values
    if not k_values:
        raise InvalidInputError("k_values", "Must not be empty")
    for k in k_values:
        if not isinstance(k, int) or k < 1:
            raise InvalidInputError("k_values", f"Each k must be a positive integer, got {k}")

    # Validate relevance_scores if provided
    if relevance_scores is not None:
        if not isinstance(relevance_scores, dict):
            raise InvalidInputError("relevance_scores", "Must be a dictionary")
        for doc_id, score in relevance_scores.items():
            if not isinstance(doc_id, str):
                raise InvalidInputError("relevance_scores", f"Keys must be strings, got {type(doc_id)}")
            if not isinstance(score, int) or score not in (0, 1, 2):
                raise InvalidInputError(
                    "relevance_scores",
                    f"Scores must be 0, 1, or 2, got {score} for {doc_id}"
                )

    # Generate warnings for edge cases
    if not retrieved_ids:
        warnings.append("No retrieved documents provided")

    if not ground_truth_ids:
        warnings.append("No ground truth documents provided")

    # Check for duplicates
    if len(retrieved_ids) != len(set(retrieved_ids)):
        warnings.append("Duplicate documents in retrieved_ids")

    # Check k values against retrieved count
    for k in k_values:
        if k > len(retrieved_ids):
            warnings.append(f"k={k} exceeds retrieved document count ({len(retrieved_ids)})")

    return warnings


# =============================================================================
# Metric Calculations
# =============================================================================


def _calculate_precision_at_k(
    retrieved_ids: list[str],
    relevant_set: set[str],
    k: int,
) -> float:
    """
    Calculate Precision@k.

    Precision@k = |relevant ∩ retrieved@k| / k

    Args:
        retrieved_ids: Retrieved document IDs in ranked order.
        relevant_set: Set of relevant document IDs.
        k: Number of top results to consider.

    Returns:
        Precision@k value between 0 and 1.
    """
    if k <= 0:
        return 0.0

    # Get top-k retrieved docs
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0

    # Count relevant docs in top-k
    relevant_in_top_k = sum(1 for doc_id in top_k if doc_id in relevant_set)

    return relevant_in_top_k / k


def _calculate_recall_at_k(
    retrieved_ids: list[str],
    relevant_set: set[str],
    k: int,
) -> float:
    """
    Calculate Recall@k.

    Recall@k = |relevant ∩ retrieved@k| / |relevant|

    Args:
        retrieved_ids: Retrieved document IDs in ranked order.
        relevant_set: Set of relevant document IDs.
        k: Number of top results to consider.

    Returns:
        Recall@k value between 0 and 1.
    """
    if not relevant_set:
        return 0.0

    # Get top-k retrieved docs
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0

    # Count relevant docs in top-k
    relevant_in_top_k = sum(1 for doc_id in top_k if doc_id in relevant_set)

    return relevant_in_top_k / len(relevant_set)


def _calculate_mrr(
    retrieved_ids: list[str],
    relevant_set: set[str],
) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR).

    MRR = 1 / rank_of_first_relevant_doc

    Args:
        retrieved_ids: Retrieved document IDs in ranked order.
        relevant_set: Set of relevant document IDs.

    Returns:
        MRR value between 0 and 1.
    """
    if not retrieved_ids or not relevant_set:
        return 0.0

    # Find rank of first relevant document (1-indexed)
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_set:
            return 1.0 / rank

    return 0.0


def _calculate_dcg_at_k(
    retrieved_ids: list[str],
    relevance_scores: dict[str, int],
    k: int,
) -> float:
    """
    Calculate Discounted Cumulative Gain at k.

    DCG@k = Σ (2^rel_i - 1) / log2(i + 1) for i in 1..k

    Args:
        retrieved_ids: Retrieved document IDs in ranked order.
        relevance_scores: Relevance scores for documents (0, 1, or 2).
        k: Number of top results to consider.

    Returns:
        DCG@k value.
    """
    if not retrieved_ids or k <= 0:
        return 0.0

    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k], start=1):
        rel = relevance_scores.get(doc_id, 0)
        # Using (2^rel - 1) / log2(i + 1) formula
        dcg += (pow(2, rel) - 1) / math.log2(i + 1)

    return dcg


def _calculate_ndcg_at_k(
    retrieved_ids: list[str],
    relevance_scores: dict[str, int],
    k: int,
) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain at k.

    NDCG@k = DCG@k / IDCG@k

    Args:
        retrieved_ids: Retrieved document IDs in ranked order.
        relevance_scores: Relevance scores for documents (0, 1, or 2).
        k: Number of top results to consider.

    Returns:
        NDCG@k value between 0 and 1.
    """
    if not retrieved_ids or not relevance_scores:
        return 0.0

    # Calculate DCG
    dcg = _calculate_dcg_at_k(retrieved_ids, relevance_scores, k)

    # Calculate ideal DCG (sort by relevance descending)
    sorted_docs = sorted(
        relevance_scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )
    ideal_order = [doc_id for doc_id, _ in sorted_docs]
    idcg = _calculate_dcg_at_k(ideal_order, relevance_scores, k)

    if idcg == 0:
        return 0.0

    return dcg / idcg


# =============================================================================
# Main Function
# =============================================================================


def calculate_metrics(
    retrieved_ids: list[str],
    ground_truth_ids: list[str],
    relevance_scores: dict[str, int] | None = None,
    k_values: list[int] | None = None,
) -> EvaluationMetrics:
    """
    Calculate comprehensive IR metrics.

    Args:
        retrieved_ids: List of retrieved document IDs in ranked order.
        ground_truth_ids: List of relevant document IDs.
        relevance_scores: Optional graded relevance (0=not, 1=partial, 2=highly).
                         If not provided, binary relevance is assumed.
        k_values: K values for metrics. Defaults to [3, 5, 10].

    Returns:
        EvaluationMetrics with all calculated metrics.

    Raises:
        InvalidInputError: If inputs are invalid.

    Example:
        >>> metrics = calculate_metrics(
        ...     retrieved_ids=["doc1", "doc3", "doc5", "doc2"],
        ...     ground_truth_ids=["doc1", "doc2", "doc3"],
        ...     k_values=[3, 5],
        ... )
        >>> print(f"Precision@3: {metrics.precision[3]:.2f}")
        >>> print(f"MRR: {metrics.mrr:.2f}")
    """
    # Set default k values
    if k_values is None:
        k_values = DEFAULT_K_VALUES.copy()

    # Sort k values
    k_values = sorted(set(k_values))

    # Validate inputs
    warnings = _validate_inputs(
        retrieved_ids,
        ground_truth_ids,
        relevance_scores,
        k_values,
    )

    # Initialize builder
    builder = MetricsBuilder(
        k_values=k_values,
        retrieved_count=len(retrieved_ids),
        relevant_count=len(ground_truth_ids),
        warnings=warnings,
    )

    # Handle edge cases - still populate metrics with 0 values
    if not retrieved_ids or not ground_truth_ids:
        for k in k_values:
            builder.precision[k] = 0.0
            builder.recall[k] = 0.0
            builder.ndcg[k] = 0.0
        builder.mrr = 0.0
        return builder.build()

    # Create relevance scores if not provided (binary relevance)
    if relevance_scores is None:
        relevance_scores = {doc_id: 1 for doc_id in ground_truth_ids}
    else:
        # Ensure all ground truth docs have scores
        for doc_id in ground_truth_ids:
            if doc_id not in relevance_scores:
                relevance_scores[doc_id] = 1

    # Create relevant set for quick lookup
    relevant_set = set(ground_truth_ids)

    # Calculate metrics for each k
    for k in k_values:
        builder.precision[k] = _calculate_precision_at_k(retrieved_ids, relevant_set, k)
        builder.recall[k] = _calculate_recall_at_k(retrieved_ids, relevant_set, k)
        builder.ndcg[k] = _calculate_ndcg_at_k(retrieved_ids, relevance_scores, k)

    # Calculate MRR (not dependent on k)
    builder.mrr = _calculate_mrr(retrieved_ids, relevant_set)

    logger.debug(
        "Calculated metrics",
        extra={
            "retrieved_count": len(retrieved_ids),
            "relevant_count": len(ground_truth_ids),
            "k_values": k_values,
            "mrr": builder.mrr,
        },
    )

    return builder.build()


# =============================================================================
# Batch Metrics
# =============================================================================


def calculate_batch_metrics(
    queries: list[dict[str, Any]],
    k_values: list[int] | None = None,
) -> dict[str, Any]:
    """
    Calculate metrics for multiple queries and aggregate results.

    Args:
        queries: List of dicts with 'retrieved_ids' and 'ground_truth_ids'.
        k_values: K values for metrics.

    Returns:
        Dictionary with per-query and aggregate metrics.

    Example:
        >>> queries = [
        ...     {"retrieved_ids": ["a", "b"], "ground_truth_ids": ["a"]},
        ...     {"retrieved_ids": ["c", "d"], "ground_truth_ids": ["d"]},
        ... ]
        >>> results = calculate_batch_metrics(queries)
    """
    if k_values is None:
        k_values = DEFAULT_K_VALUES.copy()

    per_query_metrics = []
    all_precision = {k: [] for k in k_values}
    all_recall = {k: [] for k in k_values}
    all_ndcg = {k: [] for k in k_values}
    all_mrr = []

    for i, query in enumerate(queries):
        try:
            metrics = calculate_metrics(
                retrieved_ids=query.get("retrieved_ids", []),
                ground_truth_ids=query.get("ground_truth_ids", []),
                relevance_scores=query.get("relevance_scores"),
                k_values=k_values,
            )
            per_query_metrics.append({
                "query_id": query.get("query_id", f"q{i}"),
                "metrics": metrics.to_dict(),
            })

            # Accumulate for averaging
            for k in k_values:
                all_precision[k].append(metrics.precision.get(k, 0.0))
                all_recall[k].append(metrics.recall.get(k, 0.0))
                all_ndcg[k].append(metrics.ndcg.get(k, 0.0))
            all_mrr.append(metrics.mrr)

        except InvalidInputError as e:
            per_query_metrics.append({
                "query_id": query.get("query_id", f"q{i}"),
                "error": str(e),
            })

    # Calculate averages
    def safe_avg(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    aggregate = {
        "avg_precision": {k: safe_avg(all_precision[k]) for k in k_values},
        "avg_recall": {k: safe_avg(all_recall[k]) for k in k_values},
        "avg_ndcg": {k: safe_avg(all_ndcg[k]) for k in k_values},
        "avg_mrr": safe_avg(all_mrr),
        "query_count": len(queries),
        "successful_count": len([q for q in per_query_metrics if "metrics" in q]),
    }

    return {
        "per_query": per_query_metrics,
        "aggregate": aggregate,
        "k_values": k_values,
    }
