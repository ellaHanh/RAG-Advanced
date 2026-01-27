"""
RAG-Advanced Evaluation API Routes.

POST endpoints for calculating IR metrics and batch evaluation.

Routes:
    POST /metrics - Calculate IR metrics for a single query
    POST /batch-metrics - Calculate metrics for multiple queries
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from evaluation.metrics import calculate_metrics, calculate_batch_metrics, EvaluationMetrics


logger = logging.getLogger(__name__)


# =============================================================================
# Request/Response Models
# =============================================================================


class MetricsRequest(BaseModel):
    """
    Request model for calculating IR metrics.

    Attributes:
        retrieved_ids: List of retrieved document IDs (in ranked order).
        ground_truth_ids: List of relevant document IDs.
        k_values: K values for precision/recall/ndcg (default: [1, 3, 5, 10]).
    """

    model_config = ConfigDict(frozen=True)

    retrieved_ids: list[str] = Field(..., description="Retrieved document IDs")
    ground_truth_ids: list[str] = Field(..., description="Ground truth relevant IDs")
    k_values: list[int] = Field(default=[1, 3, 5, 10], description="K values for metrics")


class MetricsResponse(BaseModel):
    """
    Response model for calculated metrics.

    Attributes:
        precision: Precision@k values.
        recall: Recall@k values.
        ndcg: NDCG@k values.
        mrr: Mean Reciprocal Rank.
        retrieved_count: Number of retrieved documents.
        relevant_count: Number of relevant documents.
    """

    model_config = ConfigDict(frozen=True)

    precision: dict[int, float] = Field(..., description="Precision@k")
    recall: dict[int, float] = Field(..., description="Recall@k")
    ndcg: dict[int, float] = Field(..., description="NDCG@k")
    mrr: float = Field(..., description="Mean Reciprocal Rank")
    retrieved_count: int = Field(..., description="Number retrieved")
    relevant_count: int = Field(..., description="Number relevant")


class BatchMetricsRequest(BaseModel):
    """
    Request model for batch metrics calculation.

    Attributes:
        queries: List of query evaluation data.
    """

    model_config = ConfigDict(frozen=True)

    queries: list[MetricsRequest] = Field(..., description="List of queries")


class BatchMetricsResponse(BaseModel):
    """
    Response model for batch metrics.

    Attributes:
        query_count: Number of queries evaluated.
        average_metrics: Average metrics across all queries.
        per_query_metrics: Metrics for each query.
    """

    model_config = ConfigDict(frozen=True)

    query_count: int = Field(..., description="Number of queries")
    average_metrics: MetricsResponse = Field(..., description="Average metrics")
    per_query_metrics: list[MetricsResponse] | None = Field(
        default=None,
        description="Per-query metrics (optional)",
    )


# =============================================================================
# Metric Calculation Functions
# =============================================================================


def calculate_metrics_endpoint(request: MetricsRequest) -> MetricsResponse:
    """
    Calculate IR metrics for a single query.

    Args:
        request: Metrics request with retrieved and ground truth IDs.

    Returns:
        MetricsResponse with calculated metrics.
    """
    metrics = calculate_metrics(
        retrieved_ids=request.retrieved_ids,
        ground_truth_ids=request.ground_truth_ids,
        k_values=request.k_values,
    )

    return MetricsResponse(
        precision=metrics.precision,
        recall=metrics.recall,
        ndcg=metrics.ndcg,
        mrr=metrics.mrr,
        retrieved_count=len(request.retrieved_ids),
        relevant_count=len(request.ground_truth_ids),
    )


def calculate_batch_metrics_endpoint(
    request: BatchMetricsRequest,
    include_per_query: bool = False,
) -> BatchMetricsResponse:
    """
    Calculate IR metrics for multiple queries.

    Args:
        request: Batch metrics request.
        include_per_query: Whether to include per-query metrics.

    Returns:
        BatchMetricsResponse with aggregated metrics.
    """
    if not request.queries:
        # Return empty metrics
        return BatchMetricsResponse(
            query_count=0,
            average_metrics=MetricsResponse(
                precision={k: 0.0 for k in [1, 3, 5, 10]},
                recall={k: 0.0 for k in [1, 3, 5, 10]},
                ndcg={k: 0.0 for k in [1, 3, 5, 10]},
                mrr=0.0,
                retrieved_count=0,
                relevant_count=0,
            ),
        )

    # Calculate individual metrics
    per_query_results = []
    all_metrics = []

    for query in request.queries:
        metrics = calculate_metrics(
            retrieved_ids=query.retrieved_ids,
            ground_truth_ids=query.ground_truth_ids,
            k_values=query.k_values,
        )
        all_metrics.append(metrics)

        if include_per_query:
            per_query_results.append(
                MetricsResponse(
                    precision=metrics.precision,
                    recall=metrics.recall,
                    ndcg=metrics.ndcg,
                    mrr=metrics.mrr,
                    retrieved_count=len(query.retrieved_ids),
                    relevant_count=len(query.ground_truth_ids),
                )
            )

    # Calculate averages
    k_values = request.queries[0].k_values

    avg_precision = {}
    avg_recall = {}
    avg_ndcg = {}

    for k in k_values:
        avg_precision[k] = sum(m.precision.get(k, 0.0) for m in all_metrics) / len(all_metrics)
        avg_recall[k] = sum(m.recall.get(k, 0.0) for m in all_metrics) / len(all_metrics)
        avg_ndcg[k] = sum(m.ndcg.get(k, 0.0) for m in all_metrics) / len(all_metrics)

    avg_mrr = sum(m.mrr for m in all_metrics) / len(all_metrics)
    total_retrieved = sum(len(q.retrieved_ids) for q in request.queries)
    total_relevant = sum(len(q.ground_truth_ids) for q in request.queries)

    return BatchMetricsResponse(
        query_count=len(request.queries),
        average_metrics=MetricsResponse(
            precision=avg_precision,
            recall=avg_recall,
            ndcg=avg_ndcg,
            mrr=avg_mrr,
            retrieved_count=total_retrieved // len(request.queries),
            relevant_count=total_relevant // len(request.queries),
        ),
        per_query_metrics=per_query_results if include_per_query else None,
    )


# =============================================================================
# Utility
# =============================================================================


def metrics_to_dict(metrics: EvaluationMetrics) -> dict[str, Any]:
    """Convert EvaluationMetrics to dict for JSON response."""
    return {
        "precision": metrics.precision,
        "recall": metrics.recall,
        "ndcg": metrics.ndcg,
        "mrr": metrics.mrr,
    }
