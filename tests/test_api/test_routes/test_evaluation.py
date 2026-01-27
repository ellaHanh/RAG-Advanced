"""
Unit tests for Evaluation API Routes.

Tests cover:
- Single query metrics calculation
- Batch metrics calculation
- Request/response models
- Edge cases
"""

from __future__ import annotations

import pytest

from api.routes.evaluation import (
    BatchMetricsRequest,
    BatchMetricsResponse,
    MetricsRequest,
    MetricsResponse,
    calculate_batch_metrics_endpoint,
    calculate_metrics_endpoint,
)


# =============================================================================
# Test: Request Models
# =============================================================================


class TestRequestModels:
    """Tests for request models."""

    def test_metrics_request_creation(self):
        """Test MetricsRequest creation."""
        request = MetricsRequest(
            retrieved_ids=["a", "b", "c"],
            ground_truth_ids=["a", "d"],
        )

        assert len(request.retrieved_ids) == 3
        assert len(request.ground_truth_ids) == 2

    def test_metrics_request_default_k_values(self):
        """Test default k values."""
        request = MetricsRequest(
            retrieved_ids=["a"],
            ground_truth_ids=["a"],
        )

        assert request.k_values == [1, 3, 5, 10]

    def test_metrics_request_custom_k_values(self):
        """Test custom k values."""
        request = MetricsRequest(
            retrieved_ids=["a"],
            ground_truth_ids=["a"],
            k_values=[1, 5, 20],
        )

        assert request.k_values == [1, 5, 20]

    def test_batch_metrics_request(self):
        """Test BatchMetricsRequest creation."""
        queries = [
            MetricsRequest(retrieved_ids=["a"], ground_truth_ids=["a"]),
            MetricsRequest(retrieved_ids=["b"], ground_truth_ids=["c"]),
        ]

        request = BatchMetricsRequest(queries=queries)

        assert len(request.queries) == 2


# =============================================================================
# Test: Response Models
# =============================================================================


class TestResponseModels:
    """Tests for response models."""

    def test_metrics_response_creation(self):
        """Test MetricsResponse creation."""
        response = MetricsResponse(
            precision={1: 1.0, 3: 0.67},
            recall={1: 0.5, 3: 0.5},
            ndcg={1: 1.0, 3: 0.8},
            mrr=1.0,
            retrieved_count=5,
            relevant_count=2,
        )

        assert response.mrr == 1.0
        assert response.precision[1] == 1.0


# =============================================================================
# Test: Metrics Calculation Endpoint
# =============================================================================


class TestMetricsEndpoint:
    """Tests for metrics calculation endpoint."""

    def test_calculate_metrics_basic(self):
        """Test basic metrics calculation."""
        request = MetricsRequest(
            retrieved_ids=["a", "b", "c"],
            ground_truth_ids=["a", "c"],
        )

        response = calculate_metrics_endpoint(request)

        assert response.mrr > 0  # First result is relevant
        assert response.retrieved_count == 3
        assert response.relevant_count == 2

    def test_calculate_metrics_perfect_results(self):
        """Test metrics with perfect results."""
        request = MetricsRequest(
            retrieved_ids=["a", "b"],
            ground_truth_ids=["a", "b"],
            k_values=[1, 2],
        )

        response = calculate_metrics_endpoint(request)

        assert response.precision[2] == 1.0
        assert response.recall[2] == 1.0
        assert response.mrr == 1.0

    def test_calculate_metrics_no_relevant(self):
        """Test metrics when no results are relevant."""
        request = MetricsRequest(
            retrieved_ids=["a", "b", "c"],
            ground_truth_ids=["x", "y", "z"],
        )

        response = calculate_metrics_endpoint(request)

        assert response.precision[1] == 0.0
        assert response.mrr == 0.0

    def test_calculate_metrics_empty_retrieved(self):
        """Test metrics with empty retrieved list."""
        request = MetricsRequest(
            retrieved_ids=[],
            ground_truth_ids=["a", "b"],
        )

        response = calculate_metrics_endpoint(request)

        assert response.precision[1] == 0.0
        assert response.recall[1] == 0.0
        assert response.mrr == 0.0


# =============================================================================
# Test: Batch Metrics Endpoint
# =============================================================================


class TestBatchMetricsEndpoint:
    """Tests for batch metrics endpoint."""

    def test_calculate_batch_metrics(self):
        """Test batch metrics calculation."""
        request = BatchMetricsRequest(
            queries=[
                MetricsRequest(
                    retrieved_ids=["a", "b"],
                    ground_truth_ids=["a"],
                ),
                MetricsRequest(
                    retrieved_ids=["c", "d"],
                    ground_truth_ids=["c", "d"],
                ),
            ]
        )

        response = calculate_batch_metrics_endpoint(request)

        assert response.query_count == 2
        assert response.average_metrics.mrr > 0

    def test_batch_metrics_with_per_query(self):
        """Test batch metrics with per-query results."""
        request = BatchMetricsRequest(
            queries=[
                MetricsRequest(
                    retrieved_ids=["a"],
                    ground_truth_ids=["a"],
                ),
                MetricsRequest(
                    retrieved_ids=["b"],
                    ground_truth_ids=["b"],
                ),
            ]
        )

        response = calculate_batch_metrics_endpoint(request, include_per_query=True)

        assert response.per_query_metrics is not None
        assert len(response.per_query_metrics) == 2

    def test_batch_metrics_empty_queries(self):
        """Test batch metrics with empty queries."""
        request = BatchMetricsRequest(queries=[])

        response = calculate_batch_metrics_endpoint(request)

        assert response.query_count == 0
        assert response.average_metrics.mrr == 0.0

    def test_batch_metrics_averages_correctly(self):
        """Test that batch metrics averages correctly."""
        request = BatchMetricsRequest(
            queries=[
                MetricsRequest(
                    retrieved_ids=["a"],
                    ground_truth_ids=["a"],
                    k_values=[1],
                ),
                MetricsRequest(
                    retrieved_ids=["x"],  # Wrong result
                    ground_truth_ids=["a"],
                    k_values=[1],
                ),
            ]
        )

        response = calculate_batch_metrics_endpoint(request)

        # One perfect (1.0) + one zero (0.0) = average 0.5
        assert response.average_metrics.precision[1] == 0.5
