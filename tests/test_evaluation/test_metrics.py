"""
Unit tests for IR Metrics Calculation.

Tests cover:
- Precision@k calculation
- Recall@k calculation
- MRR calculation
- NDCG@k calculation
- Edge cases (empty lists, k > results, etc.)
- Input validation
- Batch metrics calculation
"""

from __future__ import annotations

import pytest

from evaluation.metrics import (
    EvaluationMetrics,
    calculate_batch_metrics,
    calculate_metrics,
)
from orchestration.errors import InvalidInputError


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def perfect_retrieval():
    """Perfect retrieval: all retrieved docs are relevant."""
    return {
        "retrieved_ids": ["doc1", "doc2", "doc3"],
        "ground_truth_ids": ["doc1", "doc2", "doc3"],
    }


@pytest.fixture
def partial_retrieval():
    """Partial retrieval: some retrieved docs are relevant."""
    return {
        "retrieved_ids": ["doc1", "doc3", "doc5", "doc2", "doc6"],
        "ground_truth_ids": ["doc1", "doc2", "doc3", "doc4"],
    }


@pytest.fixture
def no_relevant_retrieval():
    """No relevant docs retrieved."""
    return {
        "retrieved_ids": ["doc5", "doc6", "doc7"],
        "ground_truth_ids": ["doc1", "doc2", "doc3"],
    }


@pytest.fixture
def graded_relevance():
    """Retrieval with graded relevance scores."""
    return {
        "retrieved_ids": ["doc1", "doc3", "doc5", "doc2"],
        "ground_truth_ids": ["doc1", "doc2", "doc3"],
        "relevance_scores": {
            "doc1": 2,  # Highly relevant
            "doc2": 1,  # Partially relevant
            "doc3": 2,  # Highly relevant
        },
    }


# =============================================================================
# Test: Precision@k
# =============================================================================


class TestPrecisionAtK:
    """Tests for Precision@k calculation."""

    def test_perfect_precision(self, perfect_retrieval):
        """Test precision when all retrieved docs are relevant."""
        metrics = calculate_metrics(
            **perfect_retrieval,
            k_values=[3],
        )

        assert metrics.precision[3] == pytest.approx(1.0)

    def test_zero_precision(self, no_relevant_retrieval):
        """Test precision when no retrieved docs are relevant."""
        metrics = calculate_metrics(
            **no_relevant_retrieval,
            k_values=[3],
        )

        assert metrics.precision[3] == pytest.approx(0.0)

    def test_partial_precision(self, partial_retrieval):
        """Test precision with partial relevance."""
        metrics = calculate_metrics(
            **partial_retrieval,
            k_values=[3, 5],
        )

        # At k=3: 2 relevant (doc1, doc3) out of 3
        assert metrics.precision[3] == pytest.approx(2 / 3)
        # At k=5: 3 relevant (doc1, doc3, doc2) out of 5
        assert metrics.precision[5] == pytest.approx(3 / 5)

    def test_precision_k_greater_than_retrieved(self):
        """Test precision when k > number of retrieved docs."""
        metrics = calculate_metrics(
            retrieved_ids=["doc1", "doc2"],
            ground_truth_ids=["doc1"],
            k_values=[5],
        )

        # Only 2 docs, so at k=5 we still only have 2 docs
        # 1 relevant out of 2, but k=5 so precision = 1/5
        assert metrics.precision[5] == pytest.approx(1 / 5)


# =============================================================================
# Test: Recall@k
# =============================================================================


class TestRecallAtK:
    """Tests for Recall@k calculation."""

    def test_perfect_recall(self, perfect_retrieval):
        """Test recall when all relevant docs are retrieved."""
        metrics = calculate_metrics(
            **perfect_retrieval,
            k_values=[3],
        )

        assert metrics.recall[3] == pytest.approx(1.0)

    def test_zero_recall(self, no_relevant_retrieval):
        """Test recall when no relevant docs are retrieved."""
        metrics = calculate_metrics(
            **no_relevant_retrieval,
            k_values=[3],
        )

        assert metrics.recall[3] == pytest.approx(0.0)

    def test_partial_recall(self, partial_retrieval):
        """Test recall with partial retrieval."""
        metrics = calculate_metrics(
            **partial_retrieval,
            k_values=[3, 5],
        )

        # At k=3: 2 relevant retrieved (doc1, doc3), 4 total relevant
        assert metrics.recall[3] == pytest.approx(2 / 4)
        # At k=5: 3 relevant retrieved (doc1, doc3, doc2), 4 total relevant
        assert metrics.recall[5] == pytest.approx(3 / 4)

    def test_recall_with_single_relevant(self):
        """Test recall with single relevant document."""
        metrics = calculate_metrics(
            retrieved_ids=["doc2", "doc1", "doc3"],
            ground_truth_ids=["doc1"],
            k_values=[1, 2, 3],
        )

        assert metrics.recall[1] == pytest.approx(0.0)  # doc1 not in first position
        assert metrics.recall[2] == pytest.approx(1.0)  # doc1 in top 2
        assert metrics.recall[3] == pytest.approx(1.0)


# =============================================================================
# Test: MRR
# =============================================================================


class TestMRR:
    """Tests for Mean Reciprocal Rank calculation."""

    def test_mrr_first_position(self):
        """Test MRR when first relevant doc is at position 1."""
        metrics = calculate_metrics(
            retrieved_ids=["doc1", "doc2", "doc3"],
            ground_truth_ids=["doc1"],
            k_values=[3],
        )

        assert metrics.mrr == pytest.approx(1.0)

    def test_mrr_second_position(self):
        """Test MRR when first relevant doc is at position 2."""
        metrics = calculate_metrics(
            retrieved_ids=["doc2", "doc1", "doc3"],
            ground_truth_ids=["doc1"],
            k_values=[3],
        )

        assert metrics.mrr == pytest.approx(0.5)

    def test_mrr_third_position(self):
        """Test MRR when first relevant doc is at position 3."""
        metrics = calculate_metrics(
            retrieved_ids=["doc4", "doc5", "doc1"],
            ground_truth_ids=["doc1"],
            k_values=[3],
        )

        assert metrics.mrr == pytest.approx(1 / 3)

    def test_mrr_no_relevant(self, no_relevant_retrieval):
        """Test MRR when no relevant docs are retrieved."""
        metrics = calculate_metrics(
            **no_relevant_retrieval,
            k_values=[3],
        )

        assert metrics.mrr == pytest.approx(0.0)

    def test_mrr_multiple_relevant(self):
        """Test MRR considers only first relevant doc."""
        metrics = calculate_metrics(
            retrieved_ids=["doc3", "doc1", "doc2"],
            ground_truth_ids=["doc1", "doc2"],  # Both relevant
            k_values=[3],
        )

        # First relevant is doc1 at position 2
        assert metrics.mrr == pytest.approx(0.5)


# =============================================================================
# Test: NDCG@k
# =============================================================================


class TestNDCGAtK:
    """Tests for NDCG@k calculation."""

    def test_ndcg_perfect_ranking(self, graded_relevance):
        """Test NDCG with perfect ranking."""
        # If we ranked by relevance: doc1(2), doc3(2), doc2(1) would be ideal
        # Our ranking: doc1(2), doc3(2), doc5(0), doc2(1)
        metrics = calculate_metrics(
            **graded_relevance,
            k_values=[3],
        )

        # At k=3, we have doc1(2), doc3(2), doc5(0)
        # DCG = (2^2-1)/log2(2) + (2^2-1)/log2(3) + (2^0-1)/log2(4)
        #     = 3/1 + 3/1.585 + 0/2 = 3 + 1.893 + 0 = 4.893
        # IDCG = (2^2-1)/log2(2) + (2^2-1)/log2(3) + (2^1-1)/log2(4)
        #      = 3/1 + 3/1.585 + 1/2 = 3 + 1.893 + 0.5 = 5.393
        # NDCG = 4.893 / 5.393 ≈ 0.907
        assert 0.9 <= metrics.ndcg[3] <= 1.0

    def test_ndcg_binary_relevance(self):
        """Test NDCG with binary (0/1) relevance."""
        metrics = calculate_metrics(
            retrieved_ids=["doc1", "doc2", "doc3"],
            ground_truth_ids=["doc1", "doc3"],  # Binary: doc1 and doc3 are relevant
            k_values=[3],
        )

        # With binary relevance (all relevant = 1)
        # Our ranking: doc1(1), doc2(0), doc3(1)
        # DCG = (2^1-1)/log2(2) + 0 + (2^1-1)/log2(4) = 1 + 0 + 0.5 = 1.5
        # IDCG (ideal: doc1, doc3, doc2): = 1 + 1/1.585 + 0 = 1.631
        # NDCG ≈ 1.5 / 1.631 ≈ 0.92
        assert 0.85 <= metrics.ndcg[3] <= 1.0

    def test_ndcg_zero_for_no_relevant(self, no_relevant_retrieval):
        """Test NDCG is 0 when no relevant docs retrieved."""
        metrics = calculate_metrics(
            **no_relevant_retrieval,
            k_values=[3],
        )

        assert metrics.ndcg[3] == pytest.approx(0.0)


# =============================================================================
# Test: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_retrieved(self):
        """Test with empty retrieved list."""
        metrics = calculate_metrics(
            retrieved_ids=[],
            ground_truth_ids=["doc1", "doc2"],
            k_values=[3],
        )

        assert metrics.precision[3] == 0.0
        assert metrics.recall[3] == 0.0
        assert metrics.mrr == 0.0
        assert metrics.has_warnings
        assert "No retrieved documents" in metrics.warnings[0]

    def test_empty_ground_truth(self):
        """Test with empty ground truth list."""
        metrics = calculate_metrics(
            retrieved_ids=["doc1", "doc2"],
            ground_truth_ids=[],
            k_values=[3],
        )

        assert metrics.recall[3] == 0.0
        assert metrics.has_warnings
        assert "No ground truth" in metrics.warnings[0]

    def test_both_empty(self):
        """Test with both lists empty."""
        metrics = calculate_metrics(
            retrieved_ids=[],
            ground_truth_ids=[],
            k_values=[3],
        )

        assert metrics.precision[3] == 0.0
        assert metrics.recall[3] == 0.0
        assert metrics.mrr == 0.0

    def test_duplicate_retrieved_warning(self):
        """Test warning for duplicate documents."""
        metrics = calculate_metrics(
            retrieved_ids=["doc1", "doc1", "doc2"],
            ground_truth_ids=["doc1"],
            k_values=[3],
        )

        assert metrics.has_warnings
        assert any("Duplicate" in w for w in metrics.warnings)

    def test_k_greater_than_retrieved(self):
        """Test k value greater than retrieved count."""
        metrics = calculate_metrics(
            retrieved_ids=["doc1"],
            ground_truth_ids=["doc1"],
            k_values=[10],
        )

        assert metrics.has_warnings
        assert any("exceeds" in w for w in metrics.warnings)

    def test_multiple_k_values(self, partial_retrieval):
        """Test with multiple k values."""
        metrics = calculate_metrics(
            **partial_retrieval,
            k_values=[1, 3, 5, 10],
        )

        assert len(metrics.precision) == 4
        assert len(metrics.recall) == 4
        assert len(metrics.ndcg) == 4
        assert metrics.k_values == [1, 3, 5, 10]

    def test_default_k_values(self, partial_retrieval):
        """Test default k values are used."""
        metrics = calculate_metrics(**partial_retrieval)

        assert metrics.k_values == [3, 5, 10]


# =============================================================================
# Test: Input Validation
# =============================================================================


class TestInputValidation:
    """Tests for input validation."""

    def test_none_retrieved_ids(self):
        """Test error for None retrieved_ids."""
        with pytest.raises(InvalidInputError) as exc_info:
            calculate_metrics(
                retrieved_ids=None,  # type: ignore
                ground_truth_ids=["doc1"],
            )

        assert "retrieved_ids" in str(exc_info.value)

    def test_none_ground_truth_ids(self):
        """Test error for None ground_truth_ids."""
        with pytest.raises(InvalidInputError) as exc_info:
            calculate_metrics(
                retrieved_ids=["doc1"],
                ground_truth_ids=None,  # type: ignore
            )

        assert "ground_truth_ids" in str(exc_info.value)

    def test_non_list_retrieved_ids(self):
        """Test error for non-list retrieved_ids."""
        with pytest.raises(InvalidInputError) as exc_info:
            calculate_metrics(
                retrieved_ids="doc1",  # type: ignore
                ground_truth_ids=["doc1"],
            )

        assert "Must be a list" in str(exc_info.value)

    def test_non_string_elements(self):
        """Test error for non-string elements."""
        with pytest.raises(InvalidInputError) as exc_info:
            calculate_metrics(
                retrieved_ids=[1, 2, 3],  # type: ignore
                ground_truth_ids=["doc1"],
            )

        assert "must be a string" in str(exc_info.value)

    def test_empty_k_values(self):
        """Test error for empty k_values."""
        with pytest.raises(InvalidInputError) as exc_info:
            calculate_metrics(
                retrieved_ids=["doc1"],
                ground_truth_ids=["doc1"],
                k_values=[],
            )

        assert "k_values" in str(exc_info.value)

    def test_invalid_k_values(self):
        """Test error for invalid k values."""
        with pytest.raises(InvalidInputError) as exc_info:
            calculate_metrics(
                retrieved_ids=["doc1"],
                ground_truth_ids=["doc1"],
                k_values=[0, -1],
            )

        assert "positive integer" in str(exc_info.value)

    def test_invalid_relevance_scores(self):
        """Test error for invalid relevance scores."""
        with pytest.raises(InvalidInputError) as exc_info:
            calculate_metrics(
                retrieved_ids=["doc1"],
                ground_truth_ids=["doc1"],
                relevance_scores={"doc1": 5},  # Invalid score
            )

        assert "0, 1, or 2" in str(exc_info.value)


# =============================================================================
# Test: EvaluationMetrics Model
# =============================================================================


class TestEvaluationMetrics:
    """Tests for EvaluationMetrics model."""

    def test_to_dict(self, partial_retrieval):
        """Test to_dict method."""
        metrics = calculate_metrics(**partial_retrieval)
        d = metrics.to_dict()

        assert "precision" in d
        assert "recall" in d
        assert "mrr" in d
        assert "ndcg" in d
        assert "k_values" in d
        assert "warnings" in d

    def test_has_warnings_true(self):
        """Test has_warnings property when warnings exist."""
        metrics = calculate_metrics(
            retrieved_ids=[],
            ground_truth_ids=["doc1"],
        )

        assert metrics.has_warnings is True

    def test_has_warnings_false(self, perfect_retrieval):
        """Test has_warnings property when no warnings."""
        metrics = calculate_metrics(**perfect_retrieval, k_values=[3])

        # Perfect retrieval with matching k should have no warnings
        assert metrics.has_warnings is False

    def test_frozen_metrics(self, perfect_retrieval):
        """Test that metrics are frozen/immutable."""
        metrics = calculate_metrics(**perfect_retrieval)

        with pytest.raises(Exception):  # Pydantic raises ValidationError
            metrics.mrr = 0.5  # type: ignore


# =============================================================================
# Test: Batch Metrics
# =============================================================================


class TestBatchMetrics:
    """Tests for batch metrics calculation."""

    def test_batch_metrics_basic(self):
        """Test basic batch metrics calculation."""
        queries = [
            {
                "query_id": "q1",
                "retrieved_ids": ["a", "b", "c"],
                "ground_truth_ids": ["a", "c"],
            },
            {
                "query_id": "q2",
                "retrieved_ids": ["d", "e", "f"],
                "ground_truth_ids": ["d"],
            },
        ]

        results = calculate_batch_metrics(queries, k_values=[3])

        assert len(results["per_query"]) == 2
        assert results["aggregate"]["query_count"] == 2
        assert results["aggregate"]["successful_count"] == 2
        assert 3 in results["aggregate"]["avg_precision"]

    def test_batch_metrics_with_errors(self):
        """Test batch metrics handles errors gracefully."""
        queries = [
            {
                "query_id": "q1",
                "retrieved_ids": ["a", "b"],
                "ground_truth_ids": ["a"],
            },
            {
                "query_id": "q2",
                "retrieved_ids": None,  # Invalid
                "ground_truth_ids": ["a"],
            },
        ]

        results = calculate_batch_metrics(queries, k_values=[3])

        assert results["aggregate"]["successful_count"] == 1
        assert "error" in results["per_query"][1]

    def test_batch_metrics_averaging(self):
        """Test that batch metrics averages correctly."""
        queries = [
            {
                "retrieved_ids": ["a"],
                "ground_truth_ids": ["a"],  # Precision=1
            },
            {
                "retrieved_ids": ["b"],
                "ground_truth_ids": ["a"],  # Precision=0
            },
        ]

        results = calculate_batch_metrics(queries, k_values=[1])

        # Average precision should be 0.5
        assert results["aggregate"]["avg_precision"][1] == pytest.approx(0.5)
