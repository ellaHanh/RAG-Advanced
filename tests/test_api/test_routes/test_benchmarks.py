"""
Unit tests for Benchmark API Routes.

Tests cover:
- Benchmark triggering
- Status retrieval
- Request/response models
"""

from __future__ import annotations

import pytest

from api.routes.benchmarks import (
    BenchmarkStatus,
    BenchmarkStatusResponse,
    BenchmarkTriggerRequest,
    BenchmarkTriggerResponse,
    get_benchmark_status,
    reset_benchmark_store,
    trigger_benchmark,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def reset_store():
    """Reset benchmark store before each test."""
    reset_benchmark_store()
    yield
    reset_benchmark_store()


# =============================================================================
# Test: Request Models
# =============================================================================


class TestRequestModels:
    """Tests for request models."""

    def test_trigger_request_creation(self):
        """Test BenchmarkTriggerRequest creation."""
        request = BenchmarkTriggerRequest(
            strategies=["standard", "reranking"],
            queries=[{"text": "test query"}],
            iterations=3,
        )

        assert len(request.strategies) == 2
        assert request.iterations == 3

    def test_trigger_request_defaults(self):
        """Test default values."""
        request = BenchmarkTriggerRequest(
            strategies=["standard"],
            queries=[{"text": "test"}],
        )

        assert request.iterations == 3
        assert request.timeout_seconds == 30.0


# =============================================================================
# Test: Response Models
# =============================================================================


class TestResponseModels:
    """Tests for response models."""

    def test_trigger_response_creation(self):
        """Test BenchmarkTriggerResponse creation."""
        from datetime import datetime, UTC

        response = BenchmarkTriggerResponse(
            benchmark_id="test-123",
            status=BenchmarkStatus.PENDING,
            created_at=datetime.now(UTC),
            estimated_duration_seconds=60,
        )

        assert response.benchmark_id == "test-123"
        assert response.status == BenchmarkStatus.PENDING

    def test_status_response_creation(self):
        """Test BenchmarkStatusResponse creation."""
        response = BenchmarkStatusResponse(
            benchmark_id="test-123",
            status=BenchmarkStatus.RUNNING,
            progress=50,
        )

        assert response.progress == 50


# =============================================================================
# Test: Trigger Endpoint
# =============================================================================


class TestTriggerEndpoint:
    """Tests for trigger endpoint."""

    def test_trigger_benchmark_creates_job(self):
        """Test triggering creates a benchmark job."""
        request = BenchmarkTriggerRequest(
            strategies=["standard"],
            queries=[{"text": "test query", "ground_truth": ["doc1"]}],
        )

        response = trigger_benchmark(request)

        assert response.benchmark_id is not None
        assert response.status == BenchmarkStatus.PENDING

    def test_trigger_returns_unique_id(self):
        """Test each trigger returns unique ID."""
        request = BenchmarkTriggerRequest(
            strategies=["standard"],
            queries=[{"text": "test"}],
        )

        response1 = trigger_benchmark(request)
        response2 = trigger_benchmark(request)

        assert response1.benchmark_id != response2.benchmark_id

    def test_trigger_estimates_duration(self):
        """Test duration estimation."""
        request = BenchmarkTriggerRequest(
            strategies=["a", "b"],
            queries=[{"text": "1"}, {"text": "2"}],
            iterations=2,
        )

        response = trigger_benchmark(request)

        # 2 queries * 2 strategies * 2 iterations * 2 seconds = 16
        assert response.estimated_duration_seconds > 0


# =============================================================================
# Test: Status Endpoint
# =============================================================================


class TestStatusEndpoint:
    """Tests for status endpoint."""

    def test_get_status_after_trigger(self):
        """Test getting status after triggering."""
        request = BenchmarkTriggerRequest(
            strategies=["standard"],
            queries=[{"text": "test"}],
        )

        trigger_response = trigger_benchmark(request)
        status = get_benchmark_status(trigger_response.benchmark_id)

        assert status is not None
        assert status.benchmark_id == trigger_response.benchmark_id

    def test_get_status_nonexistent(self):
        """Test getting status of nonexistent benchmark."""
        status = get_benchmark_status("nonexistent-id")

        assert status is None


# =============================================================================
# Test: BenchmarkStatus Enum
# =============================================================================


class TestBenchmarkStatusEnum:
    """Tests for BenchmarkStatus enum."""

    def test_status_values(self):
        """Test status enum values."""
        assert BenchmarkStatus.PENDING == "pending"
        assert BenchmarkStatus.RUNNING == "running"
        assert BenchmarkStatus.COMPLETED == "completed"
        assert BenchmarkStatus.FAILED == "failed"
