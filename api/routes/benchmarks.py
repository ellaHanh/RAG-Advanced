"""
RAG-Advanced Benchmark API Routes.

POST endpoints for triggering and monitoring async benchmarks.

Routes:
    POST /benchmarks - Start a new benchmark (async)
    GET /benchmarks/{id} - Get benchmark status
    GET /benchmarks/{id}/results - Get benchmark results
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from evaluation.benchmarks import BenchmarkConfig, BenchmarkReport, BenchmarkRunner


logger = logging.getLogger(__name__)


# =============================================================================
# Status Enum
# =============================================================================


class BenchmarkStatus(str, Enum):
    """Status of a benchmark."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# =============================================================================
# Request/Response Models
# =============================================================================


class BenchmarkTriggerRequest(BaseModel):
    """
    Request model for triggering a benchmark.

    Attributes:
        strategies: List of strategies to benchmark.
        queries: List of queries to test (each with query_id, query, optional ground_truth_chunk_ids).
        iterations: Number of iterations per strategy.
        timeout_seconds: Timeout per query.
    """

    model_config = ConfigDict(
        frozen=True,
        json_schema_extra={
            "example": {
                "strategies": ["standard", "reranking"],
                "queries": [
                    {"query_id": "q1", "query": "What is RAG?"},
                    {"query_id": "q2", "query": "How does hybrid search work?"},
                ],
                "iterations": 2,
                "timeout_seconds": 30.0,
            }
        },
    )

    strategies: list[str] = Field(..., min_length=1, description="Strategies to benchmark")
    queries: list[dict[str, Any]] = Field(..., min_length=1, description="Queries to test")
    iterations: int = Field(default=3, ge=1, le=10, description="Iterations per strategy")
    timeout_seconds: float = Field(default=30.0, ge=1.0, description="Timeout per query")


class BenchmarkTriggerResponse(BaseModel):
    """
    Response model for triggered benchmark.

    Attributes:
        benchmark_id: Unique identifier for the benchmark.
        status: Current status.
        created_at: When the benchmark was created.
        estimated_duration_seconds: Estimated time to complete.
    """

    model_config = ConfigDict(frozen=True)

    benchmark_id: str = Field(..., description="Unique benchmark ID")
    status: BenchmarkStatus = Field(..., description="Current status")
    created_at: datetime = Field(..., description="Creation timestamp")
    estimated_duration_seconds: int = Field(..., description="Estimated duration")


class BenchmarkStatusResponse(BaseModel):
    """
    Response model for benchmark status.

    Attributes:
        benchmark_id: Unique identifier.
        status: Current status.
        progress: Progress percentage (0-100).
        started_at: When execution started.
        completed_at: When execution completed.
        error: Error message if failed.
    """

    model_config = ConfigDict(frozen=True)

    benchmark_id: str = Field(..., description="Benchmark ID")
    status: BenchmarkStatus = Field(..., description="Current status")
    progress: int = Field(default=0, ge=0, le=100, description="Progress %")
    started_at: datetime | None = Field(default=None, description="Start time")
    completed_at: datetime | None = Field(default=None, description="Completion time")
    error: str | None = Field(default=None, description="Error message")


# =============================================================================
# Benchmark Store (In-Memory)
# =============================================================================


class BenchmarkJob:
    """Represents a benchmark job."""

    def __init__(
        self,
        benchmark_id: str,
        config: BenchmarkConfig,
        queries: list[dict[str, Any]],
    ) -> None:
        """Initialize benchmark job."""
        self.benchmark_id = benchmark_id
        self.config = config
        self.queries = queries
        self.status = BenchmarkStatus.PENDING
        self.progress = 0
        self.created_at = datetime.now(UTC)
        self.started_at: datetime | None = None
        self.completed_at: datetime | None = None
        self.error: str | None = None
        self.result: BenchmarkReport | None = None
        self._task: asyncio.Task[Any] | None = None


class BenchmarkStore:
    """
    In-memory store for benchmark jobs.

    Note: In production, use Redis or a database.
    """

    def __init__(self) -> None:
        """Initialize the store."""
        self._jobs: dict[str, BenchmarkJob] = {}

    def create(
        self,
        config: BenchmarkConfig,
        queries: list[dict[str, Any]],
    ) -> BenchmarkJob:
        """Create a new benchmark job."""
        benchmark_id = str(uuid.uuid4())
        job = BenchmarkJob(benchmark_id, config, queries)
        self._jobs[benchmark_id] = job
        return job

    def get(self, benchmark_id: str) -> BenchmarkJob | None:
        """Get a benchmark job by ID."""
        return self._jobs.get(benchmark_id)

    def update(self, job: BenchmarkJob) -> None:
        """Update a benchmark job."""
        self._jobs[job.benchmark_id] = job

    def delete(self, benchmark_id: str) -> bool:
        """Delete a benchmark job."""
        if benchmark_id in self._jobs:
            del self._jobs[benchmark_id]
            return True
        return False


# Global store (in production, use dependency injection)
_benchmark_store = BenchmarkStore()


# =============================================================================
# Benchmark Execution
# =============================================================================


async def _run_benchmark(job: BenchmarkJob) -> None:
    """Execute a benchmark job in the background."""
    try:
        job.status = BenchmarkStatus.RUNNING
        job.started_at = datetime.now(UTC)
        _benchmark_store.update(job)

        runner = BenchmarkRunner()
        report, _ = await runner.run(job.queries, job.config)

        job.status = BenchmarkStatus.COMPLETED
        job.completed_at = datetime.now(UTC)
        job.result = report
        job.progress = 100

    except Exception as e:
        logger.exception(f"Benchmark {job.benchmark_id} failed: {e}")
        job.status = BenchmarkStatus.FAILED
        job.completed_at = datetime.now(UTC)
        job.error = str(e)

    finally:
        _benchmark_store.update(job)


# =============================================================================
# Endpoint Functions
# =============================================================================


def trigger_benchmark(request: BenchmarkTriggerRequest) -> BenchmarkTriggerResponse:
    """
    Trigger a new benchmark asynchronously.

    Args:
        request: Benchmark configuration.

    Returns:
        BenchmarkTriggerResponse with benchmark ID.
    """
    config = BenchmarkConfig(
        strategies=request.strategies,
        iterations=request.iterations,
        timeout_seconds=request.timeout_seconds,
    )

    job = _benchmark_store.create(config, request.queries)

    # Estimate duration
    query_count = len(request.queries)
    strategy_count = len(request.strategies)
    estimated_seconds = query_count * strategy_count * request.iterations * 2  # 2s per query

    # Start background task (if event loop is running)
    try:
        loop = asyncio.get_running_loop()
        job._task = loop.create_task(_run_benchmark(job))
    except RuntimeError:
        # No running loop - job will stay in PENDING state
        # In a real web framework, the request handler runs in an event loop
        pass

    return BenchmarkTriggerResponse(
        benchmark_id=job.benchmark_id,
        status=job.status,
        created_at=job.created_at,
        estimated_duration_seconds=estimated_seconds,
    )


def get_benchmark_status(benchmark_id: str) -> BenchmarkStatusResponse | None:
    """
    Get the status of a benchmark.

    Args:
        benchmark_id: Benchmark ID.

    Returns:
        BenchmarkStatusResponse or None if not found.
    """
    job = _benchmark_store.get(benchmark_id)
    if job is None:
        return None

    return BenchmarkStatusResponse(
        benchmark_id=job.benchmark_id,
        status=job.status,
        progress=job.progress,
        started_at=job.started_at,
        completed_at=job.completed_at,
        error=job.error,
    )


def get_benchmark_results(benchmark_id: str) -> BenchmarkReport | None:
    """
    Get the results of a completed benchmark.

    Args:
        benchmark_id: Benchmark ID.

    Returns:
        BenchmarkReport or None if not found/not completed.
    """
    job = _benchmark_store.get(benchmark_id)
    if job is None or job.result is None:
        return None

    return job.result


def cancel_benchmark(benchmark_id: str) -> bool:
    """
    Cancel a running benchmark.

    Args:
        benchmark_id: Benchmark ID.

    Returns:
        True if cancelled, False otherwise.
    """
    job = _benchmark_store.get(benchmark_id)
    if job is None:
        return False

    if job.status == BenchmarkStatus.RUNNING and job._task:
        job._task.cancel()
        job.status = BenchmarkStatus.CANCELLED
        job.completed_at = datetime.now(UTC)
        _benchmark_store.update(job)
        return True

    return False


# =============================================================================
# Store Access
# =============================================================================


def get_benchmark_store() -> BenchmarkStore:
    """Get the global benchmark store."""
    return _benchmark_store


def reset_benchmark_store() -> None:
    """Reset the global benchmark store (for testing)."""
    global _benchmark_store
    _benchmark_store = BenchmarkStore()
