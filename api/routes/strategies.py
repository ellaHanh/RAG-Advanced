"""
RAG-Advanced Strategy API Routes.

REST endpoints for strategy execution, chaining, and comparison.

Routes:
    POST /execute - Execute a single strategy
    POST /chain - Execute strategy chain
    POST /compare - Compare multiple strategies
    GET /strategies - List available strategies
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from orchestration.chain_executor import ChainExecutor, ChainResult
from orchestration.comparison import ComparisonAggregator, ComparisonResult
from orchestration.executor import ParallelExecutor, StrategyExecutor
from orchestration.models import ChainStep, Document, ExecutionResult, StrategyConfig
from orchestration.registry import get_registry


logger = logging.getLogger(__name__)


# =============================================================================
# Request/Response Models
# =============================================================================


class ExecuteRequest(BaseModel):
    """
    Request model for single strategy execution.

    Attributes:
        strategy: Name of the strategy to execute.
        query: Search query.
        limit: Maximum results to return.
        initial_k: Candidates for reranking (reranking strategy; default 20).
        final_k: Results after reranking (reranking strategy; default = limit).
        timeout_seconds: Execution timeout.
    """

    model_config = ConfigDict(frozen=True)

    strategy: str = Field(..., description="Strategy name")
    query: str = Field(..., min_length=1, description="Search query")
    limit: int = Field(default=5, ge=1, le=50, description="Max results")
    initial_k: int | None = Field(default=None, ge=1, le=100, description="Reranking: candidate count")
    final_k: int | None = Field(default=None, ge=1, le=50, description="Reranking: results after rerank")
    timeout_seconds: float = Field(default=30.0, ge=1.0, description="Timeout")


class DocumentResponse(BaseModel):
    """Response model for a document."""

    model_config = ConfigDict(frozen=True)

    id: str
    content: str
    title: str | None = None
    source: str | None = None
    similarity: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExecuteResponse(BaseModel):
    """
    Response model for strategy execution.

    Attributes:
        documents: Retrieved documents.
        query: Executed query.
        strategy_name: Name of executed strategy.
        latency_ms: Execution time in milliseconds.
        cost_usd: Execution cost.
    """

    model_config = ConfigDict(frozen=True)

    documents: list[DocumentResponse] = Field(..., description="Retrieved documents")
    query: str = Field(..., description="Executed query")
    strategy_name: str = Field(..., description="Strategy name")
    latency_ms: int = Field(..., description="Latency in ms")
    cost_usd: float = Field(..., description="Cost in USD")


class ChainRequest(BaseModel):
    """
    Request model for chain execution.

    Attributes:
        steps: List of chain steps.
        query: Search query.
        continue_on_error: Whether to continue on step failure.
    """

    model_config = ConfigDict(frozen=True)

    steps: list[dict[str, Any]] = Field(..., min_length=1, description="Chain steps")
    query: str = Field(..., min_length=1, description="Search query")
    continue_on_error: bool = Field(default=False, description="Continue on error")


class ChainStepResponse(BaseModel):
    """Response model for a chain step."""

    model_config = ConfigDict(frozen=True)

    step_name: str
    strategy_name: str
    document_count: int
    duration_ms: int


class ChainResponse(BaseModel):
    """
    Response model for chain execution.

    Attributes:
        query: Executed query.
        success: Whether chain completed successfully.
        steps: Results from each step.
        total_latency_ms: Total execution time.
        total_cost_usd: Total cost.
        documents: Final documents from last step.
    """

    model_config = ConfigDict(frozen=True)

    query: str = Field(..., description="Executed query")
    success: bool = Field(..., description="Success status")
    steps: list[ChainStepResponse] = Field(..., description="Step results")
    total_latency_ms: int = Field(..., description="Total latency")
    total_cost_usd: float = Field(..., description="Total cost")
    documents: list[DocumentResponse] = Field(..., description="Final documents")
    error: str | None = Field(default=None, description="Error message")


class CompareRequest(BaseModel):
    """
    Request model for strategy comparison.

    Attributes:
        strategies: List of strategies to compare.
        query: Search query.
        timeout_seconds: Timeout per strategy.
    """

    model_config = ConfigDict(frozen=True)

    strategies: list[str] = Field(..., min_length=2, description="Strategies to compare")
    query: str = Field(..., min_length=1, description="Search query")
    timeout_seconds: float = Field(default=30.0, ge=1.0, description="Timeout")


class StrategyRankingResponse(BaseModel):
    """Response model for strategy ranking."""

    model_config = ConfigDict(frozen=True)

    strategy_name: str
    rank: int
    score: float


class CompareResponse(BaseModel):
    """
    Response model for strategy comparison.

    Attributes:
        query: Executed query.
        best_overall: Name of best strategy.
        rankings: Rankings by different criteria.
        results: Results from each strategy.
    """

    model_config = ConfigDict(frozen=True)

    query: str = Field(..., description="Executed query")
    best_overall: str | None = Field(..., description="Best strategy")
    rankings: dict[str, list[StrategyRankingResponse]] = Field(..., description="Rankings")
    results: dict[str, ExecuteResponse] = Field(..., description="Per-strategy results")
    total_cost_usd: float = Field(..., description="Total cost")


class StrategyInfoResponse(BaseModel):
    """Response model for strategy information."""

    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    strategy_type: str
    version: str


class ListStrategiesResponse(BaseModel):
    """Response model for listing strategies."""

    model_config = ConfigDict(frozen=True)

    strategies: list[StrategyInfoResponse]
    total_count: int


# =============================================================================
# Helper Functions
# =============================================================================


def _document_to_response(doc: Document) -> DocumentResponse:
    """Convert Document to DocumentResponse."""
    return DocumentResponse(
        id=doc.id,
        content=doc.content,
        title=doc.title,
        source=doc.source,
        similarity=doc.similarity,
        metadata=doc.metadata,
    )


def _result_to_response(result: ExecutionResult) -> ExecuteResponse:
    """Convert ExecutionResult to ExecuteResponse."""
    return ExecuteResponse(
        documents=[_document_to_response(d) for d in result.documents],
        query=result.query,
        strategy_name=result.strategy_name,
        latency_ms=result.latency_ms,
        cost_usd=result.cost_usd,
    )


# =============================================================================
# Endpoint Functions
# =============================================================================


async def execute_strategy_endpoint(request: ExecuteRequest) -> ExecuteResponse:
    """
    Execute a single strategy.

    Args:
        request: Execution request.

    Returns:
        ExecuteResponse with results.
    """
    executor = StrategyExecutor()
    config = StrategyConfig(
        limit=request.limit,
        initial_k=request.initial_k if request.initial_k is not None else 20,
        final_k=request.final_k if request.final_k is not None else request.limit,
    )

    result = await executor.execute(
        strategy_name=request.strategy,
        query=request.query,
        config=config,
        timeout=request.timeout_seconds,
    )

    return _result_to_response(result)


async def execute_chain_endpoint(request: ChainRequest) -> ChainResponse:
    """
    Execute a strategy chain.

    Args:
        request: Chain request.

    Returns:
        ChainResponse with results.
    """
    # Convert dict steps to ChainStep objects
    steps = []
    for step_data in request.steps:
        config = None
        if "config" in step_data and isinstance(step_data["config"], dict):
            config = StrategyConfig(**step_data["config"])
        step = ChainStep(
            strategy=step_data["strategy"],
            config=config or StrategyConfig(),
            fallback_strategy=step_data.get("fallback_strategy"),
            continue_on_error=step_data.get("continue_on_error", False),
        )
        steps.append(step)

    executor = ChainExecutor()
    result = await executor.execute_chain(steps, request.query)

    return ChainResponse(
        query=result.query,
        success=result.success,
        steps=[
            ChainStepResponse(
                step_name=s.step_name,
                strategy_name=s.strategy_name,
                document_count=s.output_document_count,
                duration_ms=s.duration_ms,
            )
            for s in result.steps
        ],
        total_latency_ms=result.total_latency_ms,
        total_cost_usd=result.total_cost_usd,
        documents=[_document_to_response(d) for d in result.final_documents],
        error=result.error,
    )


async def compare_strategies_endpoint(request: CompareRequest) -> CompareResponse:
    """
    Compare multiple strategies.

    Args:
        request: Comparison request.

    Returns:
        CompareResponse with results.
    """
    executor = ParallelExecutor()
    parallel_result = await executor.execute_all(
        strategies=request.strategies,
        query=request.query,
        timeout=request.timeout_seconds,
    )

    aggregator = ComparisonAggregator()
    comparison = aggregator.aggregate(parallel_result, query=request.query)

    # Convert rankings
    rankings = {}
    for criteria, ranking_list in comparison.rankings.items():
        rankings[criteria.value] = [
            StrategyRankingResponse(
                strategy_name=r.strategy_name,
                rank=r.rank,
                score=r.score,
            )
            for r in ranking_list
        ]

    # Convert results
    results = {}
    for name, result in parallel_result.results.items():
        results[name] = _result_to_response(result)

    return CompareResponse(
        query=request.query,
        best_overall=comparison.best_overall,
        rankings=rankings,
        results=results,
        total_cost_usd=comparison.total_cost,
    )


def list_strategies_endpoint() -> ListStrategiesResponse:
    """
    List all available strategies.

    Returns:
        ListStrategiesResponse with strategy information.
    """
    registry = get_registry()
    strategies = registry.list_strategies()

    return ListStrategiesResponse(
        strategies=[
            StrategyInfoResponse(
                name=s.name,
                description=s.description,
                strategy_type=s.strategy_type.value,
                version=s.version,
            )
            for s in strategies
        ],
        total_count=len(strategies),
    )
