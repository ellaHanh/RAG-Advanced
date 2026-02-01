"""
RAG-Advanced FastAPI Application.

Main application entrypoint with route registration, lifespan management,
health checks, and OpenAPI documentation.

Usage:
    # Via CLI (pyproject.toml script)
    rag-advanced

    # Via uvicorn directly
    uvicorn api.main:app --reload

    # Programmatic
    from api.main import app, run
    run()  # Starts server
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from api.auth import ApiKeyAuth, VerificationResult
from api.rate_limiter import RateLimitConfig, RateLimiter, RateLimitResult
from api.routes.benchmarks import (
    BenchmarkStatus,
    BenchmarkStatusResponse,
    BenchmarkTriggerRequest,
    BenchmarkTriggerResponse,
    cancel_benchmark,
    get_benchmark_results,
    get_benchmark_status,
    trigger_benchmark,
)
from api.routes.evaluation import (
    BatchMetricsRequest,
    BatchMetricsResponse,
    MetricsRequest,
    MetricsResponse,
    calculate_batch_metrics_endpoint,
    calculate_metrics_endpoint,
)
from api.routes.strategies import (
    ChainRequest,
    ChainResponse,
    CompareRequest,
    CompareResponse,
    ExecuteRequest,
    ExecuteResponse,
    ListStrategiesResponse,
    compare_strategies_endpoint,
    execute_chain_endpoint,
    execute_strategy_endpoint,
    list_strategies_endpoint,
)
from orchestration import PricingProvider, get_pricing_provider
from orchestration.errors import (
    RAGAdvancedError,
    StrategyExecutionError,
    StrategyNotFoundError,
)
from strategies.agents import register_all_strategies
from strategies.utils.embedder import embed_query as embed_query_fn


# =============================================================================
# Logging Configuration
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Application State
# =============================================================================


class AppState:
    """Application state container."""

    pricing_provider: PricingProvider | None = None
    rate_limiter: RateLimiter | None = None
    api_key_auth: ApiKeyAuth | None = None
    startup_time: datetime | None = None
    db_pool: Any = None  # Will be asyncpg pool when database is configured


app_state = AppState()


# =============================================================================
# Health Check Models
# =============================================================================


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str = Field(..., description="Health status")
    version: str = Field(..., description="Application version")
    uptime_seconds: int = Field(..., description="Uptime in seconds")
    timestamp: datetime = Field(..., description="Current timestamp")
    components: dict[str, str] = Field(
        default_factory=dict, description="Component health status"
    )


class ErrorResponse(BaseModel):
    """Standard error response model."""

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: dict[str, Any] = Field(default_factory=dict, description="Error details")


# =============================================================================
# Lifespan Manager
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.

    Handles startup and shutdown tasks:
    - Startup: Initialize pricing provider, rate limiter
    - Shutdown: Close connections gracefully
    """
    # Startup
    logger.info("Starting RAG-Advanced API...")
    app_state.startup_time = datetime.now(UTC)

    # Initialize pricing provider
    try:
        app_state.pricing_provider = await PricingProvider.create()
        logger.info("Pricing provider initialized")
    except Exception as e:
        logger.warning(f"Failed to initialize pricing provider: {e}")
        # Continue without pricing - will use defaults

    # Initialize rate limiter (if Redis is configured)
    redis_url = os.getenv("REDIS_URL")
    if redis_url:
        try:
            # Rate limiter will be initialized when first used
            app_state.rate_limiter = RateLimiter(
                redis_client=None,  # Will connect lazily
                config=RateLimitConfig(),
            )
            logger.info("Rate limiter configured (will connect on first use)")
        except Exception as e:
            logger.warning(f"Failed to configure rate limiter: {e}")

    # Initialize database pool (if DATABASE_URL is set)
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        try:
            import asyncpg
            app_state.db_pool = await asyncpg.create_pool(
                database_url,
                min_size=int(os.getenv("DATABASE_MIN_CONNECTIONS", "2")),
                max_size=int(os.getenv("DATABASE_MAX_CONNECTIONS", "10")),
                command_timeout=int(os.getenv("DATABASE_COMMAND_TIMEOUT", "60")),
            )
            logger.info("Database pool initialized")
            app_state.api_key_auth = ApiKeyAuth(db_pool=app_state.db_pool)
            logger.info("API key auth configured with database")
        except Exception as e:
            logger.warning(f"Failed to initialize database pool: {e}")
            app_state.db_pool = None
            app_state.api_key_auth = ApiKeyAuth(db_pool=None)
    else:
        app_state.api_key_auth = ApiKeyAuth(db_pool=None)

    # Always register strategies (they will error clearly if DB not configured)
    register_all_strategies(app_state.db_pool, embed_query_fn)

    logger.info("RAG-Advanced API started successfully")

    yield

    # Shutdown
    logger.info("Shutting down RAG-Advanced API...")

    # Close database pool if exists
    if app_state.db_pool:
        try:
            await app_state.db_pool.close()
            logger.info("Database pool closed")
        except Exception as e:
            logger.warning(f"Error closing database pool: {e}")

    logger.info("RAG-Advanced API shutdown complete")


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="RAG-Advanced API",
    description="""
    Strategy Orchestration and Evaluation System for Retrieval-Augmented Generation.

    ## Features

    - **Execute**: Run individual RAG strategies
    - **Chain**: Execute strategies sequentially with result passing
    - **Compare**: Run multiple strategies in parallel and compare results
    - **Evaluate**: Calculate IR metrics (Precision, Recall, MRR, NDCG)
    - **Benchmark**: Run comprehensive benchmarks across strategies

    ## Authentication

    API requests require an API key passed in the `X-API-Key` header.
    Contact the administrator to obtain an API key.

    ## Rate Limiting

    Requests are rate-limited per API key. Default limit is 60 requests per minute.
    Rate limit headers are included in all responses.
    """,
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)


# =============================================================================
# CORS Middleware
# =============================================================================

# Get allowed origins from environment, default to localhost for development
ALLOWED_ORIGINS = os.getenv(
    "CORS_ALLOWED_ORIGINS",
    "http://localhost:3000,http://localhost:8000,http://127.0.0.1:3000,http://127.0.0.1:8000",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Exception Handlers
# =============================================================================


@app.exception_handler(StrategyNotFoundError)
async def strategy_not_found_handler(
    request: Request, exc: StrategyNotFoundError
) -> JSONResponse:
    """Handle strategy not found errors."""
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content=ErrorResponse(
            error="StrategyNotFound",
            message=str(exc),
            details=exc.details,
        ).model_dump(),
    )


@app.exception_handler(StrategyExecutionError)
async def strategy_execution_handler(
    request: Request, exc: StrategyExecutionError
) -> JSONResponse:
    """Handle strategy execution errors."""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="StrategyExecutionError",
            message=str(exc),
            details=exc.details,
        ).model_dump(),
    )


@app.exception_handler(RAGAdvancedError)
async def rag_advanced_error_handler(
    request: Request, exc: RAGAdvancedError
) -> JSONResponse:
    """Handle all RAG-Advanced errors."""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error=type(exc).__name__,
            message=str(exc),
            details=exc.details,
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected exceptions."""
    logger.exception(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="InternalServerError",
            message="An unexpected error occurred",
            details={"type": type(exc).__name__},
        ).model_dump(),
    )


# =============================================================================
# Health Check Endpoints
# =============================================================================


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Health check",
    description="Check the health status of the API and its components.",
)
async def health_check() -> HealthResponse:
    """
    Health check endpoint.

    Returns the current health status of the API including:
    - Overall status (healthy/degraded/unhealthy)
    - Application version
    - Uptime in seconds
    - Component status (pricing, rate limiter, database)
    """
    components: dict[str, str] = {}

    # Check pricing provider
    if app_state.pricing_provider:
        components["pricing"] = "healthy"
    else:
        components["pricing"] = "not_initialized"

    # Check rate limiter configuration
    if app_state.rate_limiter:
        components["rate_limiter"] = "configured"
    else:
        components["rate_limiter"] = "not_configured"

    # Check database
    if app_state.db_pool:
        try:
            await app_state.db_pool.fetchval("SELECT 1")
            components["database"] = "healthy"
        except Exception:
            components["database"] = "unhealthy"
    else:
        components["database"] = "not_configured"

    # Determine overall status
    unhealthy_components = [k for k, v in components.items() if v == "unhealthy"]
    if unhealthy_components:
        overall_status = "degraded"
    else:
        overall_status = "healthy"

    # Calculate uptime
    uptime_seconds = 0
    if app_state.startup_time:
        uptime_seconds = int((datetime.now(UTC) - app_state.startup_time).total_seconds())

    return HealthResponse(
        status=overall_status,
        version="0.1.0",
        uptime_seconds=uptime_seconds,
        timestamp=datetime.now(UTC),
        components=components,
    )


@app.get(
    "/",
    tags=["Health"],
    summary="Root endpoint",
    description="API information and links.",
)
async def root() -> dict[str, Any]:
    """Root endpoint with API information."""
    return {
        "name": "RAG-Advanced API",
        "version": "0.1.0",
        "description": "Strategy Orchestration and Evaluation for RAG Systems",
        "docs": "/docs",
        "redoc": "/redoc",
        "health": "/health",
        "openapi": "/openapi.json",
    }


# =============================================================================
# Strategy Endpoints
# =============================================================================


@app.get(
    "/strategies",
    response_model=ListStrategiesResponse,
    tags=["Strategies"],
    summary="List available strategies",
    description="Get a list of all registered RAG strategies with their metadata.",
)
async def list_strategies() -> ListStrategiesResponse:
    """List all available strategies."""
    return list_strategies_endpoint()


@app.post(
    "/execute",
    response_model=ExecuteResponse,
    tags=["Strategies"],
    summary="Execute a strategy",
    description="Execute a single RAG strategy with the given query and configuration.",
)
async def execute_strategy(request: ExecuteRequest) -> ExecuteResponse:
    """Execute a single RAG strategy."""
    return await execute_strategy_endpoint(request)


@app.post(
    "/chain",
    response_model=ChainResponse,
    tags=["Strategies"],
    summary="Execute strategy chain",
    description="Execute a chain of strategies sequentially, passing results between steps.",
)
async def execute_chain(request: ChainRequest) -> ChainResponse:
    """Execute a strategy chain."""
    return await execute_chain_endpoint(request)


@app.post(
    "/compare",
    response_model=CompareResponse,
    tags=["Strategies"],
    summary="Compare strategies",
    description="Execute multiple strategies in parallel and compare their results.",
)
async def compare_strategies(request: CompareRequest) -> CompareResponse:
    """Compare multiple strategies."""
    return await compare_strategies_endpoint(request)


# =============================================================================
# Evaluation Endpoints
# =============================================================================


@app.post(
    "/metrics",
    response_model=MetricsResponse,
    tags=["Evaluation"],
    summary="Calculate IR metrics",
    description="Calculate information retrieval metrics for a single query.",
)
async def calculate_metrics(request: MetricsRequest) -> MetricsResponse:
    """Calculate IR metrics for a query."""
    return calculate_metrics_endpoint(request)


@app.post(
    "/metrics/batch",
    response_model=BatchMetricsResponse,
    tags=["Evaluation"],
    summary="Calculate batch metrics",
    description="Calculate IR metrics for multiple queries and aggregate results.",
)
async def calculate_batch_metrics(request: BatchMetricsRequest) -> BatchMetricsResponse:
    """Calculate batch IR metrics."""
    return calculate_batch_metrics_endpoint(request)


# =============================================================================
# Benchmark Endpoints
# =============================================================================


@app.post(
    "/benchmarks",
    response_model=BenchmarkTriggerResponse,
    tags=["Benchmarks"],
    summary="Start benchmark",
    description="Trigger a new benchmark run (async). Returns immediately with benchmark ID.",
)
async def start_benchmark(request: BenchmarkTriggerRequest) -> BenchmarkTriggerResponse:
    """Start a new benchmark."""
    return await trigger_benchmark(request)


@app.get(
    "/benchmarks/{benchmark_id}",
    response_model=BenchmarkStatusResponse,
    tags=["Benchmarks"],
    summary="Get benchmark status",
    description="Get the current status of a running or completed benchmark.",
)
async def benchmark_status(benchmark_id: str) -> BenchmarkStatusResponse:
    """Get benchmark status."""
    result = await get_benchmark_status(benchmark_id)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Benchmark {benchmark_id} not found",
        )
    return result


@app.get(
    "/benchmarks/{benchmark_id}/results",
    tags=["Benchmarks"],
    summary="Get benchmark results",
    description="Get the full results of a completed benchmark.",
)
async def benchmark_results(benchmark_id: str) -> dict[str, Any]:
    """Get benchmark results."""
    result = await get_benchmark_results(benchmark_id)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Benchmark {benchmark_id} not found or not completed",
        )
    return result


@app.delete(
    "/benchmarks/{benchmark_id}",
    tags=["Benchmarks"],
    summary="Cancel benchmark",
    description="Cancel a running benchmark.",
)
async def cancel_benchmark_endpoint(benchmark_id: str) -> dict[str, str]:
    """Cancel a running benchmark."""
    success = await cancel_benchmark(benchmark_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Benchmark {benchmark_id} not found or already completed",
        )
    return {"status": "cancelled", "benchmark_id": benchmark_id}


# =============================================================================
# CLI Entrypoint
# =============================================================================


def run(
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = False,
    workers: int = 1,
    log_level: str = "info",
) -> None:
    """
    Run the RAG-Advanced API server.

    This function is called by the CLI entrypoint defined in pyproject.toml:
        rag-advanced = "api.main:run"

    Args:
        host: Host to bind to.
        port: Port to bind to.
        reload: Enable auto-reload for development.
        workers: Number of worker processes.
        log_level: Logging level.
    """
    # Get configuration from environment with defaults
    host = os.getenv("API_HOST", host)
    port = int(os.getenv("API_PORT", str(port)))
    reload = os.getenv("API_RELOAD", str(reload)).lower() == "true"
    workers = int(os.getenv("API_WORKERS", str(workers)))
    log_level = os.getenv("API_LOG_LEVEL", log_level)

    logger.info(f"Starting RAG-Advanced API on {host}:{port}")
    logger.info(f"Reload: {reload}, Workers: {workers}, Log Level: {log_level}")

    uvicorn.run(
        "api.main:app",
        host=host,
        port=port,
        reload=reload,
        workers=workers if not reload else 1,  # Workers don't work with reload
        log_level=log_level,
    )


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    run()
