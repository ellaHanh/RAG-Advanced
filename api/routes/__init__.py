"""
API Route Modules.

Contains endpoint handlers for:
    - strategies: Strategy execution, chaining, comparison
    - evaluation: IR metrics calculation, batch evaluation
    - benchmarks: Async benchmark triggers and status

Exports:
    - MetricsRequest: Request model for metrics calculation
    - MetricsResponse: Response model for metrics
    - BatchMetricsRequest: Request model for batch metrics
    - BatchMetricsResponse: Response model for batch metrics
    - calculate_metrics_endpoint: Calculate metrics for single query
    - calculate_batch_metrics_endpoint: Calculate metrics for multiple queries
"""

from api.routes.evaluation import (
    BatchMetricsRequest,
    BatchMetricsResponse,
    MetricsRequest,
    MetricsResponse,
    calculate_batch_metrics_endpoint,
    calculate_metrics_endpoint,
)

from api.routes.benchmarks import (
    BenchmarkStatus,
    BenchmarkTriggerRequest,
    BenchmarkTriggerResponse,
    BenchmarkStatusResponse,
    trigger_benchmark,
    get_benchmark_status,
    get_benchmark_results,
    cancel_benchmark,
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

__all__ = [
    # Evaluation
    "BatchMetricsRequest",
    "BatchMetricsResponse",
    "MetricsRequest",
    "MetricsResponse",
    "calculate_batch_metrics_endpoint",
    "calculate_metrics_endpoint",
    # Benchmarks
    "BenchmarkStatus",
    "BenchmarkTriggerRequest",
    "BenchmarkTriggerResponse",
    "BenchmarkStatusResponse",
    "trigger_benchmark",
    "get_benchmark_status",
    "get_benchmark_results",
    "cancel_benchmark",
    # Strategies
    "ChainRequest",
    "ChainResponse",
    "CompareRequest",
    "CompareResponse",
    "ExecuteRequest",
    "ExecuteResponse",
    "ListStrategiesResponse",
    "compare_strategies_endpoint",
    "execute_chain_endpoint",
    "execute_strategy_endpoint",
    "list_strategies_endpoint",
]
