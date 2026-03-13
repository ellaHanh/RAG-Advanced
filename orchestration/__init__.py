"""
RAG-Advanced Orchestration Module.

Strategy execution, chaining, and comparison with cost tracking.

Exports:
    Models:
        - ChainContext: Immutable state passed between strategy steps
        - ChainConfig: Configuration for strategy chains
        - ChainStep: Single step in a chain
        - StrategyConfig: Configuration for strategy execution
        - StrategyMetadata: Metadata about a registered strategy
        - ExecutionResult: Result from strategy execution
        - Document: Retrieved document representation
        - TokenCounts: Token usage tracking
        - StrategyType: Enum of strategy types
        - ResourceType: Enum of resource types

    Registry:
        - StrategyRegistry: Central registry for strategies
        - register_strategy: Decorator for strategy registration
        - get_registry: Get global registry instance
        - get_strategy: Get strategy from global registry
        - get_strategy_metadata: Get strategy metadata
        - list_strategies: List all registered strategies
        - list_strategy_names: List all strategy names

    Pricing:
        - PricingProvider: Async pricing provider with historical support
        - ModelPricing: Pricing for a single model
        - get_pricing_provider: Get global pricing provider

    Cost Tracking:
        - CostTracker: Thread-safe cost aggregation
        - CostSummary: Summary of tracked costs
        - CostEntry: Single cost entry
        - calculate_cost: Calculate cost for API call

    Execution:
        - StrategyExecutor: Execute single strategies with metrics
        - ExecutorConfig: Executor configuration
        - ExecutionContext: Context passed to strategies
        - execute_strategy: Convenience function for execution
        - execute_strategy_sync: Synchronous execution
        - ParallelExecutor: Execute strategies concurrently
        - ParallelExecutionResult: Aggregated parallel results
        - execute_strategies_parallel: Parallel execution convenience

    Resource Management:
        - ResourceManager: Semaphore-based resource limiting
        - ResourceManagerConfig: Resource limit configuration
        - ResourceLimitConfig: Per-resource limits
        - get_resource_manager: Get global resource manager

    Comparison:
        - ComparisonAggregator: Aggregate parallel results
        - ComparisonResult: Aggregated comparison data
        - RankingCriteria: Criteria for ranking strategies
        - compare_results: Convenience function

    Chain Execution:
        - ChainExecutor: Sequential strategy chain execution
        - ChainResult: Result of chain execution
        - ChainStepResult: Result of single chain step
        - execute_chain: Convenience function

    Errors:
        - RAGAdvancedError: Base exception
        - StrategyNotFoundError: Strategy not in registry
        - StrategyAlreadyRegisteredError: Duplicate registration
        - InvalidStrategyError: Invalid strategy function
        - StrategyExecutionError: Strategy execution failed
        - ChainExecutionError: Chain execution failed
        - PricingConfigError: Pricing configuration error
"""

__version__ = "0.1.0"

# Models
from orchestration.models import (
    ChainConfig,
    ChainContext,
    ChainStep,
    Document,
    ExecutionResult,
    ResourceType,
    StrategyConfig,
    StrategyMetadata,
    StrategyType,
    TokenCounts,
)

# Registry
from orchestration.registry import (
    StrategyRegistry,
    get_registry,
    get_strategy,
    get_strategy_metadata,
    list_strategies,
    list_strategy_names,
    register_strategy,
)

# Pricing
from orchestration.pricing import (
    ModelPricing,
    PricingConfig,
    PricingPeriod,
    PricingProvider,
    get_pricing_provider,
    get_pricing_provider_sync,
)

# Cost Tracking
from orchestration.cost_tracker import (
    CostEntry,
    CostSummary,
    CostTracker,
    calculate_cost,
    estimate_embedding_cost,
    estimate_llm_cost,
)

# Execution
from orchestration.executor import (
    ExecutionContext,
    ExecutorConfig,
    ParallelExecutionResult,
    ParallelExecutor,
    StrategyExecutor,
    execute_strategies_parallel,
    execute_strategy,
    execute_strategy_sync,
)

# Resource Management
from orchestration.resource_manager import (
    ResourceAcquisitionTimeout,
    ResourceLimitConfig,
    ResourceManager,
    ResourceManagerConfig,
    get_resource_manager,
    reset_resource_manager,
)

# Comparison
from orchestration.comparison import (
    AggregatorConfig,
    ComparisonAggregator,
    ComparisonResult,
    RankingCriteria,
    StrategyMetrics,
    StrategyRanking,
    compare_results,
)

# Chain Execution
from orchestration.chain_executor import (
    ChainExecutor,
    ChainExecutorConfig,
    ChainResult,
    ChainStepResult,
    execute_chain,
)

# Errors
from orchestration.errors import (
    ChainConfigurationError,
    ChainError,
    ChainExecutionError,
    ConfigurationError,
    ExecutionError,
    InvalidStrategyError,
    PricingConfigError,
    RAGAdvancedError,
    RegistryError,
    StrategyAlreadyRegisteredError,
    StrategyExecutionError,
    StrategyNotFoundError,
    StrategyTimeoutError,
)

__all__ = [
    # Version
    "__version__",
    # Models
    "ChainConfig",
    "ChainContext",
    "ChainStep",
    "Document",
    "ExecutionResult",
    "ResourceType",
    "StrategyConfig",
    "StrategyMetadata",
    "StrategyType",
    "TokenCounts",
    # Registry
    "StrategyRegistry",
    "get_registry",
    "get_strategy",
    "get_strategy_metadata",
    "list_strategies",
    "list_strategy_names",
    "register_strategy",
    # Pricing
    "ModelPricing",
    "PricingConfig",
    "PricingPeriod",
    "PricingProvider",
    "get_pricing_provider",
    "get_pricing_provider_sync",
    # Cost Tracking
    "CostEntry",
    "CostSummary",
    "CostTracker",
    "calculate_cost",
    "estimate_embedding_cost",
    "estimate_llm_cost",
    # Execution
    "ExecutionContext",
    "ExecutorConfig",
    "ParallelExecutionResult",
    "ParallelExecutor",
    "StrategyExecutor",
    "execute_strategies_parallel",
    "execute_strategy",
    "execute_strategy_sync",
    # Resource Management
    "ResourceAcquisitionTimeout",
    "ResourceLimitConfig",
    "ResourceManager",
    "ResourceManagerConfig",
    "get_resource_manager",
    "reset_resource_manager",
    # Comparison
    "AggregatorConfig",
    "ComparisonAggregator",
    "ComparisonResult",
    "RankingCriteria",
    "StrategyMetrics",
    "StrategyRanking",
    "compare_results",
    # Chain Execution
    "ChainExecutor",
    "ChainExecutorConfig",
    "ChainResult",
    "ChainStepResult",
    "execute_chain",
    # Errors
    "ChainConfigurationError",
    "ChainError",
    "ChainExecutionError",
    "ConfigurationError",
    "ExecutionError",
    "InvalidStrategyError",
    "PricingConfigError",
    "RAGAdvancedError",
    "RegistryError",
    "StrategyAlreadyRegisteredError",
    "StrategyExecutionError",
    "StrategyNotFoundError",
    "StrategyTimeoutError",
]
