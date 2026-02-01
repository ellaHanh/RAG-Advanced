# Orchestration Layer

The `orchestration/` folder implements strategy registration, single and chained execution, parallel comparison, cost tracking, and resource management. It is the core layer between the API and the strategy implementations (`strategies/agents/`).

---

## Contents

| File | Purpose |
|------|--------|
| **registry.py** | `StrategyRegistry`: register and lookup strategies by name. `get_registry()`, `get_strategy()`, `get_strategy_metadata()`, `list_strategies()`, `list_strategy_names()`. Strategies are registered at app startup via `strategies.agents.register_all_strategies()`. |
| **models.py** | Pydantic and dataclass models: `StrategyConfig`, `StrategyMetadata`, `StrategyType`, `ResourceType`, `Document`, `ExecutionResult`, `TokenCounts`, `ChainStep`, `ChainConfig`, `ChainContext`, and related. |
| **executor.py** | `StrategyExecutor`: execute a single strategy with timeout, cost tracking, and metrics. `ExecutionContext` is passed to strategy functions. `ExecutorConfig` for timeout, retries, caching. |
| **chain_executor.py** | `ChainExecutor`: run a list of `ChainStep`s sequentially with the same query; optional fallback per step and `continue_on_error`. Produces `ChainResult` with per-step results and final documents. `execute_chain()` convenience function. |
| **comparison.py** | `ParallelExecutor`, `ComparisonAggregator`: run multiple strategies in parallel and aggregate results into rankings (by latency, cost, document count, etc.). Used by `POST /compare`. |
| **cost_tracker.py** | `CostTracker`: thread-safe aggregation of embedding and LLM costs. Used during strategy execution to record token usage and compute cost. |
| **pricing.py** | `PricingProvider`, `ModelPricing`: versioned pricing (e.g. from `config/pricing.json`) for embedding and chat models. `get_pricing_provider()` for app-wide access. |
| **resource_manager.py** | `ResourceManager`: semaphore-based concurrency limits per resource type (e.g. database, embedding API). Used by the executor to avoid overload. |
| **errors.py** | Custom exceptions: `RAGAdvancedError`, `StrategyNotFoundError`, `StrategyExecutionError`, `ChainExecutionError`, `ChainConfigurationError`, `StrategyTimeoutError`, and others. |

---

## Data flow

1. **Registration**: At startup, `register_all_strategies(pool, embed_query_fn)` registers each strategy (standard, reranking, multi_query, etc.) with the global registry.
2. **Single run**: API `POST /execute` → `StrategyExecutor().execute(strategy_name, query, config)` → registry returns the strategy function → `ExecutionContext` is built → strategy runs → `ExecutionResult` (documents, cost, latency) returned.
3. **Chain**: API `POST /chain` → `ChainExecutor().execute_chain(steps, query)` → for each step, executor runs the strategy with the same query → step config (limit, initial_k, etc.) is passed through → final result is the last step’s documents plus aggregated cost/latency.
4. **Compare**: API `POST /compare` → parallel execution of each strategy → `ComparisonAggregator` ranks by chosen criteria and returns best strategy and per-strategy results.

---

## Usage (programmatic)

```python
from orchestration import (
    get_registry,
    StrategyExecutor,
    StrategyConfig,
    execute_strategy,
    ChainExecutor,
    ChainStep,
    execute_chain,
)

# Single strategy
result = await execute_strategy("reranking", "What is RAG?", config=StrategyConfig(limit=5, initial_k=20, final_k=5))

# Chain
chain = [ChainStep(strategy="multi_query"), ChainStep(strategy="reranking")]
chain_result = await execute_chain(chain, "What is RAG?")
```

See `orchestration/__init__.py` for the full list of exported symbols.
