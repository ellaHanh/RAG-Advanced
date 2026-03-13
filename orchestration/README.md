# Orchestration Layer

Registry, single and chained execution, parallel comparison, cost tracking, resource management. Sits between the API and `strategies/agents/`.

## Quickstart

```bash
# Via API (see api/README.md). Programmatic:
# pip install -e . && set DATABASE_URL, etc.
```

```python
from orchestration import execute_strategy, StrategyConfig, execute_chain, ChainStep

result = await execute_strategy("reranking", "What is RAG?", config=StrategyConfig(limit=5, initial_k=20, final_k=5))
chain_result = await execute_chain(
    [ChainStep(strategy="multi_query"), ChainStep(strategy="reranking")],
    "What is RAG?",
)
```

## Features

- **Registry** — `StrategyRegistry`; register/lookup by name; `get_registry()`, `get_strategy()`, `list_strategies()`.
- **Single execution** — `StrategyExecutor`; timeout, cost tracking, metrics; `ExecutionContext` → `ExecutionResult`.
- **Chaining** — `ChainExecutor`; each step runs retrieval from scratch (same query); chain returns the last step’s documents; optional fallback, `continue_on_error`.
- **Comparison** — `ParallelExecutor`, `ComparisonAggregator`; rank by latency, cost, etc.
- **Cost** — `CostTracker`, `PricingProvider` (from `config/pricing.json`).
- **Resources** — `ResourceManager` (semaphores).

Default config: `limit=5`, `initial_k`/`final_k` (reranking), `num_variations` (multi_query/query_expansion), `timeout_seconds=30`, `use_cache=true`.

## Usage

Request bodies (for `/execute`, `/chain`, `/compare`, `/benchmarks`) — see [api/README.md](../api/README.md). Chaining: each step runs **retrieval from scratch** (same query); the chain returns the **last step’s documents**.

| File | Purpose |
|------|--------|
| `registry.py` | Strategy registration and lookup |
| `executor.py` | Single strategy execution |
| `chain_executor.py` | Sequential chain |
| `comparison.py` | Parallel comparison |
| `cost_tracker.py`, `pricing.py` | Cost and pricing |
| `resource_manager.py` | Concurrency limits |
| `models.py` | StrategyConfig, Document, ExecutionResult, ChainStep, etc. |
| `errors.py` | Custom exceptions |

## Dependencies

See root README. Orchestration uses project deps and `config/pricing.json`.

## Related

- [Root README](../README.md)
- [api/](../api/README.md) — REST routes
- [strategies/agents/](../strategies/agents/) — Strategy implementations
