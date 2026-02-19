# Orchestration Layer

The `orchestration/` folder implements strategy registration, single and chained execution, parallel comparison, cost tracking, and resource management. It is the core layer between the API and the strategy implementations (`strategies/agents/`).

---

## API request body examples (POST)

Use these in Swagger UI (`/docs`) or with `curl` when calling the strategy and benchmark endpoints.

### POST /execute — single strategy

```json
{
  "strategy": "standard",
  "query": "What is retrieval-augmented generation?",
  "limit": 5,
  "timeout_seconds": 30.0
}
```

Reranking with more candidates:

```json
{
  "strategy": "reranking",
  "query": "How does hybrid search combine keyword and semantic retrieval?",
  "limit": 5,
  "initial_k": 20,
  "final_k": 5,
  "timeout_seconds": 30.0
}
```

### POST /chain — strategy chain (step-by-step)

Minimal chain (final documents only):

```json
{
  "steps": [
    { "strategy": "multi_query", "config": { "limit": 20 } },
    { "strategy": "reranking", "config": { "limit": 5, "initial_k": 20, "final_k": 5 } }
  ],
  "query": "What is RAG and how does it improve LLM answers?",
  "continue_on_error": false
}
```

Chain with **per-step documents** (set `include_step_documents: true` to see retrieval output at each step):

```json
{
  "steps": [
    { "strategy": "multi_query", "config": { "limit": 20 } },
    { "strategy": "reranking", "config": { "limit": 5, "initial_k": 20, "final_k": 5 } }
  ],
  "query": "What is RAG and how does it improve LLM answers?",
  "continue_on_error": false,
  "include_step_documents": true
}
```

Response `steps[]` will then include a `documents` array for each step (strategy name, document count, duration, and the actual retrieved documents for that step).

### POST /compare — compare strategies

```json
{
  "strategies": ["standard", "reranking", "multi_query"],
  "query": "How does hybrid search combine keyword and semantic retrieval?",
  "timeout_seconds": 30.0
}
```

### POST /benchmarks — start benchmark (async)

```json
{
  "strategies": ["standard", "reranking"],
  "queries": [
    { "query_id": "q1", "query": "What is RAG?" },
    { "query_id": "q2", "query": "How does hybrid search work?" }
  ],
  "iterations": 2,
  "timeout_seconds": 30.0
}
```

Use the returned `benchmark_id` with `GET /benchmarks/{benchmark_id}` and `GET /benchmarks/{benchmark_id}/results`.

---

## Default strategy config

The same `StrategyConfig` (in `orchestration/models.py`) is used for **every** way strategies run: single (`POST /execute`), chain (`POST /chain`), compare (`POST /compare`), and evaluation/benchmark (pipeline or `POST /benchmarks`). Defaults and which strategies use them:

| Parameter | Default | Used by |
|-----------|---------|---------|
| `limit` | 5 | All strategies: max number of final results returned. |
| `initial_k` | 20 | **reranking**: number of candidates from vector search before rerank. |
| `final_k` | 5 | **reranking**: number of results after cross-encoder rerank (often same as `limit`). |
| `num_variations` | 3 | **multi_query**, **query_expansion**: number of query variations. |
| `max_iterations` | 2 | **self_reflective**: max refinement loops (search → grade → refine → search again). |
| `relevance_threshold` | 3.0 | **self_reflective**: min relevance score (0–5) before stopping. |
| `timeout_seconds` | 30.0 | All: execution timeout. |
| `use_cache` | true | All: whether to use result caching when available. |

Overrides: in the API you pass `config` in the request body (e.g. `"config": { "limit": 10, "initial_k": 30 }`). In the pipeline, `--limit` sets `limit` for all strategies; strategy-specific params (e.g. `initial_k`, `final_k`) use the defaults unless extended by the pipeline or API.

---

## Strategy layering and chaining

**Do other strategies run “on top of” standard?**

Conceptually, **standard** is “single-query vector search.” Other strategies use the **same retrieval primitive** (vector search via `match_chunks`) but do **not** call the standard strategy as a subroutine. Each strategy implements its own flow:

- **reranking**: runs its own vector search for `initial_k` candidates, then reranks with a cross-encoder to `final_k`.
- **multi_query**: expands the query (LLM), then runs multiple vector searches (one per variation), dedupes, and returns top `limit`.
- **query_expansion**: one LLM-expanded variation, then one vector search (same idea as standard with a different query).
- **self_reflective**: vector search → grade relevance (LLM) → optionally refine query (LLM) and search again.
- **agentic**: vector search then optional full-document fetch for the top result.

So they are “on top of” the same **retrieval mechanism** (embed + vector search), not on top of the **standard strategy function** itself.

**Real chaining: output as input**

Chaining **passes each step’s output (documents) as the next step’s input**. The chain executor sets `context.input_documents` to the previous step’s documents and passes `original_query` (the user’s query) so refiners like reranking can score against it.

- **Result**: The **final returned documents** are the **last step’s result**, which may be a **refined subset** of the previous step’s output (e.g. standard retrieves 20 → reranking reranks those 20 to 5). So chain [standard, reranking] is **not** the same as running reranking alone: the first step’s documents are passed to the second, which only reranks (no second retrieval).
- **Sensible sequences**: Use a **retrieval** step then a **refiner** step:
  - **standard → reranking**: Vector search (e.g. limit=20), then rerank those 20 to final_k=5 (no second retrieval).
  - **multi_query → reranking**: Expand query, parallel retrieval, dedupe; then rerank that set to top 5.
  - **query_expansion → reranking**: One expanded query + retrieval, then rerank with original query.

**Reranking in a chain**

When reranking receives **input_documents** from the previous step, it **skips its own retrieval** (no vector search, no DB call) and only runs the cross-encoder on those documents, using **original_query** for scoring. So the second step in [standard, reranking] or [multi_query, reranking] is rerank-only.

**What does `include_step_documents: true` do?**

It **only affects the response shape**. When true, the response’s `steps[]` array includes a `documents` field per step so you can inspect each step’s output. The top-level `documents` remain the **last** step’s documents (the final pipeline result, e.g. expanded retrieval then reranked subset).

---

## Contents

| File | Purpose |
|------|--------|
| **registry.py** | `StrategyRegistry`: register and lookup strategies by name. `get_registry()`, `get_strategy()`, `get_strategy_metadata()`, `list_strategies()`, `list_strategy_names()`. Strategies are registered at app startup via `strategies.agents.register_all_strategies()`. |
| **models.py** | Pydantic and dataclass models: `StrategyConfig`, `StrategyMetadata`, `StrategyType`, `ResourceType`, `Document`, `ExecutionResult`, `TokenCounts`, `ChainStep`, `ChainConfig`, `ChainContext`, and related. |
| **executor.py** | `StrategyExecutor`: execute a single strategy with timeout, cost tracking, and metrics. `ExecutionContext` is passed to strategy functions. `ExecutorConfig` for timeout, retries, caching. |
| **chain_executor.py** | `ChainExecutor`: run a list of `ChainStep`s sequentially; each step’s output (documents) is passed as the next step’s input. Optional fallback per step and `continue_on_error`. Produces `ChainResult` with per-step results and final documents. `execute_chain()` convenience function. |
| **comparison.py** | `ParallelExecutor`, `ComparisonAggregator`: run multiple strategies in parallel and aggregate results into rankings (by latency, cost, document count, etc.). Used by `POST /compare`. |
| **cost_tracker.py** | `CostTracker`: thread-safe aggregation of embedding and LLM costs. Used during strategy execution to record token usage and compute cost. |
| **pricing.py** | `PricingProvider`, `ModelPricing`: versioned pricing (e.g. from `config/pricing.json`) for embedding and chat models. `get_pricing_provider()` for app-wide access. |
| **resource_manager.py** | `ResourceManager`: semaphore-based concurrency limits per resource type (e.g. database, embedding API). Used by the executor to avoid overload. |
| **errors.py** | Custom exceptions: `RAGAdvancedError`, `StrategyNotFoundError`, `StrategyExecutionError`, `ChainExecutionError`, `ChainConfigurationError`, `StrategyTimeoutError`, and others. |

---

## Data flow

1. **Registration**: At startup, `register_all_strategies(pool, embed_query_fn)` registers each strategy (standard, reranking, multi_query, etc.) with the global registry.
2. **Single run**: API `POST /execute` → `StrategyExecutor().execute(strategy_name, query, config)` → registry returns the strategy function → `ExecutionContext` is built → strategy runs → `ExecutionResult` (documents, cost, latency) returned.
3. **Chain**: API `POST /chain` → `ChainExecutor().execute_chain(steps, query)` → for each step, executor runs the strategy with the current query and (when present) the previous step’s documents and original_query → step config is passed through → final result is the last step’s documents plus aggregated cost/latency.
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
