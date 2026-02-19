# API Reference

RAG-Advanced exposes a REST API via FastAPI for strategy execution, chaining, comparison, evaluation, and health checks.

---

## Overview

| Group | Endpoints |
|-------|-----------|
| **Strategies** | `GET /strategies`, `POST /execute`, `POST /chain`, `POST /compare`, `POST /generate` (retrieval + generation) |
| **Evaluation** | `POST /metrics`, `POST /metrics/batch` |
| **Benchmarks** | `POST /benchmarks`, `GET /benchmarks/{benchmark_id}`, `GET /benchmarks/{benchmark_id}/results`, `DELETE /benchmarks/{benchmark_id}` |
| **Health** | `GET /health`, `GET /` |

Interactive documentation: **Swagger UI** at `http://localhost:8000/docs`, **ReDoc** at `http://localhost:8000/redoc`, **OpenAPI JSON** at `http://localhost:8000/openapi.json`.

---

## RAG pipeline: retrieval + generation

The pipeline has three stages: **ingestion** (indexing documents), **retrieval** (strategy execution returning documents), and **generation** (turning query + documents into a natural-language answer). The API supports retrieval-only via `POST /execute` and `POST /chain`, and full RAG via **`POST /generate`**, which runs retrieval then a LangChain-based “stuff documents” chain (prompt template + LLM) to produce an answer. See [Codecademy: Build RAG pipelines](https://www.codecademy.com/article/build-rag-pipelines-in-ai-applications) and [ApX: Combining Retrieval and Generation](https://apxml.com/courses/getting-started-rag/chapter-5-building-basic-rag-pipeline/combining-retrieval-generation) for the augmented-prompt pattern, empty-context handling, and error-handling practices used here.

---

## Authentication

All endpoints except health and root require API key authentication via the `X-API-Key` header.

```bash
curl -H "X-API-Key: your-api-key" http://localhost:8000/strategies
```

---

## Rate limiting

Rate limits are enforced per API key using Redis sliding-window counters. Configure tiers and limits in the API (see `api/rate_limiter.py`). If Redis is not configured, rate limiting may be disabled.

---

## Strategy endpoints

### GET /strategies

List all registered RAG strategies with metadata (name, description, strategy_type, required_resources, estimated_latency_ms, estimated_cost_per_query, tags).

**Response:** `ListStrategiesResponse` with `strategies: list[StrategyInfoResponse]`.

---

### POST /execute

Execute a single RAG strategy.

**Request body:**

```json
{
  "strategy": "reranking",
  "query": "What is machine learning?",
  "limit": 5,
  "initial_k": 20,
  "final_k": 5
}
```

- `strategy` (required): One of `standard`, `reranking`, `multi_query`, `query_expansion`, `self_reflective`, `agentic`, `contextual_retrieval`, `context_aware_chunking`.
- `query` (required): Search query text.
- `limit`: Max results (default 5). Used by all strategies.
- `initial_k`, `final_k`: Used by `reranking` (candidates and final count).
- `num_variations`: Used by `multi_query`, `query_expansion`.

**Response:** `ExecuteResponse` with `documents`, `strategy_name`, `latency_ms`, `cost_usd`, `token_counts`, etc.

---

### POST /chain

Execute a chain of strategies sequentially. **Each step’s output (documents) is passed as the next step’s input.** The chain returns the last step’s documents (e.g. retrieval then reranked subset). Sensible sequences: **standard → reranking**, **multi_query → reranking**, **query_expansion → reranking** (first step retrieves; second step reranks those documents without a second retrieval).

**Request body:**

```json
{
  "query": "How does RAG work?",
  "steps": [
    { "strategy": "multi_query", "config": { "limit": 10, "num_variations": 3 } },
    { "strategy": "reranking", "config": { "limit": 5, "initial_k": 20, "final_k": 5 } }
  ],
  "continue_on_error": false
}
```

- `steps`: List of objects with `strategy` and optional `config` (limit, initial_k, final_k, num_variations), optional `fallback_strategy`, `continue_on_error`. When the previous step produces documents, strategies that support it (e.g. reranking) use them as input instead of retrieving again.

**Response:** `ChainResponse` with `query`, `success`, `steps`, `total_latency_ms`, `total_cost_usd`, `documents`, `error`. Use `include_step_documents: true` to include each step’s documents in `steps[]`.

---

### POST /compare

Execute multiple strategies in parallel and compare results.

**Request body:**

```json
{
  "strategies": ["standard", "reranking", "multi_query"],
  "query": "What is retrieval-augmented generation?",
  "timeout_seconds": 30
}
```

**Response:** `CompareResponse` with `query`, `best_overall`, `rankings`, `results` (per-strategy `ExecuteResponse`), `total_cost_usd`.

---

## RAG generate (retrieval + generation)

### POST /generate

Run retrieval with a single strategy, then generate a natural-language answer from the retrieved documents using a LangChain stuff-documents chain (prompt template with context + question, then LLM). Returns the **answer** plus documents (sources), model, token counts, and cost. On **empty retrieval**: by default returns a fixed user-facing message; if `no_context_fallback` is true, calls the LLM with no context (fallback to standard LLM behavior). Retrieval and generation errors are separated: retrieval failures return a clear message and 502/503; generation failures return a distinct message and 502.

**Request body:**

```json
{
  "query": "What is retrieval-augmented generation?",
  "strategy": "standard",
  "limit": 5,
  "model": "gpt-4o-mini",
  "prompt_template": null,
  "no_context_fallback": false,
  "timeout_seconds": 30,
  "initial_k": null,
  "final_k": null
}
```

- `query` (required): User question.
- `strategy`: Retrieval strategy name (default `standard`).
- `limit`: Max documents to retrieve (default 5).
- `model`: Optional override for generation model (default from env `GENERATION_MODEL` or `gpt-4o-mini`).
- `prompt_template`: Optional override for prompt (must include `{context}` and `{input}`).
- `no_context_fallback`: If true and retrieval returns no documents, call LLM with no context instead of returning a fixed message.
- `timeout_seconds`: Timeout for the retrieval step.
- `initial_k`, `final_k`: Used by `reranking` when chosen as the retrieval strategy.

**Response:** `GenerateResponse` with `answer`, `documents` (sources), `model`, `input_tokens`, `output_tokens`, `cost_usd`, `retrieval_latency_ms`, `generation_latency_ms`, `context_truncated` (true if context was truncated to fit the model window).

**Environment:** `OPENAI_API_KEY` (required for generation). Optional: `GENERATION_MODEL`, `GENERATION_PROMPT_TEMPLATE`.

---

## Evaluation endpoints

### POST /metrics

Calculate IR metrics for a single query.

**Request body:**

```json
{
  "retrieved_ids": ["chunk-1", "chunk-2", "chunk-3"],
  "ground_truth_ids": ["chunk-1", "chunk-3"],
  "k_values": [1, 3, 5, 10]
}
```

**Response:** `MetricsResponse` with `precision`, `recall`, `ndcg` (per k), `mrr`, `retrieved_count`, `relevant_count`.

---

### POST /metrics/batch

Calculate IR metrics for multiple queries and aggregate.

**Request body:** `BatchMetricsRequest` with `queries` (list of objects like above). Use query param `?include_per_query=true` to include per-query metrics.

**Response:** `BatchMetricsResponse` with `query_count`, `average_metrics`, optional `per_query_metrics`.

---

## Benchmark endpoints

### POST /benchmarks

Start an async benchmark. Returns immediately with `benchmark_id`.

**Request body:**

```json
{
  "strategies": ["standard", "reranking", "multi_query"],
  "queries": [
    {
      "query_id": "q1",
      "query": "What is RAG?",
      "ground_truth_ids": ["doc-1", "doc-2"]
    }
  ],
  "iterations": 3,
  "timeout_seconds": 30
}
```

**Response:** `BenchmarkTriggerResponse` with `benchmark_id`, `status`, `created_at`, `estimated_duration_seconds`.

---

### GET /benchmarks/{benchmark_id}

Get benchmark status.

**Response:** `BenchmarkStatusResponse` with `benchmark_id`, `status` (pending/running/completed/failed/cancelled), `progress`, `started_at`, `completed_at`, `error`.

---

### GET /benchmarks/{benchmark_id}/results

Get full benchmark results (when status is `completed`). Returns a dict with strategy statistics, latency, cost, rankings.

---

### DELETE /benchmarks/{benchmark_id}

Cancel a running benchmark.

**Response:** `{"status": "cancelled", "benchmark_id": "..."}`.

---

## Health endpoints

### GET /health

Health check: status (healthy/degraded/unhealthy), version, uptime_seconds, timestamp, component status (pricing, rate_limiter, database). No authentication required.

### GET /

Root: API name, version, description, links to docs, health, openapi. No authentication required.
