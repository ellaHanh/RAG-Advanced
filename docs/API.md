# API Reference

> **Note**: This documentation will be fully populated when the API module is implemented.

## Overview

RAG-Advanced exposes a REST API via FastAPI with the following endpoint groups:

- **Strategy Endpoints** (`/strategies/`) - Execute, chain, and compare RAG strategies
- **Evaluation Endpoints** (`/evaluation/`) - Calculate IR metrics and run benchmarks
- **Health Endpoints** (`/health/`) - Service health and readiness checks

## Authentication

All endpoints (except health) require API key authentication via the `X-API-Key` header.

```bash
curl -H "X-API-Key: your-api-key" http://localhost:8000/strategies/execute
```

## Rate Limiting

Rate limits are enforced per API key using Redis sliding window counters.

| Tier | Requests/minute | Burst |
|------|----------------|-------|
| Free | 10 | 5 |
| Standard | 100 | 20 |
| Enterprise | 1000 | 100 |

## Endpoints

### Strategy Execution

#### POST /strategies/execute

Execute a single RAG strategy.

**Request:**
```json
{
  "strategy": "reranking",
  "query": "What is machine learning?",
  "config": {
    "initial_k": 20,
    "final_k": 5
  }
}
```

**Response:**
```json
{
  "documents": [...],
  "cost_usd": 0.0012,
  "latency_ms": 250,
  "token_counts": {
    "embedding": 10,
    "llm_input": 0,
    "llm_output": 0
  }
}
```

#### POST /strategies/chain

Execute multiple strategies sequentially.

#### POST /strategies/compare

Execute multiple strategies in parallel for comparison.

### Evaluation

#### POST /evaluation/metrics

Calculate IR metrics for retrieved documents.

#### POST /evaluation/metrics/batch

Calculate metrics for multiple queries.

#### POST /evaluation/benchmark

Start an async benchmark run.

#### GET /evaluation/benchmark/{id}

Get benchmark status and results.

## OpenAPI Documentation

Interactive API documentation is available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
