# API

FastAPI REST API for strategies, evaluation, benchmarks, and health.

## Quickstart

```bash
# Start services (from repo root)
docker-compose up -d
# Or: docker-compose up -d postgres redis && uvicorn api.main:app --reload

curl http://localhost:8000/health
curl -H "X-API-Key: your-api-key" http://localhost:8000/strategies
```

Docs: **http://localhost:8000/docs** (Swagger), **http://localhost:8000/redoc**, **http://localhost:8000/openapi.json**.

## Features

| Group | Endpoints |
|-------|-----------|
| **Strategies** | `GET /strategies`, `POST /execute`, `POST /chain`, `POST /compare`, `POST /generate` |
| **Evaluation** | `POST /metrics`, `POST /metrics/batch` |
| **Benchmarks** | `POST /benchmarks`, `GET /benchmarks/{id}`, `GET /benchmarks/{id}/results`, `DELETE /benchmarks/{id}` |
| **RAGAS** | `POST /evaluate/generation` |
| **Health** | `GET /health`, `GET /` |

- **Auth:** All except health/root require `X-API-Key`.
- **Rate limiting:** Per API key (Redis); see `api/rate_limiter.py`.

## Usage

```bash
# Example: execute (all endpoints require X-API-Key except /health)
curl -X POST http://localhost:8000/execute -H "Content-Type: application/json" -H "X-API-Key: your-key" \
  -d '{"strategy": "reranking", "query": "What is RAG?", "config": {"limit": 5}}'
```

Full curl examples (execute, chain, compare, metrics, benchmarks, RAGAS): [root README](../README.md).

## Dependencies

Project: `pip install -e .`. Env: `DATABASE_URL`, `REDIS_URL`, `OPENAI_API_KEY`; optional `ANTHROPIC_API_KEY` for contextual retrieval.

## Related

- [Root README](../README.md)
- [orchestration/](../orchestration/README.md) — Execution layer
- [evaluation/](../evaluation/README.md) — Metrics and RAGAS
