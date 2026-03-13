# Tests

Pytest-based unit and integration tests for API, orchestration, evaluation, and strategies.

## Quickstart

```bash
# From repo root, venv activated
pytest tests/ -v

# Exclude integration and slow (CI-friendly)
pytest tests/ -v -m "not integration and not slow"

# With integration (needs DATABASE_URL, pgvector, schema)
pytest tests/ -v -m integration
```

## Features

- **test_api/** — Routes, auth, rate limiter.
- **test_orchestration/** — Registry, executor, chain, comparison, cost, resource manager.
- **test_evaluation/** — Metrics, benchmarks, datasets, reports, xlsx loaders, pipeline helpers/integration.
- **test_strategies/** — Ingestion; test_utils: embedder, caches.
- **test_scripts/** — run_schema (EMBEDDING_BACKEND selection).

Integration tests are marked `@pytest.mark.integration`; need `TEST_DATABASE_URL` or `DATABASE_URL`, pgvector, schema (1536 or 1024) aligned with `EMBEDDING_BACKEND`.

## Usage

```bash
pytest tests/test_orchestration/ -v
pytest tests/test_evaluation/test_ragas_eval.py -v
pytest tests/ --cov=. --cov-report=html
```

DB for integration: set `DATABASE_URL`, run Postgres (e.g. `docker-compose up -d postgres`), apply schema (`python scripts/run_schema.py`), match `EMBEDDING_BACKEND` to schema dimension.

## Dependencies

Project: `pip install -e .`. Integration tests require a running PostgreSQL with pgvector and schema.

## Related

- [Root README](../README.md)
- [evaluation/](../evaluation/README.md)
- [scripts/](../scripts/README.md) — run_schema
