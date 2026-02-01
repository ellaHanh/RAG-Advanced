# Tests

The `tests/` folder contains pytest-based unit and integration tests for the API, orchestration, evaluation, and strategies.

---

## Layout

| Directory / file | Purpose |
|------------------|--------|
| **conftest.py** | Shared pytest fixtures (e.g. registry, executor, mock pool, sample strategies). |
| **test_api/** | API route and middleware tests. |
| **test_api/test_routes/** | `test_strategies.py`, `test_evaluation.py`, `test_benchmarks.py` — request/response and endpoint behavior. |
| **test_api/test_auth.py** | API key authentication. |
| **test_api/test_rate_limiter.py** | Rate limiting logic. |
| **test_orchestration/** | Registry, executor, chain executor, comparison, cost tracking, resource manager, chain context. |
| **test_evaluation/** | Metrics, benchmarks, datasets, ground_truth_llm, reports, html_reports. |
| **test_strategies/** | Ingestion pipeline tests; **test_strategies/test_utils/** — embedding cache, result cache. |

---

## Running tests

From the repo root (with project deps installed, e.g. `pip install -e .`):

```bash
# All tests
pytest tests/ -v

# By area
pytest tests/test_api/ -v
pytest tests/test_orchestration/ -v
pytest tests/test_evaluation/ -v
pytest tests/test_strategies/ -v

# Single file
pytest tests/test_orchestration/test_executor.py -v

# With coverage
pytest tests/ --cov=. --cov-report=html
```

Tests that require a live database or external APIs may be marked (e.g. `@pytest.mark.integration`) and skipped by default; run them explicitly if you have the environment set up.
