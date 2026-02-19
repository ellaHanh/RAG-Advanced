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
| **test_evaluation/** | Metrics, benchmarks, datasets, ground_truth_llm, reports, html_reports, **test_loaders_xlsx** (gold/corpus xlsx), **test_corpus_ingest** (write corpus to dir, get_doc_id_to_chunk_ids; optional integration for full ingest), **test_pipeline_helpers** (_parse_map_arg, _load_pipeline_config), **test_pipeline_integration** (E2E pipeline with 1 query + 2 docs; `-m integration`). |
| **test_strategies/** | Ingestion pipeline tests; **test_strategies/test_utils/** — embedding cache, result cache, **test_embedder** (backend selection, dimensions, OpenAI/BGE-M3 mocked). |
| **test_scripts/** | **test_run_schema.py** — schema path selection by EMBEDDING_BACKEND. |

---

## Running tests

From the repo root (with project deps installed, e.g. `pip install -e .`):

```bash
# All tests
pytest tests/ -v

# Unit only (exclude integration and slow; fast, CI-friendly)
pytest tests/ -v -m "not integration and not slow"

# With integration tests (requires DATABASE_URL or TEST_DATABASE_URL, pgvector, and matching schema/embedder)
pytest tests/ -v -m integration

# By area
pytest tests/test_api/ -v
pytest tests/test_orchestration/ -v
pytest tests/test_evaluation/ -v
pytest tests/test_strategies/ -v

# Single file
pytest tests/test_orchestration/test_executor.py -v
pytest tests/test_strategies/test_utils/test_embedder.py -v

# With coverage
pytest tests/ --cov=. --cov-report=html
```

Tests that require a live database or external APIs are marked `@pytest.mark.integration` and are skipped unless run with `-m integration`. Integration tests need `TEST_DATABASE_URL` or `DATABASE_URL`, PostgreSQL with pgvector, and schema (1536 or 1024) aligned with `EMBEDDING_BACKEND` if you use the corpus ingest integration test.

---

## Configuring the DB for integration tests

**What the pytest output means**

When you run `pytest tests/ -v -m integration` you get one test selected: `test_ingest_corpus_and_get_chunk_map_with_db`. If it shows **SKIPPED** with a message like `Connect call failed ('127.0.0.1', 5432)` or `Connection refused`, that means:

- The test **did** see `DATABASE_URL` or `TEST_DATABASE_URL` (otherwise it would skip with "TEST_DATABASE_URL or DATABASE_URL not set").
- The test then tried to connect to PostgreSQL and **failed** — usually because nothing is listening on port 5432 (PostgreSQL not running). The test catches the error and skips so the run doesn’t fail.

**How to configure the DB**

1. **Environment**
   - Set **`DATABASE_URL`** (or **`TEST_DATABASE_URL`**) in `.env` or the shell. The app and scripts read `DATABASE_URL`; the integration test uses `TEST_DATABASE_URL` if set, otherwise `DATABASE_URL`.
   - Example (matches project Docker Compose):  
     `DATABASE_URL=postgresql://postgres:postgres@localhost:5432/rag_advanced`

2. **Run PostgreSQL**
   - Easiest: start only Postgres (and optionally Redis) with Docker Compose:
     ```bash
     docker-compose up -d postgres
     ```
   - Postgres will listen on `localhost:5432`. The Compose file uses image `pgvector/pgvector:pg16`, user `postgres`, password `postgres`, database `rag_advanced`, and applies `strategies/utils/schema.sql` and `evaluation/schema_extension.sql` on first start (1536-dim schema). For 1024-dim (BGE-M3), run `python scripts/run_schema.py` with `EMBEDDING_BACKEND=bge-m3` after the container is up.

3. **Apply schema if not using Docker init**
   - If you use a Postgres instance that didn’t run the Compose init scripts, apply schema once:
     ```bash
     python scripts/run_schema.py
     ```
   - Or with `psql`: `psql $DATABASE_URL -f strategies/utils/schema.sql` then the evaluation extension.

4. **Embedding dimension must match DB**
   - The corpus ingest test uses the embedder; its output dimension must match the DB schema. Set **`EMBEDDING_BACKEND`** in `.env` (or the shell) to match how Postgres was inited:
     - **Default Docker** (schema.sql, 1536-dim): use `EMBEDDING_BACKEND=openai` (or unset). You need `OPENAI_API_KEY` set.
     - **BGE-M3** (schema_1024.sql, 1024-dim): use `EMBEDDING_BACKEND=bge-m3` and ensure the DB was inited with the BGE-M3 override (e.g. `docker-compose -f docker-compose.yml -f docker-compose.bge-m3.yml` with a fresh volume).
   - If you see `DataError: expected 1536 dimensions, not 1024` (or the reverse), fix by aligning `EMBEDDING_BACKEND` with the DB schema or the test will skip with a short explanation.

5. **Run the integration test**
   - With `DATABASE_URL` set, Postgres running, and `EMBEDDING_BACKEND` matching the DB:
     ```bash
     pytest tests/ -v -m integration
     ```
   - The corpus ingest integration test should run (or skip with a clear message if dimension mismatch).

---

## Manual E2E checklist (testing up to now)

Use this to verify the evaluation pipeline, embedder backends, and schema selection end-to-end:

1. **Schema selection**
   - Set `EMBEDDING_BACKEND=bge-m3`, run `python scripts/run_schema.py`, and confirm the 1024-dim schema is applied (e.g. inspect `chunks` column or `match_chunks` signature).
   - Unset or set `EMBEDDING_BACKEND=openai`, run `python scripts/run_schema.py` again, and confirm the 1536-dim schema is used.

2. **Embedder**
   - Set `EMBEDDING_BACKEND=bge-m3`, run a one-liner that calls `embed_query("test")` (from `strategies.utils.embedder`) and prints `len(result)` — expect 1024.
   - With OpenAI backend and `OPENAI_API_KEY` set, expect 1536 (or `EMBEDDING_DIMENSIONS` if set).

3. **Evaluation pipeline**
   - With DB, schema, and embedder configured, run (use **actual file paths**, not placeholders like `<gold.xlsx>` — the shell treats `<` as redirection):
     ```bash
     python scripts/run_evaluation_pipeline.py \
       --gold datasets/evaluation_gold_doc_bioasq_sampled_100_title_v1_2026Feb13.xlsx \
       --corpus datasets/evaluation_corpus_bioasq_sampled_100_title_v1_2026Feb13.xlsx \
       --strategies standard --out-dir ./eval_out
     ```
   - Confirm no errors and that the report (and optional JSON dataset) is written under `--out-dir`.

---

## Lessons learned / troubleshooting

- **conftest loads `.env`** — Pytest loads the project root `.env` so integration tests and any test that reads `os.getenv` see `DATABASE_URL`, `EMBEDDING_BACKEND`, etc. Tests that need to simulate “missing” or “openai” must override explicitly (e.g. `patch.dict("os.environ", {"OPENAI_API_KEY": ""})` or `monkeypatch.setenv("EMBEDDING_BACKEND", "openai")`).
- **Integration test: only one test with `-m integration`** — There is a single test marked `@pytest.mark.integration` (`test_ingest_corpus_and_get_chunk_map_with_db`). So `pytest tests/ -v -m integration` selects 1 test; the rest are deselected by design.
- **Embedding dimension mismatch** — If the ingest integration test skips with “expected 1536 dimensions, not 1024” (or the reverse), align `EMBEDDING_BACKEND` with the DB schema (openai → 1536, bge-m3 → 1024). The test now skips with a short message instead of failing deep in asyncpg.
- **Source path on macOS** — Ingestion stores `documents.source` using `os.path.relpath(file_path, documents_dir)`. On macOS, `/var` and `/private/var` can differ, so both paths are resolved with `Path(...).resolve()` before `relpath` so the stored source is stable (e.g. `int_doc_1.txt`) and the corpus ingest doc_id→chunk_ids map matches.
- **Pytest filterwarnings** — In `pyproject.toml` under `[tool.pytest.ini_options]`, `filterwarnings` must be a **single key** with an **array** of strings (e.g. for SwigPy* deprecation warnings). Duplicate `filterwarnings` lines cause a TOML redefinition error.
