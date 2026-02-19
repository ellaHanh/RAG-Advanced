# Development Workflow: Docker, Tests, Python

Reference for running Docker, tests, and Python scripts so the same setup works reliably for later use.

---

## Docker

### What runs where

| Service   | Port | Purpose |
|----------|------|--------|
| Postgres | 5432 | Database (pgvector). **Not HTTP** — use `psql` or a DB client, not a browser. |
| Redis    | 6379 | Caching / rate limiting. |
| API      | 8000 | REST API. Use **http://localhost:8000** (e.g. `/health`, `/docs`) in a browser or curl. |

- **http://localhost:5432/** does not work: port 5432 is the Postgres protocol. Browsers send HTTP, so you may see Postgres log “invalid length of startup packet” (ignore it; use 8000 for the API).
- To reach the API you must start the **api** service (or run uvicorn on the host). Starting only `postgres` does not start the API.

### Commands

```bash
# Default (OpenAI 1536-dim schema): Postgres + Redis + API
docker-compose up -d

# Only Postgres (+ Redis); run API locally with venv
docker-compose up -d postgres redis
# In another terminal:
uvicorn api.main:app --reload --port 8000

# BGE-M3 (1024-dim): use override and a fresh volume so init runs with schema_1024.sql
docker-compose down -v
docker-compose -f docker-compose.yml -f docker-compose.bge-m3.yml up -d
```

- **Prerequisites:** Copy `example.env` to `.env` and set at least `OPENAI_API_KEY`. The api service reads `.env` via `env_file`.
- **Changing `.env` (e.g. `EMBEDDING_BACKEND=bge-m3`) does not require rebuilding the API image** — the same image reads env at runtime. Rebuild only when code or Dockerfile changes: `docker-compose build api` or `docker-compose up -d --build api`.
- **Postgres init:** Default compose mounts `schema.sql` (1536-dim). For BGE-M3, use `docker-compose.bge-m3.yml` with a **new** volume so init uses `schema_1024.sql`; otherwise dimension mismatch errors occur when ingesting.

---

## Tests

### How to run

From repo root with venv activated and `pip install -e .`:

```bash
# All tests (unit + integration)
pytest tests/ -v

# Unit only (no DB/Redis; fast, CI-friendly)
pytest tests/ -v -m "not integration and not slow"

# Integration only (needs DATABASE_URL and matching EMBEDDING_BACKEND)
pytest tests/ -v -m integration
```

- **conftest** loads the project `.env`, so `DATABASE_URL`, `EMBEDDING_BACKEND`, and `OPENAI_API_KEY` are set for tests. Tests that need “missing key” or “openai backend” must override env (e.g. `patch.dict("os.environ", {"OPENAI_API_KEY": ""})`, `monkeypatch.setenv("EMBEDDING_BACKEND", "openai")`).
- **Integration tests:** Marked with `integration`: (1) `test_ingest_corpus_and_get_chunk_map_with_db` — ingest small corpus; (2) `test_pipeline_e2e_minimal` — full pipeline with 1 query and 2 corpus docs (~10–30 s). Both need Postgres with pgvector and **embedding dimension matching** `EMBEDDING_BACKEND` (openai → 1536, bge-m3 → 1024). The pipeline E2E test catches strategy-not-found and orchestration issues early. See [README_tests.md](README_tests.md) for DB setup and dimension alignment.
- **Warnings:** SwigPy* deprecation warnings (from sentence-transformers) are filtered in `pyproject.toml` via `filterwarnings` (array of strings).

---

## Python process (scripts and app)

### Environment

- **`.env`** at repo root: set `DATABASE_URL`, `REDIS_URL`, `OPENAI_API_KEY`; for BGE-M3 set `EMBEDDING_BACKEND=bge-m3`. Scripts and the API load `.env` when present (e.g. via `python-dotenv` or conftest for pytest).
- **Embedding backend:** `EMBEDDING_BACKEND=openai` (default) or `bge-m3`. Must match the DB schema (1536 vs 1024). `EMBEDDING_MODEL` / `EMBEDDING_DIMENSIONS` are only used for the OpenAI backend; for BGE-M3 they are ignored.

### Common commands (from repo root, venv activated)

```bash
# Apply DB schema (picks schema.sql or schema_1024.sql from EMBEDDING_BACKEND)
python scripts/run_schema.py

# Ingest documents (default: ./documents) — for API/search, not for the evaluation pipeline
python -m strategies.ingestion.ingest --documents ./documents

# Evaluation pipeline (use actual paths to your xlsx files)
python scripts/run_evaluation_pipeline.py \
  --gold datasets/evaluation_gold_doc_bioasq_sampled_100_title_v1_2026Feb13.xlsx \
  --corpus datasets/evaluation_corpus_bioasq_sampled_100_title_v1_2026Feb13.xlsx \
  --strategies standard --out-dir ./eval_out

# Faster pipeline test (2 queries, 20 corpus docs) — finishes in minutes instead of an hour
python scripts/run_evaluation_pipeline.py \
  --gold datasets/evaluation_gold_doc_bioasq_sampled_100_title_v1_2026Feb13.xlsx \
  --corpus datasets/evaluation_corpus_bioasq_sampled_100_title_v1_2026Feb13.xlsx \
  --strategies standard --out-dir ./eval_out --limit-queries 2 --limit-corpus 20
```

- **Pipeline:** Pass real file paths to `--gold` and `--corpus`. Do not use shell placeholders like `<gold.xlsx>` (the shell interprets `<` as input redirection).
- **Faster pipeline test:** Use `--limit-queries N` and `--limit-corpus N` to run on a subset of data (e.g. 2 queries and 20 docs) so you can verify ingestion → benchmark → report in a few minutes. Useful to catch "strategy not found" or schema/embedder issues before a full run.

### When to run what

| What | When |
|------|------|
| **run_schema.py** | **Once** (or after recreating the DB / changing `EMBEDDING_BACKEND`). Creates or updates tables and vector dimensions. The evaluation pipeline **does not** run it — the DB must already have the correct schema. |
| **run_evaluation_pipeline.py** | Whenever you want to run benchmarks. It **loads** gold and corpus xlsx, **ingests the corpus xlsx** into the DB (writes temp files, runs ingestion, builds doc_id→chunk_ids), runs benchmarks, and writes the report. It does **not** run `run_schema.py` and does **not** ingest `./documents`. |
| **Ingest ./documents** | When you want the **API** (or other code) to search over your own documents in `./documents`. Separate from the evaluation pipeline; the pipeline only ingests the **corpus file** you pass with `--corpus`. |

### Docker + evaluation: command sequence

Starting Docker **does not** run schema or ingestion. Postgres is either empty (new volume) or has whatever data was there before (volume persists). To have data and use the API:

```bash
# 1. Start stack (e.g. BGE-M3). Postgres is empty on first run, or has previous data.
docker-compose -f docker-compose.yml -f docker-compose.bge-m3.yml up -d

# 2. (First time or new DB only) Apply schema.
python scripts/run_schema.py

# 3a. Run evaluation: ingests corpus xlsx into Postgres, then runs benchmarks.
python scripts/run_evaluation_pipeline.py --gold <gold.xlsx> --corpus <corpus.xlsx> --out-dir ./eval_out

# 3b. Same corpus, different strategies: skip ingestion (saves hours).
python scripts/run_evaluation_pipeline.py --gold <gold.xlsx> --corpus <corpus.xlsx> --out-dir ./eval_out --skip-ingest

# 3c. Or ingest ./documents for API search (separate from evaluation corpus).
python -m strategies.ingestion.ingest --documents ./documents
```

After step 3a, Postgres holds the ingested data. Restarting Docker (`docker-compose up -d`) does **not** clear it; you do **not** re-ingest unless you change corpus or want a clean DB. See [README_evaluation_pipeline.md](README_evaluation_pipeline.md) § “How it works: strategies, ingestion, and Postgres” for why ingestion repeats every pipeline run and how to shorten it Use **`--skip-ingest`** on later runs; see README_evaluation_pipeline.md for per-query results.)

---

## Quick reference

| Goal                    | Command / note |
|-------------------------|----------------|
| Run API in Docker       | `docker-compose up -d` (need `.env` with OPENAI_API_KEY). |
| Run API on host         | `docker-compose up -d postgres redis` then `uvicorn api.main:app --reload --port 8000`. |
| BGE-M3 from scratch     | `docker-compose -f docker-compose.yml -f docker-compose.bge-m3.yml up -d` with fresh volume; `EMBEDDING_BACKEND=bge-m3` in `.env`. |
| Unit tests only         | `pytest tests/ -v -m "not integration and not slow"`. |
| Integration test        | Postgres up, `DATABASE_URL` and `EMBEDDING_BACKEND` set; `pytest tests/ -v -m integration`. |
| Apply schema            | `python scripts/run_schema.py` (uses `DATABASE_URL` and `EMBEDDING_BACKEND` from `.env`). |
| Run evaluation pipeline | `python scripts/run_evaluation_pipeline.py --gold <path> --corpus <path> --out-dir ./eval_out`. |
| Re-run benchmarks only (same corpus) | Add `--skip-ingest` to skip ingestion; uses existing DB. |
| Fast pipeline check      | Add `--limit-queries 2 --limit-corpus 20`; or run `pytest tests/test_evaluation/test_pipeline_integration.py -m integration`. |
