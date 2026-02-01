# Quick Start: Issues, Fixes, and Findings

This document records issues encountered during local Quick Start setup and the fixes applied. Use it when setting up RAG-Advanced or when something doesn’t work as in the README.

---

## 1. `psql: command not found` at Step 3 (Setup Database)

**Issue**  
Step 3 in the README said to run:
```bash
psql $DATABASE_URL < strategies/utils/schema.sql
```
On machines without the PostgreSQL client (`psql`) installed, this fails with:
```text
zsh: command not found: psql
```

**Fix**  
- **Option A:** Use Docker Compose for Postgres (Step 4). The schema is applied automatically on first start via `docker-entrypoint-initdb.d`; **skip Step 3**.
- **Option B:** Apply the schema without `psql` by running the Python script from the repo root (with venv activated):
  ```bash
  python scripts/run_schema.py
  ```
  The script reads `DATABASE_URL` from `.env` and runs both `strategies/utils/schema.sql` and `evaluation/schema_extension.sql` using asyncpg.

**README change**  
Quick Start §3 was updated to describe Option A (Docker, skip step) and Option B (existing Postgres) with both `psql` and `python scripts/run_schema.py`.

---

## 2. New artifact: `scripts/run_schema.py`

**What**  
A small script that applies the RAG-Advanced schema without requiring `psql`.

**Location**  
`scripts/run_schema.py`

**Behavior**  
- Loads `.env` from the repo root (if `python-dotenv` is available).
- Uses `DATABASE_URL` from the environment.
- Runs, in order:
  - `strategies/utils/schema.sql` (pgvector, documents, chunks, match_chunks, etc.)
  - `evaluation/schema_extension.sql` (api_keys, evaluation_runs, strategy_metrics, etc.)
- Splits SQL on `;` while respecting `$$...$$` so function bodies are not broken.

**Usage**  
From repo root, with venv activated:
```bash
python scripts/run_schema.py
```

**Requirements**  
Project dependencies (e.g. `pip install -e .`), which include `asyncpg` and `python-dotenv`.

---

## 3. Optional: ivfflat index on empty `chunks` table

**Observation**  
The base schema creates an ivfflat index on `chunks.embedding` with `lists = 100`. Some pgvector versions or configurations require the table to have at least as many rows as `lists` before the index can be built. If you run the schema against an **empty** database and see an error on that index, it is likely this.

**Workaround**  
- Prefer **Docker Compose** for first-time setup so Postgres (and schema) start in a known order, or  
- Run `scripts/run_schema.py` after ensuring Postgres has the pgvector extension and, if needed, create the ivfflat index later (e.g. after ingesting some documents) or reduce `lists` in the schema to match your initial row count.

No code change was made; the schema is unchanged. Document ingestion is still “Coming Soon,” so empty `chunks` is expected initially.

---

## 4. API key and auth (optional for local testing)

**Observation**  
The API is configured to support API key auth (see `api/auth.py` and `evaluation/schema_extension.sql` for `api_keys`). The health and root endpoints do not require a key. Whether other routes require a key depends on how middleware/dependencies are wired; the OpenAPI docs at `/docs` list which endpoints need headers.

**Finding**  
For local Quick Start, you can confirm the app is up with `curl http://localhost:8000/health` and `curl http://localhost:8000/strategies` without an API key. If you later enable auth on all routes, you will need to create an API key (e.g. via a seed script or admin path) and pass it in `X-API-Key`.

---

## 5. Strategy registry may be empty

**Observation**  
`GET /strategies` returns whatever is registered in the orchestration registry. If no RAG strategy agents are registered at startup (e.g. agents from `strategies/agents/` are not yet wired into the app), the list will be empty. The API and health check still work.

**Finding**  
This is expected until strategy implementations are registered (e.g. from all-rag-strategies or local agents). It is not a Quick Start bug. Task/backlog: wire strategy agents and optional DB/embedding dependencies so `/strategies` and `/execute` return and run real strategies.

---

## 6. Redis and rate limiting

**Observation**  
Rate limiting uses Redis (`REDIS_URL`). If Redis is not running or not configured, the app starts but rate limiting may be disabled or no-op. Docker Compose starts Redis; if you run the API alone, either start Redis or accept no rate limiting.

**Finding**  
No change required for Quick Start. Document in README or env example that Redis is optional for local dev but required for rate limiting.

---

## 7. Ingestion pipeline implemented

**Change**  
The ingestion pipeline is implemented under `strategies/ingestion/`. It reads documents (Markdown, PDF, DOCX, audio via Docling), chunks them (simple or Docling HybridChunker), embeds with OpenAI, and inserts into `documents` and `chunks`.

**Usage**  
From repo root with venv activated and `DATABASE_URL` + `OPENAI_API_KEY` set:
```bash
python -m strategies.ingestion.ingest
```
Default documents path: **`./documents`** under repo root. The repo includes a `documents/` folder with example files so ingestion is self-contained.

**Docs**  
See [docs/README_ingestion.md](README_ingestion.md) for options and supported file types.

---

## Summary of changes made

| Item | Change |
|------|--------|
| README Quick Start §3 | Clarified when to skip DB setup (Docker); added Option B with `psql` and `python scripts/run_schema.py`. |
| README Quick Start §4 | Clarified Docker Compose (Postgres + Redis + API) vs. running only Postgres + Redis and API locally. |
| README after §5 | Added §5 “Verify the API” (health + /strategies) and renumbered “Ingest Documents” to §6. |
| README §6 | Replaced "Coming Soon" with ingestion CLI and link to docs/README_ingestion.md. |
| New script | `scripts/run_schema.py` to apply schema without `psql`. |
| Ingestion | `strategies/ingestion/` pipeline; CLI `python -m strategies.ingestion.ingest`. |
| Ingestion docs | `docs/README_ingestion.md` — usage, options, supported types. |
| This doc | `docs/QUICKSTART_FINDINGS.md` — issues, fixes, and findings for future setup and troubleshooting. |

---

*Last updated during Quick Start setup (post–taskmaster completion).*
