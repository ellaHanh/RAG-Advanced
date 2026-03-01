# Development: Findings, Workflow, Local Dev

Quick Start findings, Docker/tests workflow, and local embeddings (BGE-M3) for development and BioASQ-style evaluation.

---

## Quick Start findings

Use this when setup doesn’t match the main README or for troubleshooting.

| Issue | Fix |
|-------|-----|
| **`psql: command not found`** | Use Docker (schema on first start) and skip DB step, or run `python scripts/run_schema.py` (no psql needed). |
| **Schema without psql** | `scripts/run_schema.py` — loads `.env`, runs `strategies/utils/schema.sql` and `evaluation/schema_extension.sql`; respects `$$...$$`. |
| **ivfflat on empty chunks** | Some pgvector setups need rows ≥ `lists`; use Docker for first setup or create index after ingesting. |
| **API key** | Health/root are unauthenticated; other routes may need `X-API-Key` (see `api/auth.py`). |
| **Empty `/strategies`** | Expected until strategy agents are registered. |
| **Redis** | Optional for local dev; required for rate limiting. |

---

## Docker and tests

### Services

| Service | Port | Purpose |
|---------|------|---------|
| Postgres | 5432 | pgvector (use psql or DB client, not browser). |
| Redis | 6379 | Cache / rate limiting. |
| API | 8000 | REST API (`http://localhost:8000/docs`). |

```bash
# Full stack (OpenAI 1536-dim)
docker-compose up -d

# Postgres + Redis only; API on host
docker-compose up -d postgres redis
uvicorn api.main:app --reload --port 8000

# BGE-M3 (1024-dim): fresh volume
docker-compose down -v
docker-compose -f docker-compose.yml -f docker-compose.bge-m3.yml up -d
```

- `.env` is read at runtime; no rebuild for env changes. Rebuild only for code/Dockerfile changes.
- For BGE-M3, use a **new** volume so init uses `schema_1024.sql`; otherwise dimension mismatch.

### Tests

```bash
pytest tests/ -v
pytest tests/ -v -m "not integration and not slow"   # unit only, CI-friendly
pytest tests/ -v -m integration   # needs DATABASE_URL, matching EMBEDDING_BACKEND
```

Integration tests need Postgres with pgvector and schema (1536 or 1024) aligned with `EMBEDDING_BACKEND`. See [tests/README.md](../tests/README.md).

### Common commands (repo root, venv)

```bash
python scripts/run_schema.py
python -m strategies.ingestion.ingest --documents ./documents
python scripts/run_evaluation_pipeline.py --gold <path> --corpus <path> --out-dir ./eval_out
# Re-run benchmarks only (same corpus): add --skip-ingest
# Fast check: add --limit-queries 2 --limit-corpus 20
```

---

## Embeddings and local dev (BGE-M3)

- **Store:** pgvector in all environments (`schema.sql` or `schema_1024.sql`).
- **Backend:** `EMBEDDING_BACKEND=openai` (1536, needs `OPENAI_API_KEY`) or `bge-m3` (1024, CPU-friendly, no key).

**BGE-M3 setup:**

1. Set `EMBEDDING_BACKEND=bge-m3`; optional `EMBEDDING_DEVICE=cpu` or `cuda`.
2. **Schema:** Docker from scratch with BGE-M3 compose override (fresh volume), or existing Postgres: `python scripts/run_schema.py` (picks `schema_1024.sql` when `EMBEDDING_BACKEND=bge-m3`).
3. Ingest and run pipeline as usual; API and pipeline use the same embedder.

Alternatives (same pattern: backend in `strategies/utils/embedder.py` + matching `vector(N)` schema): e5-large (1024), all-MiniLM-L6-v2 (384).

---

## Related

- [Root README](../README.md)
- [scripts/README.md](../scripts/README.md) — run_schema, pipeline
- [evaluation/README.md](../evaluation/README.md) — Pipeline options
- [tests/README.md](../tests/README.md) — DB and integration tests
