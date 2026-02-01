# Migration Guide

This guide helps you migrate from the original [all-rag-strategies](https://github.com/coleam00/ottomator-agents/tree/main/all-rag-strategies) implementation to RAG-Advanced.

---

## Key differences

| Feature | all-rag-strategies | RAG-Advanced |
|---------|--------------------|--------------|
| Strategy selection | Manual (agent picks tools) | Registry-based; API or code chooses strategy |
| Chaining | Not supported | Built-in (`POST /chain`, `ChainExecutor`) |
| Comparison | Manual | Parallel execution (`POST /compare`) |
| Cost tracking | Not supported | Per-request tracking (CostTracker, pricing) |
| Evaluation | Not supported | IR metrics + benchmarks (REST API) |
| API | CLI only | REST API (FastAPI) |
| Authentication | None | API key (`X-API-Key`) |
| Rate limiting | None | Redis-based (optional) |

---

## Breaking changes

### 1. Import paths

```python
# Before (all-rag-strategies)
from implementation.agents.rag_agent_advanced import search_with_reranking
from ingestion.embedder import create_embedder

# After (RAG-Advanced)
from orchestration.executor import StrategyExecutor
from orchestration.models import StrategyConfig
from strategies.utils.embedder import embed_query  # or create_embedder-style usage via strategies
# Run via API or:
result = await executor.execute(strategy_name="reranking", query=query, config=StrategyConfig(initial_k=20, final_k=5))
```

### 2. Strategy execution

```python
# Before (all-rag-strategies)
result = await search_with_reranking(ctx, query, limit=5)

# After (RAG-Advanced)
from orchestration.executor import StrategyExecutor
from orchestration.models import StrategyConfig

executor = StrategyExecutor()
result = await executor.execute(
    strategy_name="reranking",
    query=query,
    config=StrategyConfig(limit=5, initial_k=20, final_k=5),
)
# result.documents, result.cost_usd, result.latency_ms
```

### 3. Database schema

RAG-Advanced uses the same base schema (documents, chunks, match_chunks) plus optional evaluation extensions. Apply schemas in order:

```bash
psql $DATABASE_URL < strategies/utils/schema.sql
psql $DATABASE_URL < evaluation/schema_extension.sql
# Or without psql:
python scripts/run_schema.py
```

---

## Migration steps

### Step 1: Update dependencies

```bash
pip uninstall old-package  # if you had one
pip install -e path/to/RAG-Advanced
```

### Step 2: Environment

Copy `.env.example` to `.env` and set:

- `DATABASE_URL` — PostgreSQL connection string (same as all-rag-strategies).
- `OPENAI_API_KEY` — For embeddings and LLM strategies.
- `REDIS_URL` — Optional; for rate limiting (e.g. `redis://localhost:6379`).

### Step 3: Database migration

```bash
python scripts/run_schema.py
# or
psql $DATABASE_URL < strategies/utils/schema.sql
psql $DATABASE_URL < evaluation/schema_extension.sql
```

### Step 4: Update code

Replace direct strategy calls with `StrategyExecutor().execute(...)` or use the REST API (`POST /execute`, `POST /chain`, `POST /compare`). See “Breaking changes” above.

### Step 5: API key (if using the API)

Generate an API key via your auth mechanism (see `api/auth.py` and project README for how keys are created and stored).

---

## Rollback

- Keep your original `.env` and database backup.
- Schema changes are additive; the base tables (documents, chunks) remain compatible.
- To revert code: reinstall the all-rag-strategies implementation and point your app back to it.

---

## Related docs

- [MIGRATION_AND_TASKMASTER_NOTES.md](MIGRATION_AND_TASKMASTER_NOTES.md) — Why the 11 strategies and example docs were not in the task-master PRD; current implementation status.
- [README_API.md](README_API.md) — Full API reference for RAG-Advanced.
