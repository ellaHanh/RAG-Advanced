# Migration Guide

Migrate from [all-rag-strategies](https://github.com/coleam00/ottomator-agents/tree/main/all-rag-strategies) to RAG-Advanced.

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
from strategies.utils.embedder import embed_query
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

Same base schema (documents, chunks, match_chunks) plus optional evaluation extensions. Apply in order:

```bash
psql $DATABASE_URL < strategies/utils/schema.sql
psql $DATABASE_URL < evaluation/schema_extension.sql
# Or without psql:
python scripts/run_schema.py
```

---

## Migration steps

1. **Dependencies** — `pip install -e path/to/RAG-Advanced`.
2. **Environment** — Copy `.env.example` to `.env`; set `DATABASE_URL`, `OPENAI_API_KEY`, optional `REDIS_URL`.
3. **Database** — `python scripts/run_schema.py` (or psql with both schema files).
4. **Code** — Replace direct strategy calls with `StrategyExecutor().execute(...)` or use REST API (`POST /execute`, `POST /chain`, `POST /compare`).
5. **API key** — If using the API, create keys via your auth mechanism (see `api/auth.py` and root README).

---

## Rollback

Keep original `.env` and a DB backup. Schema is additive; base tables stay compatible. Reinstall all-rag-strategies and point the app back if needed.

---

## Related

- [MIGRATION_AND_TASKMASTER_NOTES.md](MIGRATION_AND_TASKMASTER_NOTES.md) — PRD scope and strategy status.
- [api/README.md](../api/README.md) — API reference.
