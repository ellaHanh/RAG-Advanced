# all-rag-strategies vs RAG-Advanced: Folder/File Comparison

This document compares [all-rag-strategies](https://github.com/coleam00/ottomator-agents/tree/main/all-rag-strategies) with RAG-Advanced at folder and file level. It summarizes what was migrated, what was adapted, and what exists only in one project.

**Legend**

- **Migrated**: Content/behavior copied or reimplemented in RAG-Advanced with same purpose.
- **Partial**: Partially migrated or aligned; differences noted.
- **Not migrated**: Not present in RAG-Advanced (optional or out of scope).
- **RAG-Advanced only**: No counterpart in all-rag-strategies.

---

## 1. Top-level structure

| all-rag-strategies | RAG-Advanced | Status | Notes |
|--------------------|--------------|--------|-------|
| `docs/` (11 .md) | `strategies/docs/` (11 .md + README) | **Migrated** | Copied; README added for strategy reference. |
 **Migrated** | Copied; README updated with RAG-Advanced API equivalents. |
| `implementation/` | See section 2 | **Partial** | RAG-Advanced splits implementation across `strategies/`, `api/`, `orchestration/`, `evaluation/`. |
| `README.md` | `README.md` | **Partial** | RAG-Advanced README describes orchestration, API, evaluation; references all-rag-strategies. |

---

## 2. implementation/ (all-rag-strategies) vs RAG-Advanced

| all-rag-strategies path | RAG-Advanced path | Status | Notes |
|-------------------------|-------------------|--------|-------|
| `implementation/.env.example` | `.env.example` | **Partial** | RAG-Advanced has broader env vars (API, Redis, evaluation). |
| `implementation/.gitignore` | `.gitignore` | **Partial** | Both ignore venv, etc.; RAG-Advanced adds project-specific entries. |
| `implementation/cli.py` | No direct equivalent | **Partial** | RAG-Advanced: run API via `uvicorn api.main:app`; ingestion via `python -m strategies.ingestion.ingest`. |
| `implementation/docker-compose.yml` | `docker-compose.yml` | **Partial** | RAG-Advanced composes API, Postgres, Redis; may differ in services. |
| `implementation/Dockerfile` | `Dockerfile` | **Partial** | Both containerize app; RAG-Advanced image runs API. |
| `implementation/documents/` (PDF, DOCX, MD, MP3) | `documents/` | **Migrated** | Contents copied; default ingestion folder in RAG-Advanced. |
| `implementation/IMPLEMENTATION_GUIDE.md` | `strategies/ingestion/README.md`, `docs/README_development.md` | **Partial** | Setup/ingestion guidance in RAG-Advanced docs instead of single guide. |
| `implementation/ingestion/__init__.py` | `strategies/ingestion/__init__.py` | **Migrated** | — |
| `implementation/ingestion/chunker.py` | `strategies/ingestion/chunker.py` | **Partial** | RAG-Advanced: same idea (HybridChunker + simple fallback); Pydantic config, cached chunker. |
| `implementation/ingestion/chunker_no_docling.py` | — | **Not migrated** | Optional no-Docling path; RAG-Advanced uses Docling or simple chunker only. |
| `implementation/ingestion/contextual_enrichment.py` | — | **Not migrated** | Anthropic contextual retrieval; not yet in RAG-Advanced ingestion. |
| `implementation/ingestion/embedder.py` | `strategies/ingestion/embedder.py` + `strategies/utils/embedder.py` | **Partial** | Batch embed in utils; ingestion wrapper in strategies/ingestion. |
| `implementation/ingestion/ingest.py` | `strategies/ingestion/ingest.py` | **Partial** | Same pipeline (read → chunk → embed → save); RAG-Advanced uses asyncpg, own config. |
| `implementation/pyproject.toml` | `pyproject.toml` | **Partial** | RAG-Advanced single project; deps include orchestration, evaluation, API. |
| `implementation/rag_agent_advanced.py` | `strategies/agents/` (standard, reranking) | **Partial** | Only standard + reranking implemented; rest of 11 strategies not yet. |
| `implementation/rag_agent.py` | — | **Not migrated** | Simpler agent; RAG-Advanced uses orchestration layer, not Pydantic AI agent directly. |
| `implementation/README.md` | `README.md`, `docs/` | **Partial** | RAG-Advanced README + docs cover setup and usage. |
| `implementation/requirements-advanced.txt` | `pyproject.toml` (dependencies) | **Partial** | RAG-Advanced uses pyproject.toml for deps. |
| `implementation/sql/schema.sql` | `strategies/utils/schema.sql` | **Partial** | Same idea (documents, chunks, match_chunks); RAG-Advanced schema may differ (e.g. no token_count). |
| `implementation/sql/removeDocuments.sql` | — | **Not migrated** | Can be done via API or direct SQL; no dedicated script. |
| `implementation/STRATEGIES.md` | `strategies/docs/` (11 .md) | **Partial** | Strategy concepts in markdown; STRATEGIES.md not copied. |
| `implementation/utils/db_utils.py` | — | **Not migrated** | RAG-Advanced uses pool in api/main.py and ingestion; no shared db_utils. |
| `implementation/utils/models.py` | `strategies/ingestion/models.py`, `orchestration/models.py` | **Partial** | Ingestion and execution models live in different places. |
| `implementation/utils/providers.py` | — | **Not migrated** | Provider abstraction for embeddings; RAG-Advanced uses OpenAI embedder directly. |
| `implementation/uv.lock` | — | **Optional** | Lock file; RAG-Advanced may use different tooling. |

---

## 3. RAG-Advanced-only (no counterpart in all-rag-strategies)

| RAG-Advanced path | Purpose |
|-------------------|---------|
| `api/` | FastAPI app, auth, rate limiting, routes (strategies, execute, chain, compare, benchmarks, evaluation). |
| `config/` | Pricing and app config. |
| `datasets/` | Sample queries / evaluation datasets. |
| `evaluation/` | IR metrics, benchmarks, reports, ground truth, schema extensions. |
| `orchestration/` | Registry, executor, chain executor, comparison, cost tracker, resource manager. |
| `PRD/` | Product requirements and task-master (tasks, prd.txt). |
| `scripts/run_schema.py` | Apply schema without `psql`. |
| `tests/` | Unit and integration tests. |
| `docs/API.md`, `EVALUATION.md`, `MIGRATION.md`, `MIGRATION_AND_TASKMASTER_NOTES.md`, etc. | RAG-Advanced-specific documentation. |

---

## 4. Strategy implementation status

| Strategy (all-rag-strategies) | RAG-Advanced agent | Status |
|------------------------------|---------------------|--------|
| Standard semantic search | `strategies/agents/standard.py` | **Implemented** |
| Reranking (cross-encoder) | `strategies/agents/reranking.py` | **Implemented** |
| Agentic RAG | — | Not yet |
| Knowledge graphs | — | Not yet |
| Contextual retrieval | Ingestion only (optional in all-rag-strategies) | Not in RAG-Advanced ingestion |
| Query expansion | — | Not yet |
| Multi-query RAG | — | Not yet |
| Context-aware chunking | Docling HybridChunker in ingestion | **Partial** (chunking only) |
| Late chunking | — | Not yet |
| Hierarchical RAG | — | Not yet |
| Self-reflective RAG | — | Not yet |
| Fine-tuned embeddings | — | Not yet |

---

## 5. Summary of disalignments and modifications

1. **Layout**: all-rag-strategies keeps one `implementation/` tree; RAG-Advanced splits into `api/`, `orchestration/`, `evaluation/`, `strategies/`.
2. **Database**: RAG-Advanced uses asyncpg and pool in lifespan; all-rag-strategies uses db_utils and a global pool. Schema is similar but not identical (e.g. chunks table).
3. **Agents**: all-rag-strategies uses Pydantic AI agents with tools; RAG-Advanced uses orchestration (registry + executor) and strategy functions `(ExecutionContext) -> list[Document]`.
4. **Ingestion**: Same pipeline idea; RAG-Advanced has its own reader, chunker, embedder, and CLI; no contextual_enrichment or chunker_no_docling yet.
5. **Docs/examples**: all-rag-strategies `docs/` and `examples/` are copied under `strategies/docs/` and `strategies/docs/`; READMEs updated for RAG-Advanced.
6. **Example documents**: Copied from all-rag-strategies `implementation/documents/` to RAG-Advanced `documents/` as default ingestion folder.

For why the 11 strategies and example documents were not part of the task-master PRD tasks, see [MIGRATION_AND_TASKMASTER_NOTES.md](MIGRATION_AND_TASKMASTER_NOTES.md).
