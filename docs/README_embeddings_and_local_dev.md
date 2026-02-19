# Embeddings and Local Dev (pgvector + BGE-M3)

RAG-Advanced uses **pgvector** for vector storage in all environments. For local dev and BioASQ-style evaluation you can use **BGE-M3** (CPU-optimized, no API key) instead of OpenAI embeddings.

---

## Current setup

- **Vector store**: PostgreSQL with the **pgvector** extension (`strategies/utils/schema.sql` or `schema_1024.sql`). No change needed for “using pgvector for local dev”—it is already the store.
- **Embeddings**: Backend is chosen by **`EMBEDDING_BACKEND`**:
  - **`openai`** (default): OpenAI `text-embedding-3-small`, 1536 dimensions. Requires `OPENAI_API_KEY`. Use base schema `schema.sql`.
  - **`bge-m3`**: BAAI/bge-m3 via sentence-transformers, 1024 dimensions, CPU-friendly. No API key. Use **`schema_1024.sql`** so the `chunks.embedding` column and `match_chunks()` use `vector(1024)`.

---

## Local dev with pgvector + BGE-M3 (BioASQ)

1. **Environment**
   - Set `EMBEDDING_BACKEND=bge-m3`.
   - Optional: `EMBEDDING_DEVICE=cpu` (default) or `cuda` if you have GPU.
   - `DATABASE_URL` pointing at a PostgreSQL instance with pgvector.

2. **Schema**
   - Run the **1024-dim schema** so it matches BGE-M3:
     - **Docker from scratch:** use the BGE-M3 override so Postgres init uses `schema_1024.sql`:  
       `docker-compose -f docker-compose.yml -f docker-compose.bge-m3.yml up -d` (with a fresh volume; see [README_development_workflow.md](README_development_workflow.md)).
     - **Existing Postgres:** set `EMBEDDING_BACKEND=bge-m3` and run:
       ```bash
       python scripts/run_schema.py
       ```
       (the script picks `schema_1024.sql` when `EMBEDDING_BACKEND=bge-m3`)
     - Or run `schema_1024.sql` manually (e.g. `psql $DATABASE_URL -f strategies/utils/schema_1024.sql`) then the evaluation extension if needed.

3. **Ingestion and API**
   - Ingest documents as usual; chunk embedding uses BGE-M3 when `EMBEDDING_BACKEND=bge-m3`.
   - The API and evaluation pipeline use the same embedder, so retrieval is consistent.

4. **Evaluation pipeline**
   - Run the evaluation pipeline with your gold/corpus xlsx; ingestion will embed with BGE-M3 and store 1024-dim vectors in pgvector.

---

## Alternatives (other CPU-friendly embedders)

If you want options beyond BGE-M3 for local/BioASQ:

| Model | Dims | Notes |
|-------|------|--------|
| **BAAI/bge-m3** (current) | 1024 | Multilingual, long docs (8k tokens), good for BioASQ; CPU-friendly. |
| **intfloat/multilingual-e5-large** | 1024 | Strong multilingual; would need a second backend and schema_1024. |
| **intfloat/e5-large-v2** | 1024 | English; same idea. |
| **sentence-transformers/all-MiniLM-L6-v2** | 384 | Small and fast; would need `schema_384.sql` and a new backend. |

Adding another backend (e.g. e5-large) would follow the same pattern: a backend in `strategies/utils/embedder.py`, and a schema file with the matching `vector(N)` and `match_chunks(vector(N))` if the dimension differs from 1536/1024.

---

## Summary

- **pgvector**: Already used for local and other environments; no extra setup.
- **BGE-M3**: Set `EMBEDDING_BACKEND=bge-m3`, run schema with 1024-dim (e.g. `python scripts/run_schema.py` with that env), then ingest and run the evaluation pipeline as usual.
