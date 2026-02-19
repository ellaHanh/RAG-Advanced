# Scripts

The `scripts/` folder contains runnable utilities that do not require the full API to be up.

---

## run_schema.py

**Purpose:** Apply the RAG-Advanced database schema without needing the `psql` CLI.

**Behavior:**

1. Resolves the repo root (parent of `scripts/`).
2. Loads `.env` from the repo root if `python-dotenv` is available.
3. Reads `DATABASE_URL` from the environment.
4. Runs, in order:
   - Base schema: `strategies/utils/schema.sql` (1536-dim for OpenAI), or **`strategies/utils/schema_1024.sql`** when **`EMBEDDING_BACKEND=bge-m3`** (BGE-M3 local embedder).
   - `evaluation/schema_extension.sql` (evaluation-related extensions, if any).

SQL is split into statements while respecting `$$...$$` dollar-quoted strings so that semicolons inside function bodies do not break parsing.

**Usage (from repo root):**

```bash
python scripts/run_schema.py
```

**Requirements:** `DATABASE_URL` set (e.g. in `.env`). Dependencies: `asyncpg`, `python-dotenv` (from project deps). The script uses asyncpg to connect and execute each statement.

**When to use:** Initial setup or reset of the database schema when you do not have `psql` installed or prefer a single Python command. Docker Compose may apply schema on first start; see the main README.
