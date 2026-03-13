# Scripts

Runnable utilities that do not require the full API.

## Quickstart

```bash
# From repo root, venv activated
python scripts/run_schema.py
```

Requires `DATABASE_URL` (e.g. in `.env`). Uses `schema_1024.sql` when `EMBEDDING_BACKEND=bge-m3`.

## Features

- **run_schema.py** — Apply DB schema without `psql`. Runs `strategies/utils/schema.sql` (or `schema_1024.sql` for BGE-M3) then `evaluation/schema_extension.sql`. Respects `$$...$$` in SQL.
- **run_evaluation_pipeline.py** — Gold + corpus xlsx → ingest → benchmarks → report. See [evaluation/README.md](../evaluation/README.md).

## Usage

```bash
# Schema only
python scripts/run_schema.py

# Evaluation pipeline (use real paths)
python scripts/run_evaluation_pipeline.py \
  --gold path/to/gold.xlsx --corpus path/to/corpus.xlsx \
  --strategies standard reranking --out-dir ./eval_out
```

## Dependencies

Project: `pip install -e .` (asyncpg, python-dotenv for run_schema; pandas, openpyxl for pipeline).

## Related

- [Root README](../README.md)
- [evaluation/](../evaluation/README.md) — Pipeline and RAGAS
