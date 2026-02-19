# Evaluation Pipeline (Gold + Corpus xlsx)

End-to-end pipeline: load gold and corpus from xlsx, ingest corpus into the DB, run benchmarks with the same strategy executor as the API, and write metrics and reports.

---

## Overview

1. **Load gold xlsx** → Dataset (queries with `relevant_doc_ids`).
2. **Load corpus xlsx** → list of documents (id, text, title).
3. **Ingest corpus** → temp directory (one `.txt` per doc), then existing ingestion; build `doc_id → chunk_id` map.
4. **Map gold to chunk IDs** → gold `relevant_passage_ids` (corpus doc_ids) become chunk IDs for metrics.
5. **Run benchmarks** → `BenchmarkRunner` with API’s `StrategyExecutor`; output summary, optional JSON dataset, markdown report.

---

## Column mapping

Column names differ by dataset. You can:

- **Use defaults** — BioASQ v1 (first sheet): gold columns `id`, `question`, `relevant_passage_ids`; corpus `doc_id`, `passage`, `title`.
- **Config file** — YAML or JSON with `gold` and `corpus` sections (see [evaluation/config/bioasq_v1.json](../evaluation/config/bioasq_v1.json)).
- **CLI overrides** — `--gold-map query_id=id,query=question,relevant_doc_ids=relevant_passage_ids` and `--corpus-map doc_id=...,text=...,title=...`.

Gold `relevant_doc_ids` can be stored as a string list; set `list_format` to `json`, `pipe_separated`, or `comma_separated` in config.

---

## Usage

From repo root, with `DATABASE_URL` set (and `.env` loaded). **Use actual file paths** for `--gold` and `--corpus`; do not use placeholders like `<gold.xlsx>` (the shell treats `<` as input redirection).

```bash
python scripts/run_evaluation_pipeline.py \
  --gold datasets/evaluation_gold_doc_bioasq_sampled_100_title_v1_2026Feb13.xlsx \
  --corpus datasets/evaluation_corpus_bioasq_sampled_100_title_v1_2026Feb13.xlsx \
  --strategies standard reranking \
  --out-dir ./eval_out
```

With custom mapping (no config file):

```bash
python scripts/run_evaluation_pipeline.py --gold path/to/gold.xlsx --corpus path/to/corpus.xlsx \
  --gold-map query_id=id,query=question,relevant_doc_ids=relevant_passage_ids \
  --corpus-map doc_id=doc_id,text=passage,title=title \
  --out-dir ./eval_out
```

With config file:

```bash
python scripts/run_evaluation_pipeline.py \
  --config evaluation/config/bioasq_v1.json \
  --gold datasets/gold.xlsx --corpus datasets/corpus.xlsx \
  --out-dir ./eval_out
```

**Options:**

| Option | Description |
|--------|-------------|
| `--gold` | Path to gold xlsx (required). |
| `--corpus` | Path to corpus xlsx (required). |
| `--config` | Path to YAML/JSON column mapping. |
| `--gold-map` | Comma-separated key=val for gold columns. |
| `--corpus-map` | Comma-separated key=val for corpus columns. |
| `--strategies` | Strategy names (default: `standard`). |
| `--out-dir` | Directory for report and optional JSON dataset. |
| `--limit` | Retrieval limit per strategy (default: 10). |
| `--iterations` | Benchmark iterations per query (default: 1). |
| `--limit-queries` | Use only first N queries (quick pipeline test). |
| `--limit-corpus` | Ingest only first N corpus documents (quick pipeline test). Corpus is **first N rows** of the xlsx; it does not auto-select docs referenced in gold. For valid metrics, ensure those first N corpus rows include all `relevant_doc_ids` for the limited queries (see § Limit-queries and limit-corpus below). |
| `--skip-ingest` | Reuse existing DB: do not ingest; build doc_id→chunk_ids from current DB. Use after a full run with the same corpus to re-run only benchmarks (saves hours). |
| `--no-clean` | Do not clear DB before ingesting corpus. |

**When and with what code to use `--skip-ingest` and `--no-clean`**

- These arguments are defined and consumed only by the **evaluation pipeline** script: `scripts/run_evaluation_pipeline.py`. They are parsed in `main()` and passed into `run_pipeline()` / `_run_pipeline_impl()`.

- **Reuse existing DB / avoid long ingest**  
  Use `--skip-ingest` when the corpus is already loaded in Postgres (e.g. after a previous full pipeline run or after a separate ingestion). The pipeline will skip loading the corpus xlsx and will build the doc_id→chunk_ids map from the current DB, then run benchmarks only.

  Example (re-run benchmarks only; no ingest):

  ```bash
  python scripts/run_evaluation_pipeline.py \
    --gold datasets/evaluation_gold_doc_bioasq_sampled_100_title_v1_2026Feb13.xlsx \
    --corpus datasets/evaluation_corpus_bioasq_sampled_100_title_v1_2026Feb13.xlsx \
    --strategies standard reranking \
    --out-dir ./eval_out \
    --skip-ingest
  ```

- **Add to existing corpus without wiping**  
  Use `--no-clean` when you want to ingest a (possibly new) corpus **without** clearing `documents` and `chunks` first. By default the pipeline uses `clean_before_ingest=True` (clean then ingest). With `--no-clean`, it sets `clean_before_ingest=False` so existing rows are kept and new ones are added.

  Example (ingest more docs without clearing the DB):

  ```bash
  python scripts/run_evaluation_pipeline.py \
    --gold path/to/gold.xlsx --corpus path/to/corpus.xlsx \
    --out-dir ./eval_out \
    --no-clean
  ```

- **Combined**: To re-run benchmarks on an already-loaded corpus and avoid any ingest at all, use only `--skip-ingest`. To run a full ingest but preserve existing data, use `--no-clean` (and do not use `--skip-ingest`).
| `--no-json` | Do not write canonical JSON dataset. |
| `--no-report` | Do not write markdown report. |
| `--no-detailed-results` | Do not write per-query per-strategy JSON (`benchmark_results_detailed.json`). |
| `--write-db` | Persist benchmark to Postgres: insert into `evaluation_runs`, `benchmark_results`, and `strategy_metrics`. Requires the evaluation schema extension. |
| `--run-name` | Optional label for the run when using `--write-db` (stored in `evaluation_runs.config`). |
| `-v` | Verbose logging. |

---

## Populating Postgres evaluation tables

By default the pipeline writes only to **files** (summary, `benchmark_results_detailed.json`, markdown report). The Postgres tables **`evaluation_runs`**, **`benchmark_results`**, and **`strategy_metrics`** stay empty unless you pass **`--write-db`**.

To persist results to the DB:

```bash
python scripts/run_evaluation_pipeline.py --gold <path> --corpus <path> \
  --out-dir ./eval_out --write-db --run-name "bioasq_v1_run1"
```

Requires the evaluation schema (`evaluation/schema_extension.sql`) to be applied to your database. The pipeline then inserts one row in `evaluation_runs`, aggregate rows per strategy in `benchmark_results`, and per-query per-strategy rows in `strategy_metrics`. The table **`ground_truth_datasets`** is not filled by the pipeline; it is intended for API-managed datasets.

---

## Strategies

- **Default:** The benchmark runs only the `standard` strategy if you omit `--strategies`. You can pass one or multiple strategy names to evaluate others or compare several in one run.
- **No Docker required:** The pipeline is a Python script. You run it on the host (e.g. with venv and `DATABASE_URL` pointing at Postgres). Docker is only for running Postgres/Redis/API; you do not need to run the pipeline inside Docker to use different strategies.
- **Registered strategy names:** `standard`, `reranking`, `multi_query`, `query_expansion`, `self_reflective`, `agentic`. Use these exact names after `--strategies`.

**Single strategy:**

```bash
python scripts/run_evaluation_pipeline.py --gold <path> --corpus <path> \
  --strategies reranking --out-dir ./eval_out
```

**Multiple strategies (each run separately, compared in the same report):**

```bash
python scripts/run_evaluation_pipeline.py --gold <path> --corpus <path> \
  --strategies standard reranking multi_query query_expansion self_reflective agentic \
  --out-dir ./eval_out
```

The benchmark does not run a single “combined” strategy; it runs each listed strategy independently and reports metrics per strategy so you can compare them.

---

## How it works: strategies, ingestion, and Postgres

### Q1. Strategy loading

When the pipeline starts, it **registers all available strategies** (standard, reranking, multi_query, query_expansion, self_reflective, agentic) with the orchestration registry. The `--strategies` argument does **not** control what is loaded; it controls **which of those** are actually run in the benchmark. So: all strategies are loaded; only the names you pass to `--strategies` are executed and measured.

### Q2. Gold/corpus loading and ingestion

- **Does ingestion write to pgvector?** Yes. The pipeline loads the corpus xlsx, writes one `.txt` per document to a **temporary** directory, then runs the same ingestion pipeline used for `./documents` (chunking → embedding → insert into Postgres). The vectors are stored in the **chunks** table on the Postgres server (the one Docker pgvector provides when you use `DATABASE_URL`). So Docker Postgres is not "just empty space"—after ingestion it holds documents and chunk vectors.

- **Why does it take so long?** Chunking, calling the embedding API (or local embedder), and writing rows into Postgres dominate the runtime. With a large corpus, ingestion is the slowest part.

- **Same corpus, every run?** By default the pipeline **cleans** the DB (deletes all documents and chunks) and **re-ingests** the corpus every time. So yes: with the same gold/corpus, ingestion is repeated on each run. Using `--no-clean` does **not** skip ingestion: the script still writes the corpus to a new temp dir and ingests, so you would add **duplicate** documents (same content, different internal paths). So today there is no "reuse existing index for same corpus" option.

- **After a run:** Data remains in Postgres (Docker volume persists). The next time you run the pipeline with the default (clean before ingest), it will **delete** all documents/chunks and ingest again.

- **Avoiding long ingestion with the same corpus (current and possible):**
  - **Now:** Use `--limit-corpus N` for quick checks (e.g. 20 docs) so ingestion is short. For a full run with the same corpus, re-ingestion is still required each time.
  - **Implemented:** Use **`--skip-ingest`** to skip ingestion and build the doc_id→chunk_ids map from the **existing** DB. Run a full pipeline once (same corpus); then for later runs (e.g. different strategies or limit-queries) use `--skip-ingest` so only benchmarks run. Postgres (Docker or not) keeps the data; restarting Docker does not clear it.

### Q3. Docker, API, and when Postgres has data

- **Start the stack (e.g. BGE-M3):**
  ```bash
  docker-compose -f docker-compose.yml -f docker-compose.bge-m3.yml up -d
  ```
  This starts Postgres (pgvector), Redis, and the API. It does **not** run schema or ingestion.

- **Postgres state:**
  - **First time (new volume):** Postgres is empty. Init scripts create tables (e.g. `schema_1024.sql` with BGE-M3). There is **no** data until you run schema (if needed) and then **ingestion** (pipeline or ingest script).
  - **Later (existing volume):** Postgres keeps whatever data was there. So if you already ran the evaluation pipeline or `ingest --documents`, that data is still there. Restarting Docker does **not** clear it and does **not** run ingestion again.

- **Who fills Postgres?** Only **you**, by running:
  1. **Schema (once per DB/backend):**  
     `python scripts/run_schema.py`
  2. **Evaluation pipeline:**  
     `python scripts/run_evaluation_pipeline.py --gold <path> --corpus <path> --out-dir ./eval_out`  
     This ingests the **corpus xlsx** (and runs the benchmark). It does not ingest `./documents`.
  3. **Or separate ingest for API search:**  
     `python -m strategies.ingestion.ingest --documents ./documents`  
     This is for the API; it is separate from the evaluation corpus.

- **Do you need to re-ingest every time you start Docker?** No. Start Docker → Postgres has previous state (or empty). Re-ingest only when you change the corpus, change the embedding backend/schema, or want a clean DB.

- **Avoiding long ingestion (practical):**
  - Keep the **same Docker volume** so data survives restarts; then you only need to run the full pipeline (with ingestion) when you change gold/corpus or reset.
  - For quick iteration on **benchmarks only** (same corpus), use `--limit-queries` and `--limit-corpus` to shorten the run until a "benchmark-only" / `--skip-ingest` option exists.

### Limit-queries and limit-corpus (proportion and gold coverage)

- **How it works:** `--limit-queries N` uses the **first N rows** of the gold xlsx as queries. `--limit-corpus M` uses the **first M rows** of the corpus xlsx for ingestion. There is no automatic “pick only corpus docs that appear in gold”; it is purely “first N” / “first M”.
- **Proper proportion:** For metrics to be valid, every `relevant_doc_id` in the limited gold queries must exist in the limited corpus. So either:
  - **Don’t limit corpus** (full corpus), and limit only queries for speed; or
  - **Ensure** the first M corpus rows include all doc_ids referenced by the first N gold queries (e.g. if your gold xlsx is ordered so early rows reference early corpus docs, use `--limit-corpus` large enough to cover those doc_ids).
- If a query’s `relevant_doc_ids` are not in the ingested corpus, their ground-truth chunk IDs will be empty and that query’s metrics are misleading. Use full corpus when in doubt.

---

## Outputs

- **Console** — Benchmark summary (latency, MRR, cost, rankings).
- **`--out-dir`** (if set):
  - `gold_dataset_canonical.json` — Gold in canonical query format (unless `--no-json`).
  - `benchmark_report.md` — Markdown report (unless `--no-report`).
  - `benchmark_results_detailed.json` — Per-query, per-strategy results: for each query, `query_id`, `query`, `ground_truth_ids`, and per strategy `retrieved_ids`, `latency_ms`, `cost_usd`, `success`, `error`, and when available `precision_at_k`, `recall_at_k`, `ndcg_at_k`, `mrr` (unless `--no-detailed-results`).

### Strategy performance details (what you get today)

The report gives **per-strategy** aggregates, not per-phase (retrieval vs generation):

- **IR metrics:** Precision@k, Recall@k, NDCG@k, MRR — in `benchmark_report.md` under “IR Metrics” and “Strategy Rankings”.
- **Latency:** One end-to-end latency per execution (retrieval + rerank/LLM in one number). P50/P95/P99/mean per strategy — in “Latency Statistics”.
- **Cost:** Total and avg cost per query per strategy — in “Cost Analysis”.

There is **no** built-in split of “retrieval time” vs “generation time”. To get that you’d need to either instrument strategies to report phases or add timing inside the executor and extend the report.

### Evaluation IDs: chunk_id vs document_id

Benchmark and metrics use **chunk-level** IDs: `ground_truth_chunk_ids` and `retrieved_chunk_ids` are from `chunks.id` (not `documents.id`). Gold document IDs are mapped to chunk IDs after ingestion. See [README_terminology.md](README_terminology.md) for consistent naming.

### Viewing PostgreSQL (pg)

With Docker Postgres on port 5432 and `.env` with `DATABASE_URL=postgresql://postgres:postgres@localhost:5432/rag_advanced`:

```bash
# From host with psql installed
psql "postgresql://postgres:postgres@localhost:5432/rag_advanced" -c "\dt"

# Or interactive
psql "postgresql://postgres:postgres@localhost:5432/rag_advanced"

# From Docker (no local psql needed)
docker exec -it rag-advanced-postgres psql -U postgres -d rag_advanced -c "\dt"
```

Useful: `\dt` (tables), `SELECT COUNT(*) FROM documents;`, `SELECT COUNT(*) FROM chunks;`, `\d chunks` (schema). Any GUI (DBeaver, pgAdmin) can connect to `localhost:5432` with the same user/password/db.

---

## Requirements

- `DATABASE_URL` set (PostgreSQL with schema from `strategies/utils/schema.sql`).
- Dependencies: `pandas`, `openpyxl` (for xlsx); optional `pyyaml` for YAML config.
- For strategies that use embeddings/LLM, the usual env (e.g. OpenAI) must be configured.

---

## Related

- [README_datasets.md](README_datasets.md) — Dataset format and xlsx column mapping.
- [README_evaluation.md](README_evaluation.md) — IR metrics, benchmarks, API.
