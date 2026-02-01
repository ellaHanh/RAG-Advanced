# Evaluation in RAG-Advanced

This document describes how to run evaluation (IR metrics and benchmarks), the contents of the `evaluation/` folder, and why evaluation is exposed only via the REST API.

---

## Overview

Evaluation is implemented and was part of the RAG-Advanced PRD scope:

- **IR metrics**: Precision@k, Recall@k, MRR, NDCG@k — implemented in `evaluation/metrics.py`, exposed via `POST /metrics` and `POST /metrics/batch`.
- **Benchmarks**: Run multiple strategies over a set of queries (latency, cost, rankings) — implemented in `evaluation/benchmarks.py`, exposed via `POST /benchmarks`, `GET /benchmarks/{id}`, `GET /benchmarks/{id}/results`, `DELETE /benchmarks/{id}`.
- **Datasets**: Load and manage test datasets with ground truth — `evaluation/datasets.py`.
- **Reports**: Markdown and HTML report generation from benchmark results — `evaluation/reports.py`, `evaluation/html_reports.py`.
- **Ground truth**: Optional LLM-assisted relevance labeling — `evaluation/ground_truth_llm.py`.

See the main [README.md](../README.md#evaluation) for step-by-step curl examples.

---

## Metrics

### Precision@k

Measures the proportion of relevant documents in the top-k results.

```
Precision@k = (relevant docs in top-k) / k
```

- 1.0 = All top-k docs are relevant  
- Useful for precision-critical applications  

### Recall@k

Measures the proportion of all relevant documents retrieved in top-k.

```
Recall@k = (relevant docs in top-k) / (total relevant docs)
```

- 1.0 = All relevant docs retrieved  
- Useful for high-recall applications (research, legal)  

### Mean Reciprocal Rank (MRR)

Average of the reciprocal ranks of the first relevant result.

```
MRR = mean(1 / rank_of_first_relevant)
```

- 1.0 = First result always relevant  
- Useful for single-answer queries  

### NDCG@k

Normalized Discounted Cumulative Gain — ranking quality with graded relevance.

- 1.0 = Perfect ranking  
- Useful when document relevance is graded (0, 1, 2)  

---

## How to run evaluation

1. **Start the API** (e.g. `docker-compose up -d` or `uvicorn api.main:app --reload`).
2. **Metrics (single query)**  
   `POST /metrics` with body: `retrieved_ids`, `ground_truth_ids`, optional `k_values`.  
   You must supply ground truth (relevant document/chunk IDs). Retrieved IDs are the ordered list your strategy returned (e.g. from `POST /execute`).
3. **Metrics (batch)**  
   `POST /metrics/batch` with body: `queries` (list of objects with `retrieved_ids`, `ground_truth_ids`, `k_values`). Use `?include_per_query=true` for per-query metrics.
4. **Benchmarks**  
   - `POST /benchmarks`: send `strategies`, `queries` (each with `query_id`, `query`, optional `ground_truth_ids`), `iterations`, `timeout_seconds`. Returns `benchmark_id`.  
   - `GET /benchmarks/{benchmark_id}`: check status (`pending`, `running`, `completed`, `failed`).  
   - `GET /benchmarks/{benchmark_id}/results`: get full results when status is `completed`.  
   - `DELETE /benchmarks/{benchmark_id}`: cancel a running benchmark.

Ground truth can come from human labels, an LLM-based judge (`evaluation/ground_truth_llm.py`), or another gold set. The API does not generate ground truth; you provide it.

---

## Evaluation folder: file-by-file

| File | Purpose |
|------|--------|
| **metrics.py** | Core IR metric calculation: `calculate_metrics()`, `calculate_batch_metrics()`. Takes `retrieved_ids`, `ground_truth_ids`, `k_values`; returns `EvaluationMetrics` (precision, recall, ndcg, mrr). Used by API routes and benchmarks. |
| **benchmarks.py** | Benchmark runner and report: `BenchmarkRunner`, `BenchmarkConfig`, `BenchmarkReport`, `BenchmarkQuery`. Runs multiple strategies over a list of queries (with optional ground truth), collects latency/cost per strategy, computes rankings. Used by API routes (`api/routes/benchmarks.py`) to run benchmarks in the background. |
| **datasets.py** | Dataset loading and management: `Dataset`, `DatasetQuery`, `DatasetManager`, `DatasetConfig`. Loads JSON/JSONL datasets with queries and ground truth (`query_id`, `query`, `relevant_doc_ids`, optional `relevance_scores`). Supports train/test splits, validation, and optional LLM-assisted enrichment via `evaluation.ground_truth_llm.enrich_dataset_with_llm`. Use `DatasetManager().load(path)` to load a dataset; use with benchmarks or metrics scripts. |
| **reports.py** | Markdown report generation: `ReportGenerator`, `ReportConfig`. Builds markdown reports from `BenchmarkReport`: strategy rankings, latency tables (p50/p95/p99), cost breakdown, IR metrics tables. Options: include_rankings, include_cost_breakdown, include_latency_stats, include_metrics_tables, decimal_places. Use `generate_markdown(benchmark_report)` or `save(benchmark_report, path)` for file output. |
| **html_reports.py** | HTML report generation: `HtmlReportGenerator`, `HtmlReportConfig`. Builds styled HTML reports from `BenchmarkReport` with CSS themes (light/dark), sortable tables, and optional chart placeholders. Options: theme, include_charts, include_raw_data, sortable_tables, decimal_places, company_name. Use `generate(benchmark_report)` or `save(benchmark_report, path)` for file output. |
| **ground_truth_llm.py** | LLM-assisted ground truth: `generate_ground_truth_for_query()`, `enrich_dataset_with_llm()`. Uses an LLM (e.g. gpt-4o-mini) to judge relevance of candidate documents for a query and produce `relevant_doc_ids` and optional `relevance_scores`. Use to enrich a `Dataset` when you have candidates but no labels. |
| **schema_extension.sql** | Optional SQL extensions for evaluation (e.g. tables for storing benchmark runs or evaluation results). Applied after the base schema (`strategies/utils/schema.sql`) when using `python scripts/run_schema.py` or Docker Compose. |

---

## Usage examples (programmatic)

### Calculate metrics

```python
from evaluation.metrics import calculate_metrics

metrics = calculate_metrics(
    retrieved_ids=["doc_1", "doc_3", "doc_2"],
    ground_truth_ids=["doc_1", "doc_2"],
    k_values=[3, 5, 10],
)
print(metrics.precision, metrics.recall, metrics.mrr, metrics.ndcg)
```

### Run benchmark (programmatic)

```python
from evaluation.benchmarks import BenchmarkRunner, BenchmarkConfig

runner = BenchmarkRunner()
config = BenchmarkConfig(
    strategies=["standard", "reranking", "multi_query"],
    iterations=3,
)
queries = [
    {"query_id": "q1", "query": "What is RAG?", "ground_truth_ids": ["doc_1", "doc_2"]},
]
report = await runner.run(queries, config)
print(report.summary())
```

### Load dataset and generate reports

```python
from evaluation.datasets import DatasetManager
from evaluation.reports import ReportGenerator
from evaluation.html_reports import HtmlReportGenerator

manager = DatasetManager()
dataset = await manager.load("datasets/sample/basic_queries.json")

# After running a benchmark and obtaining report:
generator = ReportGenerator()
await generator.save(report, "reports/benchmark.md")

html_gen = HtmlReportGenerator()
await html_gen.save(report, "reports/benchmark.html")
```

---

## Why evaluation might seem “missing”

- **No dedicated CLI**  
  There is no `python -m evaluation.run`. Evaluation is REST-only so the same API serves production and evaluation; use curl, scripts, or the OpenAPI UI at `/docs`.

- **Task-master / PRD**  
  The PRD scoped orchestration, evaluation, and API. The evaluation layer (metrics, benchmark runner, API routes, datasets, reports) was delivered; “how to run” is documented here and in the main README.

- **Benchmarks are async**  
  Starting a benchmark returns immediately with an ID; poll status and fetch results when done.

---

## Best practices

1. Use multiple k values (e.g. [1, 3, 5, 10]) for metrics.  
2. Run multiple benchmark iterations to reduce latency variance.  
3. Include a baseline strategy (e.g. standard) in comparisons.  
4. Consider API cost alongside accuracy when choosing strategies.

---

## References

- [README.md](../README.md#evaluation) — Evaluation section with curl examples.  
- [api/routes/evaluation.py](../api/routes/evaluation.py) — Metrics request/response models.  
- [api/routes/benchmarks.py](../api/routes/benchmarks.py) — Benchmark trigger, status, results, cancel.  
- [datasets/README.md](../datasets/README.md) — Dataset format and sample files.
