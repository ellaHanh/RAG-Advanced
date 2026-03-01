# Evaluation

<!-- TOC: Quickstart, Features, Pipeline (gold + corpus xlsx), RAGAS generation, Usage, Dependencies, Related -->

IR metrics, benchmarks, gold/corpus pipeline, and RAGAS generation evaluation. The [root README](../README.md#-evaluation) summarizes endpoints and metric reference; this file adds pipeline, RAGAS, and Python usage.

## Quickstart

```bash
# Start API then call metrics/benchmarks (see root README for curl examples)
# Or run the evaluation pipeline (gold + corpus xlsx):
python scripts/run_evaluation_pipeline.py \
  --gold path/to/gold.xlsx --corpus path/to/corpus.xlsx \
  --strategies standard reranking --out-dir ./eval_out
```

## Features

- **IR metrics** — Precision@k, Recall@k, MRR, NDCG@k (`evaluation/metrics.py`); `POST /metrics`, `POST /metrics/batch`.
- **Benchmarks** — Multi-strategy over queries; latency, cost, rankings (`evaluation/benchmarks.py`); `POST /benchmarks`, `GET /benchmarks/{id}`, `GET /benchmarks/{id}/results`.
- **Datasets** — Load JSON/JSONL with ground truth (`evaluation/datasets.py`).
- **Reports** — Markdown and HTML (`evaluation/reports.py`, `html_reports.py`).
- **Ground truth** — Optional LLM-assisted labeling (`evaluation/ground_truth_llm.py`).

## Pipeline (gold + corpus xlsx)

End-to-end: load gold and corpus xlsx → ingest corpus → map gold to chunk IDs → run benchmarks → write report.

- **Config** — Use `--config evaluation/config/bioasq_v1.json` for column mapping and answer columns (for generation eval). Omit for hardcoded BioASQ v1 defaults.
- **Options** — `--gold`, `--corpus`, `--strategies`, `--out-dir`; `--skip-ingest` to reuse DB; `--run-generation-eval` for RAGAS after retrieval.
- **Outputs** — `gold_dataset_canonical.json`, `benchmark_report.md`, `benchmark_results_detailed.json`; with `--run-generation-eval`: `generation_metrics.json`.

Evaluation uses **chunk-level** IDs (`ground_truth_chunk_ids`, `retrieved_chunk_ids`). See [docs/README_terminology.md](../docs/README_terminology.md).

## RAGAS generation evaluation

- **Input (per sample):** `question`, `contexts`, `answer`, `ground_truth`.
- **Output:** faithfulness, answer_relevancy, context_precision; optional per-sample; `ragas_llm_usage`, `ragas_embedding_usage`.
- **Location:** `evaluation/ragas_eval.py` — `evaluate_generation()`, `RagasEvaluationResult`.
- **API:** `POST /evaluate/generation` with `{ "samples": [ ... ] }`.
- **Pipeline:** `--run-generation-eval` (gold must have answer column via config).

## Usage

```python
from evaluation.metrics import calculate_metrics

metrics = calculate_metrics(
    retrieved_ids=["c1", "c3", "c2"],
    ground_truth_ids=["c1", "c2"],
    k_values=[3, 5, 10],
)
```

```python
from evaluation.ragas_eval import evaluate_generation

result = evaluate_generation([
    {"question": "What is RAG?", "contexts": ["..."], "answer": "...", "ground_truth": "..."}
])
print(result.scores)
```

## Dependencies

- Project deps (`pip install -e .`). For RAGAS: `ragas`, `datasets` (in pyproject.toml).
- Pipeline xlsx: `pandas`, `openpyxl`; optional `pyyaml` for YAML config.

## Related

- [Root README](../README.md)
- [api/](../api/README.md) — REST endpoints
- [datasets/](../datasets/README.md) — Dataset format
- [scripts/](../scripts/README.md) — `run_evaluation_pipeline.py`
