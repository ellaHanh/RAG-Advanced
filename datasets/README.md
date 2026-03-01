# Datasets

Evaluation datasets: query sets with ground truth (relevant document/chunk IDs, optional relevance scores) for IR metrics and benchmarks.

## Quickstart

```bash
# No commands — store JSON/JSONL here. Load in code or pass to pipeline.
# Sample format: datasets/sample/basic_queries.json
```

```python
from evaluation.datasets import DatasetManager

manager = DatasetManager()
dataset = await manager.load("datasets/sample/basic_queries.json")
train, test = manager.split(dataset, train_ratio=0.8)
```

## Features

- **Format** — JSON/JSONL: `name`, `description`, `queries` (each with `query_id`, `query`, `relevant_doc_ids`, optional `relevance_scores`). API/benchmarks use `ground_truth_ids` (same concept).
- **Relevance scores** — 0 = not relevant, 1 = partial, 2 = highly relevant (NDCG).
- **Gold/corpus xlsx** — Column mapping via config or CLI; see [evaluation/README.md](../evaluation/README.md) (pipeline).

## Usage

Dataset shape (see `evaluation/datasets.py` — `Dataset`, `DatasetQuery`):

```json
{
  "name": "dataset_name",
  "description": "Optional",
  "queries": [
    {
      "query_id": "q1",
      "query": "What is machine learning?",
      "relevant_doc_ids": ["doc_1", "doc_2"],
      "relevance_scores": { "doc_1": 2, "doc_2": 1 },
      "category": "optional",
      "metadata": {}
    }
  ],
  "metadata": {}
}
```

**sample/** — `basic_queries.json` is a format example. Its `relevant_doc_ids` are **placeholder IDs** (not real DB IDs). For real evaluation, replace with actual document or chunk IDs after ingestion. Placeholder mapping: [sample/PLACEHOLDER_IDS.md](sample/PLACEHOLDER_IDS.md). This file is not used by the evaluation pipeline or tests at runtime; pipeline gets queries from the request; tests use in-memory fixtures.

## Dependencies

None for folder; loading uses `evaluation.datasets` (project deps).

## Related

- [Root README](../README.md)
- [evaluation/](../evaluation/README.md) — Metrics, benchmarks, pipeline
- [sample/PLACEHOLDER_IDS.md](sample/PLACEHOLDER_IDS.md) — Placeholder ID descriptions
