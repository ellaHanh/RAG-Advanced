# Datasets Folder

The `datasets/` folder holds **evaluation datasets**: query sets with ground truth (relevant document/chunk IDs and optional relevance scores) for computing IR metrics and running benchmarks.

---

## Purpose

- Store JSON/JSONL dataset files that define queries and their relevant doc IDs.
- Used by the evaluation layer (`evaluation/datasets.py`) to load datasets for batch metrics or benchmark runs.
- Sample data under `datasets/sample/` provides a minimal format example (e.g. `basic_queries.json`).

---

## Dataset format

Each dataset file should follow this structure (see also `evaluation/datasets.py` — `Dataset`, `DatasetQuery`):

```json
{
  "name": "dataset_name",
  "description": "Optional description",
  "queries": [
    {
      "query_id": "q1",
      "query": "What is machine learning?",
      "relevant_doc_ids": ["doc_1", "doc_2"],
      "relevance_scores": {
        "doc_1": 2,
        "doc_2": 1
      },
      "category": "optional_category",
      "metadata": {}
    }
  ],
  "metadata": {}
}
```

- **query_id**: Unique identifier for the query.
- **query**: Query text.
- **relevant_doc_ids**: List of document or chunk IDs considered relevant (ground truth).
- **relevance_scores** (optional): Graded relevance per doc: 0 = not relevant, 1 = partially relevant, 2 = highly relevant. Used for NDCG-style metrics.
- **category** (optional): For stratified train/test splits.
- **metadata** (optional): Extra fields per query or at dataset level.

Some code or docs may use `relevant_doc_ids`; the evaluation API and benchmarks use `ground_truth_ids` in request bodies — these are the same concept (list of relevant IDs).

---

## Loading datasets (code)

Use `evaluation.datasets` (module name is `datasets.py`, not `dataset_manager`):

```python
from evaluation.datasets import DatasetManager, Dataset

manager = DatasetManager()
dataset = await manager.load("datasets/sample/basic_queries.json")

# Train/test split
train, test = manager.split(dataset, train_ratio=0.8)
```

For LLM-assisted ground truth generation, see `evaluation/ground_truth_llm.py` and [README_EVALUATION.md](README_EVALUATION.md).

---

## Sample datasets

- **sample/basic_queries.json**: Example queries and ground truth in the expected format. Use as a template or for quick sanity checks.

---

## Best practices

1. Use at least ~20 queries for meaningful aggregate metrics.
2. Where possible, include at least a few relevant docs per query for recall/NDCG.
3. Document annotation guidelines so labels stay consistent.
4. Use `category` if you need stratified splits (e.g. by domain or difficulty).
