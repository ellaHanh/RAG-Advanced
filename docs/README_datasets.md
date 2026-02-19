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

## xlsx (gold and corpus)

Evaluation can use **Excel (.xlsx) files** for gold (query + relevant IDs) and corpus (documents to search). Column names are configurable so different sources (e.g. BioASQ) can be used.

- **Gold xlsx**: One row per query. Required columns (or mapped names): query id, query text, and a column containing the list of relevant document/passage IDs (string parsed as JSON, pipe-, or comma-separated).
- **Corpus xlsx**: One row per document/passage. Required columns (or mapped): doc id, text (passage), and optionally title.

**Default (BioASQ v1, first sheet):**

- Gold: `id`, `question`, `relevant_passage_ids`.
- Corpus: `doc_id`, `passage`, `title`.

Use the **evaluation pipeline** to load both, ingest the corpus, and run benchmarks: see [README_evaluation_pipeline.md](README_evaluation_pipeline.md). Column mapping can be set via config file (`evaluation/config/bioasq_v1.json`) or CLI (`--gold-map`, `--corpus-map`). Loaders live in `evaluation/loaders/` (`xlsx_loader.py`, `xlsx_config.py`).

---

## Sample datasets

- **sample/basic_queries.json**: Example queries and ground truth in the expected format. **Origin**: Created for RAG-Advanced (not from all-rag-strategies; that repo has no evaluation datasets). The `relevant_doc_ids` in this file (e.g. `doc_ml_intro`, `doc_ml_basics`) are **placeholder IDs** — they do not refer to real documents in your database. See [datasets/sample/README.md](../datasets/sample/README.md) for how to use the sample and [datasets/sample/PLACEHOLDER_IDS.md](../datasets/sample/PLACEHOLDER_IDS.md) for a mapping of each placeholder ID to a short description. For real evaluation, replace placeholders with actual document or chunk IDs from your DB after ingestion.

**Use in pipelines/tests:** This dataset is **not** used to build or test the evaluation or integration pipelines. The evaluation pipeline gets queries from the API request; tests use in-memory fixtures and temporary files, not `basic_queries.json`. The file is a format example and for manual/demo use only.

---

## Best practices

1. Use at least ~20 queries for meaningful aggregate metrics.
2. Where possible, include at least a few relevant docs per query for recall/NDCG.
3. Document annotation guidelines so labels stay consistent.
4. Use `category` if you need stratified splits (e.g. by domain or difficulty).
