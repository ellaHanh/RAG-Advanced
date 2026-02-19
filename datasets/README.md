# Evaluation Datasets

This directory contains test datasets with ground truth for evaluating RAG strategies.

## Dataset Format

Each dataset should be a JSON or JSONL file with the following structure:

```json
{
  "metadata": {
    "name": "dataset_name",
    "description": "Description of the dataset",
    "created_at": "2026-01-22T00:00:00Z",
    "version": "1.0"
  },
  "queries": [
    {
      "query_id": "q1",
      "query": "What is machine learning?",
      "relevant_doc_ids": ["doc_1", "doc_2"],
      "relevance_scores": {
        "doc_1": 2,
        "doc_2": 1
      }
    }
  ]
}
```

## Relevance Scores

- **0**: Not relevant
- **1**: Partially relevant
- **2**: Highly relevant

## Sample datasets

### sample/

Contains example datasets for testing:

- **basic_queries.json** — Simple factual queries. **Not from all-rag-strategies**; created for RAG-Advanced. The `relevant_doc_ids` (e.g. `doc_ml_intro`, `doc_ml_basics`) are **placeholder IDs** for schema demonstration; they do not point to real docs in your DB. To see what each ID represents, see [sample/PLACEHOLDER_IDS.md](sample/PLACEHOLDER_IDS.md). To run evaluation against your corpus, replace them with actual document or chunk IDs from your database. See [sample/README.md](sample/README.md) for details.
- Other sample files (e.g. `complex_queries.json`, `ambiguous_queries.json`) may be added; use the same JSON format.

## Creating Your Own Dataset

1. Prepare your queries
2. Run retrieval across strategies
3. Manually label relevance or use LLM-assisted annotation
4. Validate with the dataset manager in `evaluation.datasets`.

## Usage

```python
from evaluation.datasets import DatasetManager

manager = DatasetManager()
dataset = await manager.load("datasets/sample/basic_queries.json")

# Split into train/test
train, test = manager.split(dataset, test_ratio=0.2)
```

## Best Practices

1. **Minimum 20 queries** for meaningful evaluation
2. **At least 3 relevant docs per query** for NDCG calculation
3. **Include edge cases** (no relevant docs, many relevant docs)
4. **Document annotation guidelines** for consistency
