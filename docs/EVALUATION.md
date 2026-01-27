# Evaluation Guide

> **Note**: This documentation will be fully populated when the evaluation module is implemented.

## Overview

RAG-Advanced provides comprehensive evaluation capabilities for comparing RAG strategies using standard Information Retrieval (IR) metrics.

## Metrics

### Precision@k

Measures the proportion of relevant documents in the top-k results.

```
Precision@k = (relevant docs in top-k) / k
```

**Interpretation:**
- 1.0 = All top-k docs are relevant
- 0.5 = Half of top-k docs are relevant
- Useful for: Precision-critical applications

### Recall@k

Measures the proportion of all relevant documents retrieved in top-k.

```
Recall@k = (relevant docs in top-k) / (total relevant docs)
```

**Interpretation:**
- 1.0 = All relevant docs retrieved
- 0.5 = Half of relevant docs retrieved
- Useful for: High-recall applications (research, legal)

### Mean Reciprocal Rank (MRR)

Average of the reciprocal ranks of the first relevant result.

```
MRR = mean(1 / rank_of_first_relevant)
```

**Interpretation:**
- 1.0 = First result always relevant
- 0.5 = First relevant result averages rank 2
- Useful for: Single-answer queries

### Normalized Discounted Cumulative Gain (NDCG@k)

Measures ranking quality with graded relevance.

**Interpretation:**
- 1.0 = Perfect ranking
- Accounts for position and relevance level
- Useful for: When document relevance varies

## Usage

### Calculate Metrics

```python
from evaluation.metrics import calculate_metrics

metrics = calculate_metrics(
    retrieved_ids=["doc_1", "doc_3", "doc_2"],
    ground_truth_ids=["doc_1", "doc_2"],
    k_values=[3, 5, 10]
)

print(f"Precision@3: {metrics.precision[3]}")
print(f"Recall@5: {metrics.recall[5]}")
print(f"MRR: {metrics.mrr}")
print(f"NDCG@10: {metrics.ndcg[10]}")
```

### Run Benchmark

```python
from evaluation.benchmarks import BenchmarkRunner

runner = BenchmarkRunner()
results = await runner.run(
    strategies=["standard", "reranking", "multi_query"],
    test_dataset="datasets/sample/basic_queries.json",
    iterations=3
)

print(results.summary())
```

## Benchmark Reports

Reports are generated in multiple formats:

- **Markdown**: Human-readable tables
- **HTML**: Styled interactive reports
- **JSON**: Machine-readable data

## Best Practices

1. **Use multiple k values**: [3, 5, 10] covers different use cases
2. **Run multiple iterations**: Reduces variance in latency measurements
3. **Include baseline**: Always compare against standard search
4. **Consider cost**: Factor in API costs alongside accuracy
