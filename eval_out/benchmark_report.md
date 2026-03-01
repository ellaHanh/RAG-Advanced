# RAG Strategy Benchmark Report

## Overview

- **Benchmark ID**: `18761b07-b9fe-45b3-bf12-12e789ab1dd7`
- **Total Queries**: 2
- **Total Executions**: 2
- **Duration**: 0.14s
- **Strategies**: standard
- **Iterations**: 1

## IR Metrics

### Precision@k

| Strategy | P@3 | P@5 | P@10 |
| --- | ---: | ---: | ---: |
| standard | 0.333 | 0.200 | 0.100 |

### Recall@k

| Strategy | R@3 | R@5 | R@10 |
| --- | ---: | ---: | ---: |
| standard | 1.000 | 1.000 | 1.000 |

### NDCG@k

| Strategy | NDCG@3 | NDCG@5 | NDCG@10 |
| --- | ---: | ---: | ---: |
| standard | 1.000 | 1.000 | 1.000 |

### Mean Reciprocal Rank (MRR)

| Strategy | MRR |
| --- | ---: |
| standard | 1.000 |

## Latency Statistics

| Strategy | P50 (ms) | P95 (ms) | P99 (ms) | Mean (ms) | Std Dev |
| --- | ---: | ---: | ---: | ---: | ---: |
| standard | 71 | 79 | 80 | 71 | 12.7 |

## Cost Analysis

| Strategy | Total Cost ($) | Avg Cost/Query ($) | Success Rate |
| --- | ---: | ---: | ---: |
| standard | 0.0000 | 0.0000 | 100.0% |

**Total Cost**: $0.0000

## Strategy Rankings

### By Latency (Lower is Better)

- **latency_p50**: standard
- **latency_p95**: standard

### By IR Metrics (Higher is Better)

- **mrr**: standard
- **precision@3**: standard
- **ndcg@3**: standard
- **precision@5**: standard
- **ndcg@5**: standard
- **precision@10**: standard
- **ndcg@10**: standard

### By Cost (Lower is Better)

- **cost**: standard

---
*Report generated on 2026-02-24 15:27:20*
