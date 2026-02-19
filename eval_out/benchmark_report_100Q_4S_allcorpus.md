# RAG Strategy Benchmark Report

## Overview

- **Benchmark ID**: `f6a568c2-e9b7-4e83-9261-e967b1bfc4f8`
- **Total Queries**: 100
- **Total Executions**: 400
- **Duration**: 328.78s
- **Strategies**: standard, reranking, multi_query, query_expansion
- **Iterations**: 1

## IR Metrics

### Precision@k

| Strategy | P@3 | P@5 | P@10 |
| --- | ---: | ---: | ---: |
| standard | 0.227 | 0.180 | 0.127 |
| reranking | 0.240 | 0.192 | 0.096 |
| multi_query | 0.307 | 0.228 | 0.159 |
| query_expansion | 0.243 | 0.180 | 0.119 |

### Recall@k

| Strategy | R@3 | R@5 | R@10 |
| --- | ---: | ---: | ---: |
| standard | 0.164 | 0.184 | 0.230 |
| reranking | 0.173 | 0.206 | 0.206 |
| multi_query | 0.268 | 0.289 | 0.344 |
| query_expansion | 0.178 | 0.199 | 0.226 |

### NDCG@k

| Strategy | NDCG@3 | NDCG@5 | NDCG@10 |
| --- | ---: | ---: | ---: |
| standard | 0.306 | 0.292 | 0.285 |
| reranking | 0.316 | 0.304 | 0.262 |
| multi_query | 0.415 | 0.397 | 0.398 |
| query_expansion | 0.314 | 0.296 | 0.282 |

### Mean Reciprocal Rank (MRR)

| Strategy | MRR |
| --- | ---: |
| standard | 0.428 |
| reranking | 0.427 |
| multi_query | 0.566 |
| query_expansion | 0.421 |

## Latency Statistics

| Strategy | P50 (ms) | P95 (ms) | P99 (ms) | Mean (ms) | Std Dev |
| --- | ---: | ---: | ---: | ---: | ---: |
| standard | 72 | 92 | 112 | 74 | 12.9 |
| reranking | 288 | 425 | 605 | 331 | 258.9 |
| multi_query | 1338 | 2015 | 2475 | 1400 | 342.3 |
| query_expansion | 1347 | 2469 | 3163 | 1480 | 449.8 |

## Cost Analysis

| Strategy | Total Cost ($) | Avg Cost/Query ($) | Success Rate |
| --- | ---: | ---: | ---: |
| standard | 0.0014 | 0.0000 | 100.0% |
| reranking | 0.0014 | 0.0000 | 100.0% |
| multi_query | 0.0240 | 0.0002 | 100.0% |
| query_expansion | 0.0192 | 0.0002 | 100.0% |

**Total Cost**: $0.0461

## Strategy Rankings

### By Latency (Lower is Better)

- **latency_p50**: standard → reranking → multi_query → query_expansion
- **latency_p95**: standard → reranking → multi_query → query_expansion

### By IR Metrics (Higher is Better)

- **mrr**: multi_query → standard → reranking → query_expansion
- **precision@3**: multi_query → query_expansion → reranking → standard
- **ndcg@3**: multi_query → reranking → query_expansion → standard
- **precision@5**: multi_query → reranking → standard → query_expansion
- **ndcg@5**: multi_query → reranking → query_expansion → standard
- **precision@10**: multi_query → standard → query_expansion → reranking
- **ndcg@10**: multi_query → standard → query_expansion → reranking

### By Cost (Lower is Better)

- **cost**: standard → reranking → query_expansion → multi_query

---
*Report generated on 2026-02-14 19:03:19*
