# RAG Strategy Examples (from all-rag-strategies)

These scripts are copied from [all-rag-strategies/examples](https://github.com/coleam00/ottomator-agents/tree/main/all-rag-strategies/examples). They are **reference pseudocode** for each of the 11 RAG strategies (Pydantic AI + PG Vector). They are **not runnable as-is** in RAG-Advanced: they use `psycopg2`/`pgvector.psycopg2` and placeholder functions (`get_embedding`, `cross_encoder_score`, etc.). Use them to understand each strategy’s flow.

## RAG-Advanced equivalents

RAG-Advanced has **8 runnable strategies** (7 from all-rag-strategies + **standard** baseline). See [docs/ALL_RAG_STRATEGIES_7_VS_RAG_ADVANCED.md](../../docs/ALL_RAG_STRATEGIES_7_VS_RAG_ADVANCED.md) for mapping.

| Example | RAG-Advanced status | How to run equivalent |
|---------|---------------------|------------------------|
| 01_reranking.py | **Implemented** | `POST /execute` with `"strategy": "reranking"` |
| 02_agentic_rag.py | **Implemented** | `POST /execute` with `"strategy": "agentic"` (chunks + full doc for top) |
| 03_knowledge_graphs.py | Not yet (pseudocode only in all-rag-strategies) | — |
| 04_contextual_retrieval.py | **Implemented** | `POST /execute` with `"strategy": "contextual_retrieval"` (use with ingestion `--contextual`) |
| 05_query_expansion.py | **Implemented** | `POST /execute` with `"strategy": "query_expansion"` (expand then single search) |
| 06_multi_query_rag.py | **Implemented** | `POST /execute` with `"strategy": "multi_query"` |
| 07_context_aware_chunking.py | **Implemented** | `POST /execute` with `"strategy": "context_aware_chunking"` (chunking at ingestion) |
| 08_late_chunking.py | Not yet (pseudocode only) | — |
| 09_hierarchical_rag.py | Not yet (pseudocode only) | — |
| 10_self_reflective_rag.py | **Implemented** | `POST /execute` with `"strategy": "self_reflective"` |
| 11_fine_tuned_embeddings.py | Not yet (pseudocode only) | — |

**Standard semantic search** (no example number): **Implemented** — `POST /execute` with `"strategy": "standard"` (baseline vector search).

See [docs/MIGRATION_AND_TASKMASTER_NOTES.md](../../docs/MIGRATION_AND_TASKMASTER_NOTES.md) for why these 11 strategies and example documents were not part of the task-master PRD tasks.

---

## Single-strategy run (API)

Each implemented strategy can be run alone via `POST /execute`:

| Strategy | Description | Optional body params |
|----------|-------------|------------------------|
| **standard** | Vector search only (fast, baseline) | `limit` (default 5) |
| **reranking** | Vector search → cross-encoder rerank | `limit`, `initial_k` (default 20), `final_k` (default 5) |
| **multi_query** | LLM expansion → parallel search → dedupe | `limit`, `num_variations` (default 3) |
| **query_expansion** | LLM expansion → single search (cheaper than multi_query) | `limit`, `num_variations` (default 1) |
| **self_reflective** | Search → grade (LLM) → refine query if low → search again | `limit`, `relevance_threshold` (default 3) |
| **agentic** | Vector search + full document for top result | `limit` |
| **contextual_retrieval** | Same as standard; for contextually enriched chunks (ingestion `--contextual`) | `limit` |
| **context_aware_chunking** | Same as standard; chunking at ingestion (Docling) | `limit` |

Example:

```json
POST /execute
{ "query": "What is RAG?", "strategy": "self_reflective", "limit": 5 }
```

---

## Strategy chaining

Chains run strategies **sequentially**. Each step receives the **same query**; the next step does **not** receive the previous step’s documents (each step runs retrieval from scratch). The chain returns the **last step’s** documents and aggregates latency/cost.

- **All combinations** of the eight strategies are valid (same query per step; result = last step’s documents). **Recommended chains**: see [root README — Strategy Guide](../../README.md#-strategy-guide) and [Chain Strategies](../../README.md#chain-strategies).

Per-step config (e.g. `limit`, `initial_k`, `final_k`, `num_variations`) can be set in the chain request; each step’s config is passed through to that strategy.

Example:

```json
POST /chain
{
  "query": "How does retrieval-augmented generation work?",
  "steps": [
    { "strategy": "multi_query", "config": { "limit": 10, "num_variations": 3 } },
    { "strategy": "reranking", "config": { "limit": 5, "initial_k": 20, "final_k": 5 } }
  ],
  "continue_on_error": false
}
```

Fallback: set `fallback_strategy` on a step (e.g. `"strategy": "multi_query", "fallback_strategy": "standard"`) to run the fallback if the primary fails.

---

## Reference: script → strategy (all-rag-strategies)

The `.py` files in this folder are **conceptual pseudocode** (Pydantic AI + pgvector; placeholders like `get_embedding`, `cross_encoder_score`). They are not runnable in RAG-Advanced. For strategy concepts and when to use each, see [strategies/docs/](../docs/README.md).

| Script | Strategy concept |
|--------|-------------------|
| 01_reranking.py | Two-stage retrieval, cross-encoder |
| 02_agentic_rag.py | Agent tools (vector, SQL, web) |
| 03_knowledge_graphs.py | Vector + graph |
| 04_contextual_retrieval.py | Document context on chunks |
| 05_query_expansion.py | Query variations, single search |
| 06_multi_query_rag.py | Parallel reformulated queries |
| 07_context_aware_chunking.py | Semantic chunk boundaries |
| 08_late_chunking.py | Embed full doc then chunk (Jina) |
| 09_hierarchical_rag.py | Parent-child chunks |
| 10_self_reflective_rag.py | Grade → refine → search again |
| 11_fine_tuned_embeddings.py | Domain-specific embeddings |
