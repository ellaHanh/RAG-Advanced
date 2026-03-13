# Why RAG-Advanced Had 2 of 7 Strategies (and Where “Standard” Came From)

This doc answers: (1) which strategies in [all-rag-strategies README](https://github.com/coleam00/ottomator-agents/blob/main/all-rag-strategies/README.md) have **full code** in `implementation/`, (2) why RAG-Advanced initially had only **2** of those 7 (excluding “standard”), and (3) where the **standard** strategy came from.

---

## The 7 strategies with “Code Example” in all-rag-strategies

The all-rag-strategies README table lists **11** strategies. Of those, **7** are marked **✅ Code Example** (full code in `implementation/`, mainly `rag_agent_advanced.py`):

| # | Strategy              | Status in all-rag-strategies |
|---|------------------------|------------------------------|
| 1 | Re-ranking             | ✅ Code Example              |
| 2 | Agentic RAG            | ✅ Code Example              |
| 4 | Contextual Retrieval   | ✅ Code Example              |
| 5 | Query Expansion        | ✅ Code Example              |
| 6 | Multi-Query RAG        | ✅ Code Example              |
| 7 | Context-Aware Chunking | ✅ Code Example              |
| 10| Self-Reflective RAG    | ✅ Code Example              |

The other 4 (Knowledge Graphs, Late Chunking, Hierarchical RAG, Fine-tuned Embeddings) are **📝 Pseudocode Only** in `examples/`.

So there are **7** strategies with full implementation in all-rag-strategies. “Standard” is **not** one of them.

---

## Why RAG-Advanced only had 2 of those 7 (at first)

1. **PRD scope**  
   The RAG-Advanced PRD focused on **orchestration, evaluation, and API** (registry, execution, chaining, metrics, benchmarks, REST). It did **not** require implementing all 7 strategies; it required a **pluggable** strategy layer.

2. **Incremental adoption**  
   Strategies were added one by one. Initially only **reranking** and **multi_query** (plus the baseline “standard”) were implemented so that:
   - Single-strategy and chaining worked end-to-end.
   - The rest could be added later without changing the orchestration design.

3. **Where the 5 others live in all-rag-strategies**  
   - **Contextual Retrieval** and **Context-Aware Chunking** are largely **ingestion-time** (contextual enrichment, Docling HybridChunker). Query-time “retrieval” is still vector search over the same chunks.
   - **Agentic RAG** there is the **agent** choosing among tools (search_knowledge_base, retrieve_full_document, etc.), not a single retrieval strategy. In RAG-Advanced we don’t have an in-loop agent; we implement “agentic” as a **strategy** (e.g. LLM picks a strategy, or we composite vector + full-doc retrieval).
   - **Query Expansion** and **Self-Reflective RAG** are retrieval strategies that fit our interface; they were simply not implemented yet.

So “only 2 of 7” was a **scope and ordering** choice, not a technical limit. The same orchestration supports all 7 once implemented.

---

## Where “standard” came from

In all-rag-strategies, the README table lists only **advanced** strategies. The **baseline** retrieval is implemented as `search_knowledge_base` in `rag_agent_advanced.py`: one query → embed → vector search over chunks. That’s the same as what every RAG system does by default.

RAG-Advanced gave that baseline a **name** and made it a first-class strategy:

- **Name**: `standard`
- **Behavior**: Single query → embed → `match_chunks` in PostgreSQL → return top‑k chunks.

So **standard** is the RAG-Advanced name for “plain semantic search over chunks.” It was added so the orchestration layer (single run, chaining, comparison) always has a simple, fast baseline strategy, and so the API can expose “run standard” explicitly. It was not “missing” from all-rag-strategies; it’s the unnamed baseline there and a named strategy in RAG-Advanced.

---

## Summary

| Question | Answer |
|----------|--------|
| How many strategies in all-rag-strategies have full code? | **7** (Re-ranking, Agentic RAG, Contextual Retrieval, Query Expansion, Multi-Query RAG, Context-Aware Chunking, Self-Reflective RAG). |
| Why did RAG-Advanced only have 2 of those 7? | PRD focused on orchestration/API/evaluation; strategies were added incrementally. Only reranking and multi_query were implemented first. |
| Where did “standard” come from? | It’s the RAG-Advanced name for the baseline in all-rag-strategies (`search_knowledge_base`): plain vector search. Added so orchestration has a named, fast default strategy. |

RAG-Advanced now implements **all 7** of the all-rag-strategies “Code Example” strategies plus **standard** for baseline retrieval:

| all-rag-strategies (7 with code) | RAG-Advanced strategy name | Notes |
|-----------------------------------|----------------------------|-------|
| Re-ranking | `reranking` | Two-stage vector + cross-encoder |
| Agentic RAG | `agentic` | Chunks + full document for top result |
| Contextual Retrieval | `contextual_retrieval` | Same as standard; use when ingestion used `--contextual` |
| Query Expansion | `query_expansion` | Expand then single search (cheaper than multi_query) |
| Multi-Query RAG | `multi_query` | Expand + parallel search + dedupe |
| Context-Aware Chunking | `context_aware_chunking` | Same as standard; chunking at ingestion (Docling) |
| Self-Reflective RAG | `self_reflective` | Search → grade → refine query if low → search again |
| (baseline) | `standard` | Plain vector search (RAG-Advanced name for baseline) |
