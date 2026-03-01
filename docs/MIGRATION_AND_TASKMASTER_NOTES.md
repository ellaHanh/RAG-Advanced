# Migration and Task-Master Notes

This document explains why the **11 RAG strategies** and **example documents** from [all-rag-strategies](https://github.com/coleam00/ottomator-agents/tree/main/all-rag-strategies) were not explicitly included in the RAG-Advanced task-master PRD tasks, and how they are being adopted in RAG-Advanced.

---

## Why the 11 strategies were not in the task-master tasks

1. **PRD scope was orchestration and evaluation, not strategy implementations**  
   The RAG-Advanced PRD (see `PRD/.taskmaster/docs/prd.txt`) focuses on:
   - Strategy **registry** (type-safe lookup, metadata)
   - **Execution** of strategies (single, chain, parallel)
   - **Cost tracking** and pricing
   - **Evaluation** (IR metrics, benchmarks, reports)
   - **REST API** (auth, rate limiting, routes)

   The PRD states that “Existing solutions like all-rag-strategies provide individual strategy implementations but lack the orchestration, evaluation, and API layers.” So the **tasks** were defined to build the orchestration and evaluation layers, not to re-implement each of the 11 strategies from scratch.

2. **Strategies are pluggable**  
   The design assumes strategies are **registered** with the registry and executed via a common interface (`ExecutionContext` → `list[Document]`). The task-master did not enumerate “migrate strategy 1, 2, … 11” because the PRD did not require a fixed set of strategies; it required a **mechanism** to run any strategy. Migrating the 11 strategies is therefore an **incremental, post-PRD** activity: add one strategy at a time (e.g. standard, reranking, then multi-query, etc.) as needed.

3. **All 7 “Code Example” strategies from all-rag-strategies are now implemented**  
   RAG-Advanced implements and registers:
   - **standard** (baseline semantic vector search; not in the all-rag-strategies table)
   - **reranking**, **multi_query**, **query_expansion**, **self_reflective**, **agentic**, **contextual_retrieval**, **context_aware_chunking** (the 7 strategies with full code in all-rag-strategies)

   See [docs/ALL_RAG_STRATEGIES_7_VS_RAG_ADVANCED.md](ALL_RAG_STRATEGIES_7_VS_RAG_ADVANCED.md) for the mapping and why “standard” exists. The remaining 4 (knowledge graphs, late chunking, hierarchical, fine-tuned embeddings) are pseudocode-only in all-rag-strategies and can be added later if needed.

---

## Why the example documents were not in the task-master tasks

1. **Ingestion was out of initial scope**  
   The PRD capability tree emphasizes **strategy execution**, **orchestration**, **evaluation**, and **API**. Document **ingestion** (reading files, chunking, embedding, writing to PostgreSQL) was not a top-level capability in the task breakdown. The README originally listed “Ingest Documents (Coming Soon).” Ingestion was implemented later so that RAG-Advanced could run end-to-end with real data.

2. **Example documents are reference data**  
   The [all-rag-strategies/implementation/documents](https://github.com/coleam00/ottomator-agents/tree/main/all-rag-strategies/implementation/documents) folder (PDFs, DOCX, MD, audio) is **reference content** for testing ingestion and retrieval. Copying these into RAG-Advanced (`RAG-Advanced/documents/`) was a **convenience and alignment** choice so the project is self-contained and can run ingestion without depending on the all-rag-strategies repo. The task-master did not include “copy N documents” because the PRD did not specify a particular dataset; it specified evaluation **metrics** and **benchmark** flows, not a mandatory corpus.

3. **Where they live now**  
   - **RAG-Advanced/documents/** — Default folder for ingestion; contains copies of the all-rag-strategies example documents.
   - **RAG-Advanced/strategies/examples/** — Contains the 11 pseudocode example scripts from all-rag-strategies (reference only; not runnable as-is in RAG-Advanced).
   - **RAG-Advanced/strategies/docs/** — Contains the 11 strategy markdown docs from all-rag-strategies (concepts, when to use, pros/cons).

---

## Summary

| Item | In task-master? | Reason | Current state in RAG-Advanced |
|------|-----------------|--------|-------------------------------|
| 11 strategy **implementations** | No | PRD focused on orchestration/API/evaluation; strategies are pluggable | All 7 “Code Example” strategies + standard implemented; 4 pseudocode-only remain |
| 11 strategy **docs** (markdown) | No | Reference material from all-rag-strategies | Copied to `strategies/docs/` |
| 11 strategy **examples** (scripts) | No | Pseudocode for learning; not part of PRD deliverables | Copied to `strategies/examples/` with README |
| Example **documents** (PDF, DOCX, etc.) | No | Ingestion was “Coming Soon”; no mandated dataset in PRD | Copied to `documents/` as default ingestion folder |

The task-master tasks were scoped to **orchestration, evaluation, and API**. The 11 strategies and example documents are adopted from all-rag-strategies **alongside** that scope to maximize alignment and to give RAG-Advanced a clear path to full strategy coverage and self-contained ingestion.

**Evaluation**: Evaluation (IR metrics and benchmarks) was in the PRD and is implemented. How to run it (curl examples, benchmark flow) is documented in the main [README.md](../README.md#evaluation) and in [evaluation/README.md](../evaluation/README.md). There is no separate CLI for evaluation; it is exposed only via the REST API.
