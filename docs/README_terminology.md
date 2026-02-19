# Terminology: chunk_id vs document_id

This doc defines how we use **chunk_id** and **document_id** so code and docs stay consistent.

## Database (PostgreSQL)

| Term | Meaning | Where |
|------|--------|--------|
| **document_id** | UUID of a row in the `documents` table. Primary key of a stored document. | `documents.id`, `chunks.document_id` (FK) |
| **chunk_id** | UUID of a row in the `chunks` table. Primary key of a chunk; each chunk belongs to one document via `chunks.document_id`. | `chunks.id` |

One document can have many chunks; each chunk has one `document_id`.

## Retrieval and API

- **Strategies** return a list of **chunks** (from `match_chunks`). Each item has `chunks.id` (chunk_id) and `chunks.document_id` (document_id).
- In **orchestration**, `ExecutionResult.documents` is a list of `Document`; each `Document.id` is set to the **chunk_id** (from `chunks.id`) in this codebase. So `result.document_ids` in practice are chunk IDs when results come from the DB. The name “document” is kept for API shape; the value is the chunk primary key.

## Evaluation (benchmarks, metrics)

Evaluation is **chunk-level**: we compare which chunks were retrieved vs which chunks are relevant.

| Term | Meaning | Used in |
|------|--------|--------|
| **ground_truth_chunk_ids** | List of relevant chunk IDs (from `chunks.id`) for a query. Built by mapping gold document IDs to chunk IDs after ingestion. | `BenchmarkQuery`, metrics input, `benchmark_results_detailed.json` |
| **retrieved_chunk_ids** | Ranked list of chunk IDs returned by a strategy (same as `chunks.id`). | `StrategyResult`, metrics input, `benchmark_results_detailed.json` |

So in evaluation code and JSON:

- Use **chunk_id** / **\*_chunk_ids** for any ID that comes from `chunks.id` (ground truth list, retrieved list).
- Use **document_id** / **doc_id** when referring to the parent document (e.g. gold `relevant_doc_ids`, corpus doc_id, `chunks.document_id`).

## Quick reference

| Context | ID type | Example variable / field |
|--------|---------|----------------------------|
| Gold dataset (from corpus) | document_id | `relevant_doc_ids` |
| After mapping gold → chunks | chunk_id | `ground_truth_chunk_ids` |
| Strategy result (from DB) | chunk_id | `retrieved_chunk_ids`, `ExecutionResult.document_ids` (semantically chunk IDs) |
| DB columns | document_id / chunk_id | `chunks.document_id`, `chunks.id` |
