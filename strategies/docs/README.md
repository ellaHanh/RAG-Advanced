# Strategy Reference Docs (from all-rag-strategies)

These markdown files are copied from [all-rag-strategies/docs](https://github.com/coleam00/ottomator-agents/tree/main/all-rag-strategies/docs). They describe each of the 11 RAG strategies conceptually: what they are, when to use them, pros/cons, and simple code patterns.

## Files

| File | Strategy |
|------|----------|
| 01-reranking.md | Two-stage retrieval with cross-encoder |
| 02-agentic-rag.md | Agent autonomously chooses tools |
| 03-knowledge-graphs.md | Vector + graph relationships |
| 04-contextual-retrieval.md | Anthropic: add document context to chunks |
| 05-query-expansion.md | Multiple query variations |
| 06-multi-query-rag.md | Parallel searches with reformulations |
| 07-context-aware-chunking.md | Semantic chunk boundaries |
| 08-late-chunking.md | Embed full doc, then chunk (Jina) |
| 09-hierarchical-rag.md | Parent-child chunks |
| 10-self-reflective-rag.md | Iterative refinement with grading |
| 11-fine-tuned-embeddings.md | Domain-specific embeddings |

## RAG-Advanced implementation status

- **Implemented and registered**: `standard`, `reranking` (see `strategies/agents/`).
- **Planned**: Other strategies can be added incrementally; the orchestration layer (registry, chain, compare, evaluation) is strategy-agnostic.

See [docs/README_ingestion.md](../../docs/README_ingestion.md) for ingestion and [README](../../README.md) for API usage.
