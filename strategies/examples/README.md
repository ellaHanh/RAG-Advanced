# RAG Strategy Examples (from all-rag-strategies)

These scripts are copied from [all-rag-strategies/examples](https://github.com/coleam00/ottomator-agents/tree/main/all-rag-strategies/examples). They are **reference pseudocode** for each of the 11 RAG strategies (Pydantic AI + PG Vector). They are **not runnable as-is** in RAG-Advanced: they use `psycopg2`/`pgvector.psycopg2` and placeholder functions (`get_embedding`, `cross_encoder_score`, etc.). Use them to understand each strategy’s flow.

## RAG-Advanced equivalents

All **7** strategies with “Code Example” in [all-rag-strategies README](https://github.com/coleam00/ottomator-agents/blob/main/all-rag-strategies/README.md) are implemented in RAG-Advanced, plus **standard** (baseline). See [docs/ALL_RAG_STRATEGIES_7_VS_RAG_ADVANCED.md](../../docs/ALL_RAG_STRATEGIES_7_VS_RAG_ADVANCED.md) for why we had 2 of 7 initially and where “standard” came from.

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

- **All combinations** of the eight strategies are valid in a chain (each step runs with the same query; result is the last step’s documents).
- **Recommended** orderings:
  - **Recall then precision**: `[multi_query, reranking]` or `[query_expansion, reranking]`.
  - **Fast then precise**: `[standard, reranking]`.
  - **Self-correcting**: `[self_reflective]` or `[self_reflective, reranking]`.
  - **Full context then chunks**: `[agentic]` (single strategy; returns full doc + chunks).
  - **contextual_retrieval** / **context_aware_chunking**: use when ingestion was done with `--contextual` or Docling; chain like standard.

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

Fallback: you can set `fallback_strategy` on a step so that if the primary strategy fails, the fallback runs (e.g. `"strategy": "multi_query", "fallback_strategy": "standard"`).

## Framework Overview (all-rag-strategies original)

- **Pydantic AI**: Python agent framework with `@agent.tool` decorators for function calling
- **PG Vector**: PostgreSQL extension for vector similarity search with `<=>` operator
- All examples are under 50 lines and show the core concept in action

## Scripts

### 01_query_expansion.py
**Strategy**: Generate multiple query variations to improve recall
- Shows: Expanding a single query into 3+ variations
- Tool: `expand_query()` and `search_knowledge_base()`
- Key: Searches with multiple perspectives and deduplicates results

### 02_reranking.py
**Strategy**: Two-stage retrieval with cross-encoder refinement
- Shows: Fast vector search (20 candidates) → accurate re-ranking (top 5)
- Tool: `search_with_reranking()`
- Key: Balance between retrieval speed and precision

### 03_agentic_rag.py
**Strategy**: Agent autonomously chooses tools (vector, SQL, web)
- Shows: Multiple tools for different data types
- Tools: `vector_search()`, `sql_query()`, `web_search()`
- Key: Agent decides which tool(s) to use based on query

### 04_multi_query_rag.py
**Strategy**: Parallel searches with reformulated queries
- Shows: Multiple query perspectives executed in parallel
- Tool: `multi_query_search()`
- Key: Unique union of all results from different query angles

### 05_context_aware_chunking.py
**Strategy**: Semantic chunking based on embedding similarity
- Shows: `semantic_chunk()` function that groups similar sentences
- Key: Chunk boundaries determined by semantic similarity, not fixed size
- Ingestion: Compares consecutive sentence embeddings

### 06_late_chunking.py
**Strategy**: Embed full document before chunking (Jina AI approach)
- Shows: `late_chunk()` processes entire document through transformer first
- Key: Token-level embeddings capture full context, then pooled per chunk
- Ingestion: `transformer_embed()` → chunk boundaries → mean pooling

### 07_hierarchical_rag.py
**Strategy**: Parent-child relationships with metadata
- Shows: Two tables (`parent_chunks`, `child_chunks`) with foreign keys
- Tool: `search_knowledge_base()` searches children, returns parents
- Key: Small chunks for matching, large parents for context

### 08_contextual_retrieval.py
**Strategy**: Add document context to chunks (Anthropic method)
- Shows: `add_context_to_chunk()` prepends LLM-generated context
- Key: Each chunk gets document-level context before embedding
- Ingestion: Original chunk → contextualized → embedded

### 09_self_reflective_rag.py
**Strategy**: Iterative refinement with self-assessment
- Shows: `search_and_grade()`, `refine_query()`, `answer_with_verification()`
- Tools: Grade relevance, refine queries, verify answers
- Key: Multiple LLM calls for reflection and improvement

### 10_knowledge_graphs.py
**Strategy**: Combine vector search with graph relationships
- Shows: Two tables (`entities`, `relationships`) forming a graph
- Tool: `search_knowledge_graph()` does hybrid vector + graph traversal
- Ingestion: Extract entities and relationships, store in graph structure

### 11_fine_tuned_embeddings.py
**Strategy**: Custom embedding model trained on domain data
- Shows: `fine_tune_model()` trains on query-document pairs
- Key: Domain-specific embeddings (medical, legal, financial)
- Ingestion: Uses fine-tuned model instead of generic embeddings

## Common Patterns

All scripts follow this structure:
```python
from pydantic_ai import Agent
import psycopg2
from pgvector.psycopg2 import register_vector

# Initialize agent
agent = Agent('openai:gpt-4o', system_prompt='...')

# Database connection
conn = psycopg2.connect("dbname=rag_db")
register_vector(conn)

# Ingestion function (strategy-specific)
def ingest_document(text: str):
    # ... chunking logic varies by strategy
    pass

# Agent tools (strategy-specific)
@agent.tool
def search_knowledge_base(query: str) -> str:
    # ... search logic varies by strategy
    pass

# Run agent
result = agent.run_sync("query")
print(result.data)
```

## Notes

- Functions like `get_embedding()`, `llm_generate()`, etc. are placeholders for clarity
- Database schemas are simplified; production would need proper table creation
- Each example focuses on demonstrating the core RAG strategy concept
- All scripts use pgvector's `<=>` operator for cosine distance similarity search

## Database Schema Examples

**Basic chunks table**:
```sql
CREATE TABLE chunks (
    id SERIAL PRIMARY KEY,
    content TEXT,
    embedding vector(768)
);
CREATE INDEX ON chunks USING ivfflat (embedding vector_cosine_ops);
```

**Hierarchical (parent-child)**:
```sql
CREATE TABLE parent_chunks (id INT PRIMARY KEY, content TEXT);
CREATE TABLE child_chunks (id SERIAL PRIMARY KEY, content TEXT, embedding vector(768), parent_id INT);
```

**Knowledge graph**:
```sql
CREATE TABLE entities (name TEXT PRIMARY KEY, embedding vector(768));
CREATE TABLE relationships (source TEXT, relation TEXT, target TEXT);
```
