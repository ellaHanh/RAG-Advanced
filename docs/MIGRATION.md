# Migration Guide

> **Note**: This documentation will be fully populated when all modules are implemented.

## Migrating from all-rag-strategies

This guide helps you migrate from the original [all-rag-strategies](https://github.com/coleam00/ottomator-agents/tree/main/all-rag-strategies) repository to RAG-Advanced.

## Key Differences

| Feature | all-rag-strategies | RAG-Advanced |
|---------|-------------------|--------------|
| Strategy Selection | Manual | Registry-based |
| Chaining | Not supported | Built-in |
| Comparison | Manual | Parallel execution |
| Cost Tracking | Not supported | Per-request tracking |
| Evaluation | Not supported | IR metrics built-in |
| API | CLI only | REST API |
| Authentication | None | API key required |
| Rate Limiting | None | Redis-based |

## Breaking Changes

### 1. Import Paths

```python
# Before (all-rag-strategies)
from implementation.agents.rag_agent_advanced import search_with_reranking
from implementation.ingestion.embedder import create_embedder

# After (RAG-Advanced)
from orchestration.executor import execute_strategy
from strategies.ingestion.embedder import create_embedder
```

### 2. Strategy Execution

```python
# Before
result = await search_with_reranking(ctx, query, limit=5)

# After
from orchestration.executor import execute_strategy
from orchestration.models import StrategyConfig

result = await execute_strategy(
    strategy_name="reranking",
    query=query,
    config=StrategyConfig(initial_k=20, final_k=5)
)
```

### 3. Database Schema

New tables are added for API keys and evaluation:

```sql
-- Run this migration
psql $DATABASE_URL < evaluation/schema_extension.sql
```

## Migration Steps

### Step 1: Update Dependencies

```bash
# Remove old package
pip uninstall docling-rag-agent

# Install new package
pip install -e path/to/RAG-Advanced
```

### Step 2: Update Environment

```bash
# Add new environment variables
echo "REDIS_URL=redis://localhost:6379" >> .env
```

### Step 3: Run Database Migration

```bash
psql $DATABASE_URL < evaluation/schema_extension.sql
```

### Step 4: Update Code

See "Breaking Changes" above for code updates.

### Step 5: Generate API Key

```bash
# Generate a new API key
python -m api.auth generate_key --user-id=your-user-id
```

## Rollback Plan

If you need to rollback:

1. Keep your original `.env` file
2. Database changes are additive (old schema still works)
3. Re-install original package

```bash
pip uninstall rag-advanced
pip install -e path/to/all-rag-strategies/implementation
```

## Support

For migration issues, open an issue on GitHub.
