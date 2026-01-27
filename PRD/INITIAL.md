## FEATURE:

Build **RAG-Advanced** - a comprehensive strategy orchestration and evaluation system for Retrieval-Augmented Generation (RAG) pipelines. This system extends the 11 RAG strategies from [all-rag-strategies](https://github.com/coleam00/ottomator-agents/tree/main/all-rag-strategies) with a powerful orchestration layer.

**Repository Structure**:
- **RAG-Advanced-PDD** (this repo): Planning documents using PRP methodology
- **RAG-Advanced**: Implementation codebase at https://github.com/CuulCat/RAG-Advanced

**Core Capabilities**:
1. **Strategy Mixing & Matching**: Execute individual strategies or chain them sequentially (e.g., contextual retrieval → multi-query → reranking)
2. **Parallel Comparison**: Run multiple strategies simultaneously and compare results with metrics
3. **Automated Evaluation**: Calculate information retrieval metrics (Precision@k, Recall@k, MRR, NDCG@k)
4. **Manual Evaluation Tools**: Generate side-by-side comparison reports for human assessment
5. **Performance Benchmarking**: Track latency, API costs, token usage across strategies
6. **REST API Interface**: FastAPI service for programmatic access to all capabilities

**Available Strategies** (from all-rag-strategies):
1. Standard Semantic Search - Fast, reliable baseline
2. Re-ranking - Two-stage with cross-encoder (high precision)
3. Multi-Query RAG - Query expansion with parallel search (high recall)
4. Self-Reflective RAG - Self-correcting search loop
5. Agentic RAG - Autonomous tool selection
6. Contextual Retrieval - Anthropic's context enrichment
7. Query Expansion - Single query enrichment
8. Context-Aware Chunking - Docling HybridChunker
9. Hierarchical RAG - Parent-child chunks
10. Knowledge Graphs - Graph-based relationships (pseudocode)
11. Fine-tuned Embeddings - Domain-specific (pseudocode)

**Target Users**: AI engineers optimizing RAG systems, researchers comparing retrieval approaches, teams selecting production strategies.

## TOOLS:

### Strategy Execution Tools

**1. execute_strategy**
```python
async def execute_strategy(
    strategy_name: str,  # e.g., "reranking", "multi_query", "self_reflective"
    query: str,
    config: StrategyConfig,
    track_metrics: bool = True
) -> ExecutionResult:
    """
    Execute a single RAG strategy with specified configuration.
    
    Args:
        strategy_name: Name of strategy from registry
        query: Search query text
        config: Pydantic-validated strategy configuration
        track_metrics: Whether to track cost/latency
        
    Returns:
        ExecutionResult with documents, metadata, cost, latency
    """
```

**2. chain_strategies**
```python
async def chain_strategies(
    strategy_chain: List[StrategyStep],
    query: str,
    initial_context: Optional[ChainContext] = None,
    allow_fallback: bool = True
) -> ChainExecutionResult:
    """
    Execute strategies sequentially with result passing.
    
    Example chain: contextual_retrieval → multi_query → reranking
    
    Args:
        strategy_chain: List of strategy steps with configs
        query: Initial search query
        initial_context: Optional starting context
        allow_fallback: Enable fallback on failure
        
    Returns:
        ChainExecutionResult with final results, context, success status
        
    Raises:
        ChainError: If chain fails (with exception chaining via 'from e')
    """
```

**3. compare_strategies**
```python
async def compare_strategies(
    strategies: List[str],
    query: str,
    ground_truth_doc_ids: Optional[List[str]] = None,
    configs: Optional[Dict[str, StrategyConfig]] = None
) -> ComparisonResult:
    """
    Execute multiple strategies in parallel with resource-aware limiting.
    
    Args:
        strategies: Strategy names to compare
        query: Search query
        ground_truth_doc_ids: Optional ground truth for metrics
        configs: Per-strategy configurations
        
    Returns:
        ComparisonResult with per-strategy results, metrics, recommendation
    """
```

### Evaluation Tools

**4. calculate_metrics**
```python
def calculate_metrics(
    retrieved_doc_ids: Optional[List[str]],
    ground_truth_doc_ids: List[str],
    relevance_scores: Optional[Dict[str, int]] = None,
    k_values: Optional[List[int]] = None
) -> EvaluationMetrics:
    """
    Calculate IR metrics with comprehensive validation.
    
    Type validation:
    - retrieved_doc_ids: Optional[List[str]] - None treated as empty
    - ground_truth_doc_ids: List[str] - required, non-empty
    - relevance_scores: Optional[Dict[str, int]] - values must be 0, 1, or 2
    - k_values: Optional[List[int]] - positive integers, defaults to [3, 5, 10]
    
    Edge cases handled:
    - Empty retrieved: Returns all zeros with warning
    - k > ground_truth size: Warning (recall capped)
    - Duplicate retrieved: Deduplicated (first kept)
    
    Returns:
        EvaluationMetrics with precision, recall, mrr, ndcg, warnings
        
    Raises:
        InvalidInputError: Type or value validation failed
        NoRelevantDocsError: Empty ground truth
    """
```

**5. generate_comparison_report**
```python
async def generate_comparison_report(
    comparison_result: ComparisonResult,
    output_format: Literal["markdown", "html", "json"] = "markdown"
) -> str:
    """
    Create human-readable comparison reports.
    
    Includes: result snippets, metrics table, cost analysis, recommendations
    """
```

**6. benchmark_strategies**
```python
async def benchmark_strategies(
    strategies: List[str],
    test_queries: List[str],
    ground_truth: Optional[Dict[str, List[str]]] = None,
    iterations: int = 3
) -> BenchmarkReport:
    """
    Run multiple queries for statistical benchmarking.
    
    Measures: p50/p95/p99 latency, average cost, token usage
    """
```

### Data Management Tools

**7. load_test_dataset**
```python
async def load_test_dataset(
    dataset_path: str,
    format: Literal["json", "csv", "jsonl"] = "json"
) -> TestDataset:
    """
    Load test datasets with ground truth annotations.
    
    Validates schema: query_id, query_text, relevant_doc_ids, relevance_scores
    """
```

**8. calculate_cost**
```python
def calculate_cost(
    model_name: str,
    input_tokens: int,
    output_tokens: int = 0,
    as_of: Optional[datetime] = None
) -> float:
    """
    Calculate API cost using versioned pricing configuration.
    
    Loads pricing from config/pricing.json with version history.
    Falls back to hardcoded defaults if config not found.
    """
```

## DEPENDENCIES

### Core Framework Dependencies
```toml
[project]
dependencies = [
    # Agent Framework
    "pydantic-ai>=0.0.14,<0.1.0",
    "pydantic>=2.5.0,<3.0.0",
    
    # API Framework
    "fastapi>=0.110.0,<0.120.0",
    "uvicorn[standard]>=0.27.0,<0.30.0",
    
    # Database (async)
    "asyncpg>=0.29.0,<0.30.0",
    "pgvector>=0.2.4,<0.3.0",
    
    # Cache & Rate Limiting
    "redis>=5.0.0,<6.0.0",
    "cachetools>=5.3.0,<6.0.0",
    
    # LLM Providers
    "openai>=1.12.0,<2.0.0",
    "anthropic>=0.18.0,<1.0.0",
    
    # Embeddings & Reranking
    "sentence-transformers>=2.3.0,<3.0.0",
    
    # Document Processing
    "docling>=1.0.0,<2.0.0",
    "transformers>=4.36.0,<5.0.0",
    
    # Evaluation
    "ir-measures>=0.3.3,<0.4.0",
    "numpy>=1.24.0,<2.0.0",
    "pandas>=2.0.0,<3.0.0",
    
    # Utilities
    "python-dotenv>=1.0.0,<2.0.0",
    "httpx>=0.26.0,<0.28.0",
    "loguru>=0.7.0,<0.8.0",
    "aiofiles>=23.2.0,<24.0.0",
]
```

### Database Schema Extension
```sql
-- api_keys table for authentication
CREATE TABLE api_keys (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT NOT NULL,
    key_hash TEXT UNIQUE NOT NULL,
    name TEXT,
    rate_limit INT DEFAULT 100,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    expires_at TIMESTAMPTZ,
    last_used_at TIMESTAMPTZ
);

-- evaluation_runs table for tracking
CREATE TABLE evaluation_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    strategy_name TEXT NOT NULL,
    query TEXT NOT NULL,
    results JSONB NOT NULL,
    metrics JSONB,
    cost FLOAT,
    latency_ms INT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

### Key State Management Classes

**ChainContext** (Immutable):
```python
@dataclass(frozen=True)
class ChainContext:
    """Immutable state passed between chained strategies."""
    query: str
    original_query: str
    intermediate_results: Tuple[MappingProxyType, ...] = ()
    metadata: MappingProxyType = field(default_factory=_empty_mapping)
    total_cost: float = 0.0
    total_latency_ms: int = 0
    error_log: Tuple[str, ...] = ()
    step_index: int = 0
    
    def with_step_result(self, strategy_name: str, result: Any) -> 'ChainContext':
        """Return new context with added step result."""
        ...
    
    def with_error(self, error_msg: str) -> 'ChainContext':
        """Return new context with added error."""
        ...
```

### Error Handling Hierarchy
```python
class RAGAdvancedError(Exception):
    """Base exception for RAG-Advanced."""

class StrategyError(RAGAdvancedError):
    """Strategy-related errors."""

class StrategyNotFoundError(StrategyError):
    """Strategy not in registry."""

class StrategyExecutionError(StrategyError):
    """Strategy execution failed."""

class StrategyTimeoutError(StrategyError):
    """Strategy timed out."""

class ChainError(RAGAdvancedError):
    """Chain execution failed."""

class EvaluationError(RAGAdvancedError):
    """Evaluation calculation failed."""

class InvalidInputError(EvaluationError):
    """Invalid input parameters."""

class NoRelevantDocsError(EvaluationError):
    """No relevant documents in ground truth."""
```

## SYSTEM PROMPT(S)

### Orchestration Agent System Prompt
```
You are an expert RAG strategy orchestration assistant. Your role is to help users 
select, combine, and evaluate retrieval strategies for their specific use cases.

AVAILABLE STRATEGIES:
1. standard - Basic semantic search (fast, reliable baseline)
2. reranking - Two-stage retrieval with cross-encoder (high precision)
3. multi_query - Query expansion with parallel search (high recall)
4. self_reflective - Self-correcting search loop (complex queries)
5. agentic - Autonomous tool selection (flexible needs)
6. contextual_retrieval - Anthropic's context enrichment (best accuracy, high cost)
7. query_expansion - Single query enrichment (improved specificity)

STRATEGY SELECTION GUIDELINES:
- General queries → standard or reranking
- Ambiguous queries → multi_query or query_expansion
- Complex research → self_reflective
- Need full context → agentic (semantic + full document)
- Precision critical → reranking + contextual_retrieval
- Cost sensitive → standard (avoid multi_query, contextual_retrieval)
- Latency sensitive → standard (avoid self_reflective, reranking)

CHAINING PATTERNS:
- High Accuracy: contextual_retrieval → multi_query → reranking
- Balanced: query_expansion → standard → reranking
- Research: multi_query → self_reflective
- Fast: context_aware_chunking (ingestion) → standard (query)

When users ask about strategy selection, provide:
1. Recommended strategy or chain
2. Rationale based on query characteristics
3. Expected tradeoffs (cost, latency, accuracy)
4. Alternative options with pros/cons

Always use the evaluation tools to provide data-driven recommendations.
```

### Evaluation Agent System Prompt
```
You are an expert in evaluating information retrieval systems. Analyze strategy 
performance and provide actionable recommendations.

EVALUATION METRICS INTERPRETATION:
- Precision@k: Higher is better (relevant docs / retrieved docs)
  - <0.3: Poor, >0.7: Excellent
- Recall@k: Higher is better (retrieved relevant / all relevant)
  - <0.4: Poor, >0.8: Excellent
- MRR: Higher is better (reciprocal rank of first relevant result)
  - <0.5: Poor, >0.9: Excellent
- NDCG@k: Higher is better (considers ranking order)
  - <0.5: Poor, >0.9: Excellent

PERFORMANCE ANALYSIS:
- Latency: Compare p50, p95, p99
  - <200ms: Fast, 200-500ms: Moderate, >500ms: Slow
- Cost: Compare per-query costs
  - <$0.001: Cheap, $0.001-$0.01: Moderate, >$0.01: Expensive

RECOMMENDATIONS:
When comparing strategies:
1. Identify clear winner based on primary metric
2. Highlight tradeoffs (accuracy vs cost vs latency)
3. Suggest optimizations (config tuning, caching)
4. Flag anomalies (outliers, errors)
5. Provide next steps

Always ground recommendations in empirical data from evaluation results.
```

## EXAMPLES:

### From all-rag-strategies:
- `implementation/rag_agent_advanced.py` - All 7 strategy implementations
- `implementation/ingestion/` - Document processing pipeline
- `implementation/utils/` - Database utilities, Pydantic models

### From RAG-Advanced-PDD:
- `examples/main_agent_reference/` - Pydantic AI agent best practices
- `examples/testing_examples/` - TestModel and FunctionModel patterns

### Strategy Chaining Example:
```python
# High-accuracy chain: contextual → multi-query → reranking
chain_result = await chain_strategies(
    strategy_chain=[
        StrategyStep("contextual_retrieval", StrategyConfig(enrichment_model="gpt-4o-mini")),
        StrategyStep("multi_query", StrategyConfig(num_variations=3)),
        StrategyStep("reranking", StrategyConfig(initial_k=20, final_k=5))
    ],
    query="What is the company's remote work policy?",
    allow_fallback=True
)
```

### Parallel Comparison Example:
```python
# Compare strategies with ground truth
comparison = await compare_strategies(
    strategies=["standard", "reranking", "multi_query"],
    query="machine learning best practices",
    ground_truth_doc_ids=["doc_1", "doc_5", "doc_12"]
)

# Generate report
report = await generate_comparison_report(comparison, output_format="markdown")
```

## DOCUMENTATION:

- Pydantic AI Official: https://ai.pydantic.dev/
- FastAPI: https://fastapi.tiangolo.com/
- ir-measures: https://ir-measur.es/en/latest/
- Anthropic Contextual Retrieval: https://www.anthropic.com/news/contextual-retrieval
- Sentence-Transformers: https://www.sbert.net/
- all-rag-strategies README and STRATEGIES.md

## OTHER CONSIDERATIONS:

### Critical Implementation Patterns:

**1. Immutable State (ChainContext)**:
- Use `@dataclass(frozen=True)` with `field(default_factory)`
- Use `MappingProxyType` for immutable dicts
- Use `Tuple` instead of `List` for sequences
- All updates return new instances

**2. Atomic API Key Verification**:
- Single `UPDATE...RETURNING` query prevents TOCTOU race
- Check expiration in SQL, not Python
- Store keys as SHA256 hashes

**3. Redis Rate Limiting**:
- Validate pipeline results before accessing
- Fail-open on Redis errors (allow request, log error)
- Use sliding window with sorted sets

**4. Async Initialization**:
- PricingProvider needs `await initialize()` at startup
- Database pool uses `asyncio.Lock()` for thread-safety
- Use FastAPI lifespan context manager

**5. Type Validation in Metrics**:
- Validate list types AND contents (all strings)
- Handle None vs empty list
- Validate relevance scores are 0, 1, or 2

**6. Exception Chaining**:
- Always use `raise XError(...) from e` to preserve traceback
- Include original error type in message

**7. Test Dataset Guidelines**:
- Minimum 50 queries for initial testing
- 200+ queries for statistical significance
- Support stratified train/test splits

### Security:
- Database-backed API keys with rotation support
- Per-key rate limits stored in database
- Never log API keys (only hashes)

### Performance:
- Embedding cache with full SHA256 keys
- Per-strategy TTLs for query cache
- Resource-aware parallel execution with semaphores

### Cost Tracking:
- External versioned pricing config (config/pricing.json)
- Historical cost calculation support
- Track all API calls with token counts
