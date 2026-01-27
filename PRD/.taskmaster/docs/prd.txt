# RAG-Advanced: Strategy Orchestration and Evaluation System

## Problem Statement

AI engineers building RAG systems face a common challenge: **choosing the right retrieval strategy**. With 11+ different strategies available (standard search, reranking, multi-query, self-reflective, etc.), teams struggle to:

1. **Compare strategies objectively** - No standardized way to measure which strategy works best for their data
2. **Combine strategies effectively** - Want to chain strategies (e.g., multi-query → reranking) but no orchestration layer exists
3. **Track costs and performance** - Need to balance accuracy vs latency vs cost without proper instrumentation
4. **Evaluate with ground truth** - Lack tools to calculate IR metrics (Precision, Recall, MRR, NDCG) systematically

Existing solutions like `all-rag-strategies` provide individual strategy implementations but lack the orchestration, evaluation, and API layers needed for production use.

## Target Users

**Primary**: AI/ML Engineers building production RAG systems
- Need to select optimal strategy for their use case
- Want data-driven decisions about cost vs accuracy tradeoffs
- Require REST API for integration with existing systems

**Secondary**: Researchers comparing retrieval approaches
- Need reproducible benchmarks across strategies
- Want to test new strategies against baselines
- Require standard IR metrics

## Success Metrics

- **Strategy selection time**: Reduce from days to hours (run benchmark, get recommendation)
- **Evaluation coverage**: Support Precision@k, Recall@k, MRR, NDCG@k at k=[3,5,10]
- **API response time**: < 500ms for single strategy execution
- **Test coverage**: 85% for orchestration, 90% for evaluation modules

---

## Capability Tree

### Capability: Strategy Execution
Execute individual RAG strategies with configuration and metrics tracking.

#### Feature: Single Strategy Execution
- **Description**: Execute one RAG strategy with specified parameters and return results
- **Inputs**: Strategy name, query text, StrategyConfig (Pydantic model)
- **Outputs**: ExecutionResult with documents, cost, latency, token counts
- **Behavior**: Load strategy from registry, execute with config, track all API calls for cost

#### Feature: Strategy Registry
- **Description**: Central registry for type-safe strategy lookup and metadata
- **Inputs**: Strategy name string
- **Outputs**: Strategy class with metadata (description, resource requirements)
- **Behavior**: Decorator-based registration, validation on register, list all available strategies

#### Feature: Cost Tracking
- **Description**: Calculate and aggregate API costs using versioned pricing
- **Inputs**: Model name, input tokens, output tokens, optional datetime
- **Outputs**: Cost in USD (float)
- **Behavior**: Load pricing from JSON config, support historical pricing, fallback to defaults

### Capability: Strategy Orchestration
Chain and compare multiple strategies with proper state management.

#### Feature: Chain State Management
- **Description**: Manage immutable ChainContext that passes between strategy steps
- **Inputs**: Initial query, optional metadata
- **Outputs**: ChainContext with query, intermediate_results, cost, latency
- **Behavior**: Frozen dataclass with functional updates via with_* methods

#### Feature: Sequential Strategy Executor
- **Description**: Execute strategies one after another in sequence
- **Inputs**: List of StrategyStep, ChainContext
- **Outputs**: Updated ChainContext with step results appended
- **Behavior**: Iterate through steps, execute each, update context immutably

#### Feature: Chain Fallback Handler
- **Description**: Handle failures during chain execution with graceful fallback
- **Inputs**: Failed step info, ChainContext, fallback strategy (optional)
- **Outputs**: Recovery result or re-raised exception with chain
- **Behavior**: Log error, try fallback if configured, use exception chaining (raise from)

#### Feature: Parallel Execution Engine
- **Description**: Run multiple strategies concurrently using asyncio.gather
- **Inputs**: List of strategy names, query, config per strategy
- **Outputs**: List of ExecutionResult (or exceptions)
- **Behavior**: Launch all strategies concurrently, collect results/errors

#### Feature: Resource Semaphore Manager
- **Description**: Limit concurrent access to shared resources (DB, API)
- **Inputs**: ResourceLimits config (max DB, embedding, LLM connections)
- **Outputs**: Acquired semaphore context manager
- **Behavior**: Per-resource-type asyncio.Semaphore, AsyncExitStack for cleanup

#### Feature: Comparison Result Aggregator
- **Description**: Aggregate parallel execution results for comparison
- **Inputs**: List of ExecutionResult, optional ground truth
- **Outputs**: ComparisonResult with rankings, metrics per strategy
- **Behavior**: Calculate relative metrics, rank by latency/cost/accuracy

#### Feature: Timeout and Partial Failure Handler
- **Description**: Handle timeouts and partial failures in parallel execution
- **Inputs**: asyncio.gather results with return_exceptions=True
- **Outputs**: Filtered results with error summaries
- **Behavior**: Separate successes from failures, log timeouts, continue with partial results

### Capability: Evaluation Framework
Calculate IR metrics and generate benchmarks.

#### Feature: IR Metrics Calculation
- **Description**: Calculate Precision@k, Recall@k, MRR, NDCG@k with comprehensive validation
- **Inputs**: Retrieved doc IDs, ground truth doc IDs, optional relevance scores, k values
- **Outputs**: EvaluationMetrics with all metrics and warnings
- **Behavior**: Type validation, edge case handling (empty results, k > ground truth), uses ir-measures library

#### Feature: Benchmark Runner
- **Description**: Run multiple queries across strategies for statistical analysis
- **Inputs**: Strategy list, test queries, optional ground truth, iterations count
- **Outputs**: BenchmarkReport with p50/p95/p99 latency, aggregate metrics
- **Behavior**: Multiple iterations for stability, statistical summary, strategy ranking

#### Feature: Ground Truth Management
- **Description**: Load, validate, and manage test datasets with ground truth
- **Inputs**: Dataset path, format (JSON/CSV/JSONL)
- **Outputs**: TestDataset with validated queries and relevance scores
- **Behavior**: Schema validation, train/test split support, LLM-assisted generation with Cohen's Kappa

#### Feature: Markdown Report Generator
- **Description**: Generate markdown-formatted evaluation reports
- **Inputs**: ComparisonResult or BenchmarkReport
- **Outputs**: Markdown string with tables and sections
- **Behavior**: Metrics tables, strategy rankings, cost breakdown

#### Feature: HTML Report Generator  
- **Description**: Generate styled HTML evaluation reports
- **Inputs**: ComparisonResult or BenchmarkReport
- **Outputs**: HTML string with CSS styling
- **Behavior**: Interactive tables, visual charts placeholders, color-coded metrics

#### Feature: JSON Report Exporter
- **Description**: Export evaluation data as structured JSON
- **Inputs**: ComparisonResult or BenchmarkReport
- **Outputs**: JSON string (serializable dict)
- **Behavior**: Include all metrics, timestamps, config used

### Capability: REST API
Expose all functionality via FastAPI endpoints.

#### Feature: API Key Hash Verification
- **Description**: Verify API key against database using SHA256 hash
- **Inputs**: Raw API key from X-API-Key header
- **Outputs**: Boolean validity + APIKey record if valid
- **Behavior**: Hash key, query database, check is_active and expires_at

#### Feature: Atomic Key Usage Update
- **Description**: Update last_used_at atomically during verification
- **Inputs**: API key hash
- **Outputs**: Updated APIKey record
- **Behavior**: Single UPDATE...RETURNING query to prevent TOCTOU race

#### Feature: Redis Sliding Window Counter
- **Description**: Implement sliding window rate limiting with Redis sorted sets
- **Inputs**: Key identifier, window size, max requests
- **Outputs**: Current request count in window
- **Behavior**: ZADD with timestamp, ZREMRANGEBYSCORE for cleanup, ZCARD for count

#### Feature: Rate Limit Result Calculator
- **Description**: Calculate rate limit status from Redis counter
- **Inputs**: Current count, max limit, window reset time
- **Outputs**: RateLimitResult (allowed, remaining, reset_at)
- **Behavior**: Compare count vs limit, calculate remaining, return reset timestamp

#### Feature: Rate Limiter Fail-Open Handler
- **Description**: Handle Redis failures gracefully
- **Inputs**: Redis exception
- **Outputs**: Default allow result with warning
- **Behavior**: Log error, return allowed=True to prevent blocking on Redis outage

#### Feature: Strategy Endpoints
- **Description**: REST endpoints for execute, chain, compare operations
- **Inputs**: HTTP POST with JSON body
- **Outputs**: JSON response with results and metadata
- **Behavior**: Request validation with Pydantic, OpenAPI documentation

#### Feature: Metrics Calculation Endpoint
- **Description**: POST endpoint for calculating IR metrics
- **Inputs**: JSON body with retrieved_ids, ground_truth_ids, k_values
- **Outputs**: JSON response with precision, recall, mrr, ndcg
- **Behavior**: Validate inputs, call calculate_metrics(), return results

#### Feature: Batch Metrics Endpoint
- **Description**: POST endpoint for evaluating multiple queries at once
- **Inputs**: JSON array of {query_id, retrieved_ids, ground_truth_ids}
- **Outputs**: JSON response with per-query metrics and aggregates
- **Behavior**: Process each query, aggregate results, return summary stats

#### Feature: Async Benchmark Trigger Endpoint
- **Description**: POST endpoint to start long-running benchmark
- **Inputs**: JSON body with strategies, test_dataset_id, iterations
- **Outputs**: JSON response with benchmark_id for polling
- **Behavior**: Validate inputs, queue benchmark task, return immediately

#### Feature: Benchmark Status Endpoint
- **Description**: GET endpoint to check benchmark progress
- **Inputs**: benchmark_id path parameter
- **Outputs**: JSON with status (pending/running/complete), progress, results if complete
- **Behavior**: Query benchmark status from database/cache

### Capability: Caching Layer
Cache embeddings and query results for performance.

#### Feature: Embedding Cache Core
- **Description**: LRU cache storage for embedding vectors
- **Inputs**: Cache key (SHA256 hash of text + model)
- **Outputs**: Embedding vector or cache miss
- **Behavior**: Fixed-size LRU eviction, thread-safe access

#### Feature: Embedding Cache Key Generator
- **Description**: Generate deterministic cache keys for embeddings
- **Inputs**: Text string, model name
- **Outputs**: SHA256 hash string
- **Behavior**: Normalize text, combine with model, full hash (no truncation)

#### Feature: Query Result TTL Cache
- **Description**: Time-based cache for strategy execution results
- **Inputs**: Strategy name, query, config hash
- **Outputs**: Cached ExecutionResult or cache miss
- **Behavior**: Per-strategy TTL configuration, automatic expiration

#### Feature: Cache Statistics Tracker
- **Description**: Track cache performance metrics
- **Inputs**: Cache hit/miss events
- **Outputs**: CacheStats (hits, misses, hit_rate, size)
- **Behavior**: Thread-safe counters, periodic logging

### Capability: Database Layer
Manage PostgreSQL connections and schema.

#### Feature: Connection Pool
- **Description**: Thread-safe asyncpg connection pool
- **Inputs**: DatabaseConfig (min/max size, timeouts)
- **Outputs**: Connection pool instance
- **Behavior**: asyncio.Lock for initialization, double-check pattern, graceful shutdown

#### Feature: Schema Management
- **Description**: Database schema for API keys, evaluation runs, metrics
- **Inputs**: SQL migration files
- **Outputs**: Database tables and indexes
- **Behavior**: Base schema from all-rag-strategies + evaluation extensions

---

## Repository Structure

```
RAG-Advanced/
├── api/                        # Maps to: REST API capability
│   ├── main.py                 # FastAPI application, lifespan
│   ├── auth.py                 # API key verification
│   ├── rate_limiter.py         # Redis rate limiting
│   ├── database.py             # Connection pool management
│   ├── monitoring.py           # Prometheus metrics
│   └── routes/
│       ├── strategies.py       # Strategy endpoints
│       ├── evaluation.py       # Evaluation endpoints
│       └── benchmarks.py       # Benchmark endpoints
│
├── orchestration/              # Maps to: Strategy Execution + Orchestration
│   ├── executor.py             # Single strategy execution
│   ├── chainer.py              # Sequential chaining
│   ├── comparator.py           # Parallel comparison
│   ├── cost_tracker.py         # Cost calculation
│   ├── pricing.py              # Versioned pricing provider
│   ├── registry.py             # Strategy registry
│   ├── models.py               # ChainContext, StrategyConfig
│   └── errors.py               # Exception hierarchy
│
├── evaluation/                 # Maps to: Evaluation Framework
│   ├── metrics.py              # IR metrics calculation
│   ├── benchmarks.py           # Benchmark runner
│   ├── dataset_manager.py      # Test dataset handling
│   ├── ground_truth.py         # Ground truth generation
│   ├── report_generator.py     # Report generation
│   └── schema_extension.sql    # Database schema
│
├── strategies/                 # Maps to: RAG Strategies (from all-rag-strategies)
│   ├── ingestion/              # Document processing
│   │   ├── ingest.py
│   │   ├── chunker.py
│   │   ├── embedder.py
│   │   └── contextual_enrichment.py
│   ├── agents/
│   │   └── rag_agent_advanced.py
│   ├── utils/
│   │   ├── db_utils.py
│   │   ├── models.py
│   │   └── schema.sql
│   └── cache.py                # Embedding/query cache
│
├── config/
│   └── pricing.json            # Versioned pricing configuration
│
├── datasets/
│   ├── sample/                 # Example test datasets
│   └── README.md
│
├── tests/
│   ├── conftest.py
│   ├── test_orchestration/
│   ├── test_evaluation/
│   └── test_api/
│
├── docs/
│   ├── API.md
│   ├── EVALUATION.md
│   └── MIGRATION.md
│
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```

## Module Definitions

### Module: orchestration
- **Maps to capability**: Strategy Execution, Strategy Orchestration
- **Responsibility**: Execute, chain, and compare RAG strategies
- **Exports**:
  - `execute_strategy()` - Execute single strategy
  - `chain_strategies()` - Sequential chaining
  - `compare_strategies()` - Parallel comparison
  - `ChainContext` - Immutable state class
  - `StrategyRegistry` - Type-safe registry

### Module: evaluation
- **Maps to capability**: Evaluation Framework
- **Responsibility**: Calculate metrics, run benchmarks, manage datasets
- **Exports**:
  - `calculate_metrics()` - IR metrics calculation
  - `BenchmarkRunner` - Statistical benchmarking
  - `DatasetManager` - Test dataset handling
  - `generate_report()` - Report generation

### Module: api
- **Maps to capability**: REST API
- **Responsibility**: HTTP interface, authentication, rate limiting
- **Exports**:
  - `app` - FastAPI application instance
  - `verify_api_key()` - Authentication dependency
  - `RedisRateLimiter` - Rate limiting

### Module: strategies
- **Maps to capability**: RAG Strategies (inherited from all-rag-strategies)
- **Responsibility**: Individual strategy implementations
- **Exports**:
  - Strategy implementations (standard, reranking, multi_query, etc.)
  - Document ingestion pipeline
  - Embedding and caching utilities

---

## Dependency Chain

### Foundation Layer (Phase 0)
No dependencies - these are built first.

- **orchestration/errors.py**: Exception hierarchy for all modules
- **orchestration/models.py**: ChainContext, StrategyConfig, ExecutionResult (Pydantic models)
- **api/database.py**: Connection pool with asyncio.Lock protection

### Configuration Layer (Phase 1)
- **orchestration/pricing.py**: Depends on [aiofiles for async I/O]
- **config/pricing.json**: Static configuration file
- **strategies/cache.py**: Depends on [orchestration/models.py for types]

### Strategy Layer (Phase 2)
- **orchestration/registry.py**: Depends on [orchestration/models.py, orchestration/errors.py]
- **orchestration/executor.py**: Depends on [orchestration/registry.py, orchestration/pricing.py, strategies/cache.py]
- **orchestration/cost_tracker.py**: Depends on [orchestration/pricing.py]

### Orchestration Layer (Phase 3)
- **orchestration/chainer.py**: Depends on [orchestration/executor.py, orchestration/models.py, orchestration/errors.py]
- **orchestration/comparator.py**: Depends on [orchestration/executor.py, orchestration/models.py]

### Evaluation Layer (Phase 4)
- **evaluation/metrics.py**: Depends on [orchestration/errors.py] (uses ir-measures library)
- **evaluation/ground_truth.py**: Depends on [evaluation/metrics.py]
- **evaluation/dataset_manager.py**: Depends on [evaluation/ground_truth.py]
- **evaluation/benchmarks.py**: Depends on [orchestration/executor.py, evaluation/metrics.py]
- **evaluation/report_generator.py**: Depends on [evaluation/metrics.py, evaluation/benchmarks.py]

### API Layer (Phase 5)
- **api/auth.py**: Depends on [api/database.py]
- **api/rate_limiter.py**: Depends on [redis client]
- **api/routes/strategies.py**: Depends on [orchestration/executor.py, orchestration/chainer.py, orchestration/comparator.py, api/auth.py]
- **api/routes/evaluation.py**: Depends on [evaluation/metrics.py, evaluation/benchmarks.py, api/auth.py]
- **api/main.py**: Depends on [api/routes/*, api/database.py, orchestration/pricing.py]

### Infrastructure Layer (Phase 6)
- **Dockerfile**: Depends on [pyproject.toml]
- **docker-compose.yml**: Depends on [Dockerfile, strategies/utils/schema.sql, evaluation/schema_extension.sql]
- **docs/MIGRATION.md**: Depends on [all modules complete]

---

## Development Phases

### Phase 0: Foundation
**Goal**: Establish core types, error handling, and database connectivity

**Entry Criteria**: Clean repository with pyproject.toml

**Tasks**:
- [ ] Create orchestration/errors.py with exception hierarchy (depends on: none)
  - Acceptance criteria: All error classes defined with proper inheritance
  - Test strategy: Unit test exception creation and inheritance

- [ ] Create orchestration/models.py with ChainContext and StrategyConfig (depends on: none)
  - Acceptance criteria: Frozen ChainContext with MappingProxyType, all with_* methods
  - Test strategy: Test immutability, functional updates, serialization

- [ ] Create api/database.py with connection pool (depends on: none)
  - Acceptance criteria: Thread-safe pool creation with asyncio.Lock
  - Test strategy: Test concurrent initialization, pool lifecycle

**Exit Criteria**: Foundation modules importable without errors, tests pass

**Delivers**: Base types and infrastructure for all other modules

---

### Phase 1: Configuration
**Goal**: Setup pricing configuration and caching infrastructure

**Entry Criteria**: Phase 0 complete

**Tasks**:
- [ ] Create config/pricing.json with versioned pricing (depends on: none)
  - Acceptance criteria: Valid JSON with pricing history, multiple models
  - Test strategy: JSON schema validation

- [ ] Create orchestration/pricing.py with async PricingProvider (depends on: [orchestration/models.py])
  - Acceptance criteria: Async initialize(), historical pricing, fallback defaults
  - Test strategy: Test async loading, missing config, historical lookup

- [ ] Create strategies/cache.py with cache key generator (depends on: none)
  - Acceptance criteria: SHA256 hash of text + model, no truncation
  - Test strategy: Test deterministic key generation, collision resistance

- [ ] Add EmbeddingCache LRU core to strategies/cache.py (depends on: [strategies/cache.py key generator])
  - Acceptance criteria: Fixed-size LRU, thread-safe get/set
  - Test strategy: Test eviction, concurrent access

- [ ] Add QueryResultCache TTL to strategies/cache.py (depends on: [orchestration/models.py])
  - Acceptance criteria: Per-strategy TTL config, automatic expiration
  - Test strategy: Test TTL expiration, different strategy TTLs

- [ ] Add CacheStats tracker to strategies/cache.py (depends on: [strategies/cache.py])
  - Acceptance criteria: Thread-safe hit/miss counters, hit_rate calculation
  - Test strategy: Test counter accuracy, concurrent updates

**Exit Criteria**: Pricing and caching utilities working

**Delivers**: Cost calculation and caching infrastructure

---

### Phase 2: Strategy Infrastructure
**Goal**: Build strategy registry and single execution

**Entry Criteria**: Phase 1 complete

**Tasks**:
- [ ] Create orchestration/registry.py with StrategyRegistry (depends on: [orchestration/models.py, orchestration/errors.py])
  - Acceptance criteria: Decorator registration, validation, list_strategies()
  - Test strategy: Test registration, duplicate detection, protocol compliance

- [ ] Create orchestration/cost_tracker.py with CostTracker (depends on: [orchestration/pricing.py])
  - Acceptance criteria: Thread-safe aggregation, per-model tracking
  - Test strategy: Test concurrent tracking, summary generation

- [ ] Create orchestration/executor.py with execute_strategy() (depends on: [orchestration/registry.py, orchestration/pricing.py, strategies/cache.py])
  - Acceptance criteria: Strategy execution, cost tracking, cache integration
  - Test strategy: Mock strategy execution, verify cost/latency tracking

**Exit Criteria**: Single strategy execution working with cost tracking

**Delivers**: Ability to execute individual strategies

---

### Phase 3: Orchestration
**Goal**: Implement strategy chaining and parallel comparison

**Entry Criteria**: Phase 2 complete

**Tasks**:
- [ ] Implement Chain State Management in orchestration/models.py (depends on: [orchestration/models.py])
  - Acceptance criteria: ChainContext with with_step_result(), with_error() methods
  - Test strategy: Test immutability, functional updates

- [ ] Create orchestration/chainer.py with sequential executor (depends on: [orchestration/executor.py, orchestration/models.py])
  - Acceptance criteria: Execute strategies in order, pass context between steps
  - Test strategy: Test 2-3 step chains with mock strategies

- [ ] Add fallback handler to orchestration/chainer.py (depends on: [orchestration/chainer.py, orchestration/errors.py])
  - Acceptance criteria: Catch failures, try fallback strategy, exception chaining
  - Test strategy: Test fallback execution, error propagation with `raise from`

- [ ] Create orchestration/comparator.py parallel engine (depends on: [orchestration/executor.py])
  - Acceptance criteria: Run strategies with asyncio.gather, return_exceptions=True
  - Test strategy: Test concurrent execution timing

- [ ] Add resource semaphore manager to orchestration/comparator.py (depends on: [orchestration/comparator.py])
  - Acceptance criteria: Per-resource semaphores, AsyncExitStack cleanup
  - Test strategy: Test resource limiting with mock slow strategies

- [ ] Add result aggregator to orchestration/comparator.py (depends on: [orchestration/comparator.py])
  - Acceptance criteria: Combine results, rank strategies, calculate relative metrics
  - Test strategy: Test ranking logic with varied results

- [ ] Add timeout/failure handler to orchestration/comparator.py (depends on: [orchestration/comparator.py])
  - Acceptance criteria: Filter timeouts, continue with partial results
  - Test strategy: Test with intentionally failing/slow strategies

**Exit Criteria**: Chain and compare operations working with proper error handling

**Delivers**: Full orchestration capabilities

---

### Phase 4: Evaluation
**Goal**: Implement IR metrics and benchmarking

**Entry Criteria**: Phase 3 complete

**Tasks**:
- [ ] Create evaluation/metrics.py with calculate_metrics() (depends on: [orchestration/errors.py])
  - Acceptance criteria: Type validation, all metrics via ir-measures, edge case handling
  - Test strategy: Test all edge cases, validate against known results

- [ ] Create evaluation/ground_truth.py with agreement calculation (depends on: [evaluation/metrics.py])
  - Acceptance criteria: Cohen's Kappa, LLM-assisted generation, quality metrics
  - Test strategy: Test agreement calculation, conservative merging

- [ ] Create evaluation/dataset_manager.py with DatasetSplit (depends on: [evaluation/ground_truth.py])
  - Acceptance criteria: Stratified splits, validation, schema checking
  - Test strategy: Test split ratios, stratification by field

- [ ] Create evaluation/benchmarks.py with BenchmarkRunner (depends on: [orchestration/executor.py, evaluation/metrics.py])
  - Acceptance criteria: Multiple iterations, statistical summary, ranking
  - Test strategy: Test benchmark execution, statistics accuracy

- [ ] Create evaluation/report_generator.py with Markdown generator (depends on: [evaluation/metrics.py])
  - Acceptance criteria: Markdown tables, strategy rankings section
  - Test strategy: Test table formatting, section headers

- [ ] Add HTML report generator to evaluation/report_generator.py (depends on: [evaluation/report_generator.py markdown])
  - Acceptance criteria: Styled HTML with CSS, readable tables
  - Test strategy: Test HTML validity, styling applied

- [ ] Add JSON exporter to evaluation/report_generator.py (depends on: [evaluation/metrics.py])
  - Acceptance criteria: Serializable dict output, all metrics included
  - Test strategy: Test JSON schema, round-trip serialization

- [ ] Create evaluation/schema_extension.sql (depends on: none)
  - Acceptance criteria: api_keys, evaluation_runs, strategy_metrics tables
  - Test strategy: SQL syntax validation, constraint testing

**Exit Criteria**: Full evaluation framework operational

**Delivers**: Metrics calculation, benchmarking, report generation

---

### Phase 5: API Layer
**Goal**: Expose all functionality via REST API

**Entry Criteria**: Phase 4 complete

**Tasks**:
- [ ] Create api/auth.py with SHA256 key hash verification (depends on: [api/database.py])
  - Acceptance criteria: Hash comparison, check is_active and expires_at
  - Test strategy: Test valid/invalid/expired keys

- [ ] Add atomic usage update to api/auth.py (depends on: [api/auth.py])
  - Acceptance criteria: UPDATE...RETURNING query, no race condition
  - Test strategy: Test concurrent verification calls

- [ ] Create api/rate_limiter.py with Redis sliding window (depends on: none)
  - Acceptance criteria: ZADD/ZREMRANGEBYSCORE/ZCARD operations
  - Test strategy: Test window behavior with mock Redis

- [ ] Add rate limit result calculator to api/rate_limiter.py (depends on: [api/rate_limiter.py])
  - Acceptance criteria: Calculate allowed/remaining/reset_at
  - Test strategy: Test boundary conditions

- [ ] Add fail-open handler to api/rate_limiter.py (depends on: [api/rate_limiter.py])
  - Acceptance criteria: Return allowed=True on Redis error, log warning
  - Test strategy: Test with simulated Redis failures

- [ ] Create api/routes/strategies.py (depends on: [orchestration/executor.py, orchestration/chainer.py, orchestration/comparator.py, api/auth.py])
  - Acceptance criteria: POST /execute, /chain, /compare endpoints
  - Test strategy: Integration tests with mocked strategies

- [ ] Create api/routes/evaluation.py with metrics endpoint (depends on: [evaluation/metrics.py, api/auth.py])
  - Acceptance criteria: POST /metrics calculates IR metrics
  - Test strategy: Test with sample retrieved/ground_truth data

- [ ] Add batch metrics endpoint to api/routes/evaluation.py (depends on: [api/routes/evaluation.py])
  - Acceptance criteria: POST /metrics/batch for multiple queries
  - Test strategy: Test batch processing, aggregate results

- [ ] Add async benchmark trigger to api/routes/evaluation.py (depends on: [evaluation/benchmarks.py, api/auth.py])
  - Acceptance criteria: POST /benchmark returns benchmark_id
  - Test strategy: Test task queuing, immediate response

- [ ] Add benchmark status endpoint to api/routes/evaluation.py (depends on: [api/routes/evaluation.py benchmark])
  - Acceptance criteria: GET /benchmark/{id} returns status/results
  - Test strategy: Test pending/running/complete states

- [ ] Create api/monitoring.py with Prometheus metrics (depends on: none)
  - Acceptance criteria: Request counters, latency histograms, error tracking
  - Test strategy: Verify metric registration

- [ ] Create api/main.py with FastAPI app (depends on: [api/routes/*, api/database.py, orchestration/pricing.py])
  - Acceptance criteria: Lifespan manager, OpenAPI docs, health endpoint
  - Test strategy: App startup/shutdown tests

**Exit Criteria**: API operational with authentication and rate limiting

**Delivers**: Production-ready REST API

---

### Phase 6: Infrastructure
**Goal**: Containerization and documentation

**Entry Criteria**: Phase 5 complete

**Tasks**:
- [ ] Create Dockerfile (depends on: [pyproject.toml])
  - Acceptance criteria: Multi-stage build, health check, non-root user
  - Test strategy: Build and run container

- [ ] Create docker-compose.yml (depends on: [Dockerfile, strategies/utils/schema.sql, evaluation/schema_extension.sql])
  - Acceptance criteria: Postgres, Redis, API services with health checks
  - Test strategy: docker-compose up works end-to-end

- [ ] Create docs/MIGRATION.md (depends on: [all modules])
  - Acceptance criteria: Breaking changes, code examples, rollback plan
  - Test strategy: Follow guide on fresh setup

- [ ] Create datasets/sample/ with example ground truth (depends on: [evaluation/dataset_manager.py])
  - Acceptance criteria: Valid JSON format, at least 20 queries
  - Test strategy: Load with DatasetManager

**Exit Criteria**: Full deployment capability

**Delivers**: Production-ready system

---

## Test Pyramid

```
        /\
       /E2E\       ← 10% (Full API flows with real DB)
      /------\
     /Integration\ ← 30% (Module interactions, mocked externals)
    /------------\
   /  Unit Tests  \ ← 60% (Fast, isolated, deterministic)
  /----------------\
```

## Coverage Requirements
- Line coverage: 80% minimum
- Branch coverage: 75% minimum
- orchestration module: 85% minimum
- evaluation module: 90% minimum
- api module: 75% minimum

## Critical Test Scenarios

### ChainContext (orchestration/models.py)
**Happy path**:
- Create context, add step results, verify immutability
- Expected: New instance returned, original unchanged

**Edge cases**:
- Empty intermediate_results tuple
- Very long error_log accumulation
- Expected: Handles gracefully without performance issues

**Error cases**:
- Attempt to mutate frozen dataclass
- Expected: FrozenInstanceError raised

### calculate_metrics (evaluation/metrics.py)
**Happy path**:
- Valid retrieved and ground truth, calculate all metrics
- Expected: Correct precision, recall, MRR, NDCG values

**Edge cases**:
- Empty retrieved list (None and [])
- k > len(ground_truth)
- Duplicate documents in retrieved
- Expected: Warnings generated, metrics calculated correctly

**Error cases**:
- Non-list inputs
- Non-string elements in lists
- Invalid relevance scores (not 0/1/2)
- Expected: InvalidInputError with clear message

### RedisRateLimiter (api/rate_limiter.py)
**Happy path**:
- Check rate limit, increment counter
- Expected: Returns allowed=True when under limit

**Edge cases**:
- Exactly at limit
- Window reset timing
- Expected: Correct boundary behavior

**Error cases**:
- Redis connection failure
- Pipeline partial failure
- Expected: Fail-open (allowed=True), error logged

## Test Generation Guidelines
- Use pytest-asyncio for all async tests
- Mock external services (OpenAI, Redis, PostgreSQL) in unit tests
- Use TestModel from Pydantic AI for agent testing
- Integration tests should use docker-compose test environment
- All tests must be deterministic (seed random, mock time)

---

## System Components

```
┌─────────────────────────────────────────────────────────────┐
│                      REST API (FastAPI)                      │
│  ┌─────────┐  ┌─────────┐  ┌──────────┐  ┌──────────────┐  │
│  │ Execute │  │  Chain  │  │ Compare  │  │  Evaluate    │  │
│  └────┬────┘  └────┬────┘  └────┬─────┘  └──────┬───────┘  │
└───────┼────────────┼────────────┼───────────────┼──────────┘
        │            │            │               │
┌───────┴────────────┴────────────┴───────────────┴──────────┐
│                    Orchestration Layer                      │
│  ┌──────────┐  ┌─────────┐  ┌───────────┐  ┌────────────┐  │
│  │ Executor │  │ Chainer │  │ Comparator│  │  Registry  │  │
│  └────┬─────┘  └────┬────┘  └─────┬─────┘  └────────────┘  │
└───────┼─────────────┼─────────────┼────────────────────────┘
        │             │             │
┌───────┴─────────────┴─────────────┴────────────────────────┐
│                    Strategy Layer                           │
│  ┌──────────┐  ┌──────────┐  ┌───────────┐  ┌───────────┐  │
│  │ Standard │  │ Reranking│  │ MultiQuery│  │Self-Reflec│  │
│  └──────────┘  └──────────┘  └───────────┘  └───────────┘  │
└───────────────────────────┬────────────────────────────────┘
                            │
┌───────────────────────────┴────────────────────────────────┐
│                    Infrastructure                           │
│  ┌──────────┐  ┌──────────┐  ┌───────────┐  ┌───────────┐  │
│  │PostgreSQL│  │  Redis   │  │  OpenAI   │  │ Anthropic │  │
│  │+ pgvector│  │  Cache   │  │  API      │  │    API    │  │
│  └──────────┘  └──────────┘  └───────────┘  └───────────┘  │
└────────────────────────────────────────────────────────────┘
```

## Data Models

### ChainContext (Immutable)
```python
@dataclass(frozen=True)
class ChainContext:
    query: str
    original_query: str
    intermediate_results: Tuple[MappingProxyType, ...]
    metadata: MappingProxyType
    total_cost: float
    total_latency_ms: int
    error_log: Tuple[str, ...]
    step_index: int
```

### EvaluationMetrics
```python
@dataclass
class EvaluationMetrics:
    precision: Dict[int, float]  # k -> value
    recall: Dict[int, float]
    mrr: float
    ndcg: Dict[int, float]
    warnings: List[str]
```

### API Key (Database)
```sql
CREATE TABLE api_keys (
    id UUID PRIMARY KEY,
    user_id TEXT NOT NULL,
    key_hash TEXT UNIQUE NOT NULL,
    rate_limit INT DEFAULT 100,
    is_active BOOLEAN DEFAULT true,
    expires_at TIMESTAMPTZ,
    last_used_at TIMESTAMPTZ
);
```

## Technology Stack

| Component | Technology | Rationale |
|-----------|------------|-----------|
| Agent Framework | Pydantic AI | Type-safe, structured outputs |
| API Framework | FastAPI | Async, OpenAPI, validation |
| Database | PostgreSQL + pgvector | Vector search, JSON support |
| Cache | Redis | Rate limiting, result cache |
| Embeddings | OpenAI text-embedding-3-small | Quality, cost balance |
| Reranking | sentence-transformers | Local inference, free |
| Metrics | ir-measures | Standard IR evaluation |

---

## Technical Risks

**Risk**: Redis unavailability breaks rate limiting
- **Impact**: High - API unprotected from abuse
- **Likelihood**: Medium
- **Mitigation**: Fail-open strategy with logging
- **Fallback**: In-memory rate limiter with cleanup

**Risk**: Circular dependencies between orchestration modules
- **Impact**: High - Import errors, untestable code
- **Likelihood**: Low - Clear dependency graph defined
- **Mitigation**: Strict layer boundaries, no upward imports
- **Fallback**: Dependency injection pattern

**Risk**: ir-measures library edge cases
- **Impact**: Medium - Incorrect metrics
- **Likelihood**: Low - Well-tested library
- **Mitigation**: Comprehensive validation before calling library
- **Fallback**: Manual metric implementation for edge cases

## Dependency Risks

**Risk**: OpenAI API rate limits during benchmarks
- **Impact**: Medium - Benchmarks fail or slow
- **Mitigation**: Embedding cache, batch requests
- **Fallback**: Reduce benchmark iterations

**Risk**: PostgreSQL pgvector extension not installed
- **Impact**: High - Vector search fails
- **Mitigation**: Docker image with pgvector pre-installed
- **Fallback**: Clear error message with installation instructions

## Scope Risks

**Risk**: Feature creep into strategy implementations
- **Impact**: Medium - Delays core functionality
- **Mitigation**: Keep all-rag-strategies code as-is, only add orchestration layer
- **Fallback**: Defer strategy improvements to v2

---

## References

- [all-rag-strategies](https://github.com/coleam00/ottomator-agents/tree/main/all-rag-strategies) - Base implementation
- [Pydantic AI Documentation](https://ai.pydantic.dev/) - Agent framework
- [ir-measures Documentation](https://ir-measur.es/en/latest/) - IR metrics
- [Anthropic Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) - Strategy research
- [FastAPI Documentation](https://fastapi.tiangolo.com/) - API framework

## Glossary

- **RAG**: Retrieval-Augmented Generation - combining retrieval with LLM generation
- **IR Metrics**: Information Retrieval metrics (Precision, Recall, MRR, NDCG)
- **MRR**: Mean Reciprocal Rank - average of 1/rank of first relevant result
- **NDCG**: Normalized Discounted Cumulative Gain - ranking quality metric
- **ChainContext**: Immutable state passed between chained strategies
- **Ground Truth**: Known relevant documents for evaluation queries

## Open Questions

1. Should we support custom strategy plugins from user code?
2. What's the maximum reasonable chain length before performance degrades?
3. Should benchmark results be persisted to database for historical comparison?
4. Do we need multi-tenancy with isolated datasets per API key?
