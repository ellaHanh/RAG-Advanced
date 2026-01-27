# RAG-Advanced

**Strategy Orchestration and Evaluation for Retrieval-Augmented Generation Systems**

A comprehensive system for mixing, matching, chaining, and evaluating RAG strategies. Built on top of [all-rag-strategies](https://github.com/coleam00/ottomator-agents/tree/main/all-rag-strategies) with an orchestration layer and REST API.

## Based On

This project is based on the original [**all-rag-strategies**](https://github.com/coleam00/ottomator-agents/tree/main/all-rag-strategies) repository by [**Cole Medin (coleam00)**](https://github.com/coleam00), which provides excellent implementations of 11 RAG strategies using Pydantic AI and PostgreSQL with pgvector.

RAG-Advanced extends the original work by adding:
- **Orchestration Layer**: Strategy registry, chaining, and parallel comparison
- **Evaluation Framework**: IR metrics (Precision, Recall, MRR, NDCG) and benchmarking
- **REST API**: FastAPI service with authentication and rate limiting
- **Cost Tracking**: Per-request API cost calculation with versioned pricing

We are grateful to Cole Medin for creating the foundational RAG strategy implementations that made this project possible.

## Features

- **11 RAG Strategies**: Standard search, reranking, multi-query, self-reflective, agentic, and more
- **Strategy Chaining**: Execute strategies sequentially (e.g., contextual → multi-query → reranking)
- **Parallel Comparison**: Run multiple strategies and compare results with metrics
- **Automated Evaluation**: Precision@k, Recall@k, MRR, NDCG@k
- **Performance Benchmarking**: Latency, cost, token usage tracking
- **REST API**: FastAPI service with OpenAPI documentation

## Architecture (Workflow Pipeline)

> If your Markdown viewer doesn't support Mermaid, the diagram below is rendered as an SVG and should still work.
> Click to open full-size.

<a href="https://mermaid.ink/svg/Zmxvd2NoYXJ0IFRCCiAgJSUgPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PQogICUlIEhpZ2gtbGV2ZWwgd29ya2Zsb3c6IGluZ2VzdCAtPiBzdG9yZSAtPiByZXRyaWV2ZS9leGVjdXRlIC0-IGV2YWx1YXRlIC0-IHJlcG9ydAogICUlID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT0KCiAgc3ViZ3JhcGggQ2xpZW50WyJDbGllbnRzIl0KICAgIENMSVsiQ0xJIC8gTm90ZWJvb2tzIl0KICAgIEFQUFsiRnJvbnRlbmQgLyBTZXJ2aWNlIl0KICBlbmQKCiAgc3ViZ3JhcGggQVBJWyJGYXN0QVBJIEFQSSAoYGFwaS9tYWluLnB5YCkiXQogICAgUk9VVEVTWyJSb3V0ZXNcbi9zdHJhdGVnaWVzLCAvZXhlY3V0ZSwgL2NoYWluLCAvY29tcGFyZVxuL21ldHJpY3MsIC9iZW5jaG1hcmtzIl0KICAgIEFVVEhbIkFQSSBLZXkgQXV0aFxuKGBhcGkvYXV0aC5weWApIl0KICAgIFJMWyJSYXRlIExpbWl0ZXIgKFJlZGlzKVxuKGBhcGkvcmF0ZV9saW1pdGVyLnB5YCkiXQogIGVuZAoKICBzdWJncmFwaCBPUkNIWyJPcmNoZXN0cmF0aW9uIExheWVyIChgb3JjaGVzdHJhdGlvbi8qYCkiXQogICAgUkVHWyJTdHJhdGVneSBSZWdpc3RyeVxuKGBvcmNoZXN0cmF0aW9uL3JlZ2lzdHJ5LnB5YCkiXQogICAgRVhFQ1siU2luZ2xlIEV4ZWN1dG9yXG4oYG9yY2hlc3RyYXRpb24vZXhlY3V0b3IucHlgKSJdCiAgICBDSEFJTlsiU2VxdWVudGlhbCBDaGFpbiBFeGVjdXRvclxuKGBvcmNoZXN0cmF0aW9uL2NoYWluX2V4ZWN1dG9yLnB5YCkiXQogICAgQ09NUFsiUGFyYWxsZWwgQ29tcGFyaXNvblxuKGBvcmNoZXN0cmF0aW9uL2NvbXBhcmlzb24ucHlgKSJdCiAgICBSRVNbIlJlc291cmNlIE1hbmFnZXIgLyBTZW1hcGhvcmVzXG4oYG9yY2hlc3RyYXRpb24vcmVzb3VyY2VfbWFuYWdlci5weWApIl0KICAgIENPU1RbIkNvc3QgVHJhY2tpbmcgKyBQcmljaW5nXG4oYG9yY2hlc3RyYXRpb24vY29zdF90cmFja2VyLnB5YCwgYG9yY2hlc3RyYXRpb24vcHJpY2luZy5weWApIl0KICBlbmQKCiAgc3ViZ3JhcGggU1RSQVRbIlJBRyBTdHJhdGVnaWVzIChgc3RyYXRlZ2llcy9hZ2VudHMvKmApIl0KICAgIFMxWyJTdHJhdGVneSBBIl0KICAgIFMyWyJTdHJhdGVneSBCIl0KICAgIFNOWyIuLi4iXQogIGVuZAoKICBzdWJncmFwaCBEQVRBWyJEYXRhICYgSW5mcmEiXQogICAgUEdbIlBvc3RncmVTUUwgKyBwZ3ZlY3RvclxuKGRvY3VtZW50cywgY2h1bmtzKVxuYHN0cmF0ZWdpZXMvdXRpbHMvc2NoZW1hLnNxbGAiXQogICAgUEdFWFRbIkV2YWx1YXRpb24gU2NoZW1hIEV4dGVuc2lvbnNcbmBldmFsdWF0aW9uL3NjaGVtYV9leHRlbnNpb24uc3FsYCJdCiAgICBSRURJU1siUmVkaXNcbihjYWNoZSArIHJhdGUgbGltaXRpbmcpIl0KICBlbmQKCiAgc3ViZ3JhcGggRVZBTFsiRXZhbHVhdGlvbiAmIEJlbmNobWFya2luZyAoYGV2YWx1YXRpb24vKmApIl0KICAgIE1FVFsiTWV0cmljc1xuKFByZWNpc2lvbi9SZWNhbGwvTVJSL05EQ0cpIl0KICAgIEJFTkNIWyJCZW5jaG1hcmtzXG4obGF0ZW5jeSArIGNvc3QgKyBtZXRyaWNzKSJdCiAgICBSRVBSVFsiUmVwb3J0c1xuKE1hcmtkb3duIC8gSFRNTCkiXQogICAgRFNFVFNbIkRhdGFzZXRzXG4oYGRhdGFzZXRzLypgKSJdCiAgZW5kCgogIHN1YmdyYXBoIElOR0VTVFsiSW5nZXN0aW9uIChXSVApIl0KICAgIERPQ1NbIkRvY3VtZW50c1xuKFBERi9NRC9IVE1ML2V0Yy4pIl0KICAgIENIVU5LWyJDaHVua2luZyArIEVtYmVkZGluZ1xuKFRCRCkiXQogIGVuZAoKICAlJSAtLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tIEluZ2VzdGlvbiBwYXRoIC0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0KICBET0NTIC0tPiBDSFVOSyAtLT4gUEcKICBQR0VYVCAtLT4gUEcKCiAgJSUgLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLSBPbmxpbmUgcmVxdWVzdCBwYXRoIC0tLS0tLS0tLS0tLS0tLS0tLS0tCiAgQ0xJIC0tPiBST1VURVMKICBBUFAgLS0-IFJPVVRFUwoKICBST1VURVMgLS0-IEFVVEggLS0-IFJMIC0tPiBSRUcKCiAgUkVHIC0tPiBFWEVDCiAgUkVHIC0tPiBDSEFJTgogIFJFRyAtLT4gQ09NUAoKICBFWEVDIC0tPiBSRVMgLS0-IFNUUkFUCiAgQ0hBSU4gLS0-IFJFUyAtLT4gU1RSQVQKICBDT01QIC0tPiBSRVMgLS0-IFNUUkFUCgogIFNUUkFUIC0tPiBQRwogIFJMIDwtLT4gUkVESVMKCiAgQ09TVCAtLT4gUk9VVEVTCgogICUlIC0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0gRXZhbHVhdGlvbiBwYXRoIC0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLQogIFJPVVRFUyAtLT4gTUVUCiAgUk9VVEVTIC0tPiBCRU5DSAoKICBEU0VUUyAtLT4gTUVUCiAgRFNFVFMgLS0-IEJFTkNICgogIEJFTkNIIC0tPiBSRVBSVAogIE1FVCAtLT4gUkVQUlQK)" target="_blank" rel="noopener noreferrer">
  <img
    alt="RAG-Advanced Architecture Diagram"
    src="https://mermaid.ink/svg/Zmxvd2NoYXJ0IFRCCiAgJSUgPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PQogICUlIEhpZ2gtbGV2ZWwgd29ya2Zsb3c6IGluZ2VzdCAtPiBzdG9yZSAtPiByZXRyaWV2ZS9leGVjdXRlIC0-IGV2YWx1YXRlIC0-IHJlcG9ydAogICUlID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT0KCiAgc3ViZ3JhcGggQ2xpZW50WyJDbGllbnRzIl0KICAgIENMSVsiQ0xJIC8gTm90ZWJvb2tzIl0KICAgIEFQUFsiRnJvbnRlbmQgLyBTZXJ2aWNlIl0KICBlbmQKCiAgc3ViZ3JhcGggQVBJWyJGYXN0QVBJIEFQSSAoYGFwaS9tYWluLnB5YCkiXQogICAgUk9VVEVTWyJSb3V0ZXNcbi9zdHJhdGVnaWVzLCAvZXhlY3V0ZSwgL2NoYWluLCAvY29tcGFyZVxuL21ldHJpY3MsIC9iZW5jaG1hcmtzIl0KICAgIEFVVEhbIkFQSSBLZXkgQXV0aFxuKGBhcGkvYXV0aC5weWApIl0KICAgIFJMWyJSYXRlIExpbWl0ZXIgKFJlZGlzKVxuKGBhcGkvcmF0ZV9saW1pdGVyLnB5YCkiXQogIGVuZAoKICBzdWJncmFwaCBPUkNIWyJPcmNoZXN0cmF0aW9uIExheWVyIChgb3JjaGVzdHJhdGlvbi8qYCkiXQogICAgUkVHWyJTdHJhdGVneSBSZWdpc3RyeVxuKGBvcmNoZXN0cmF0aW9uL3JlZ2lzdHJ5LnB5YCkiXQogICAgRVhFQ1siU2luZ2xlIEV4ZWN1dG9yXG4oYG9yY2hlc3RyYXRpb24vZXhlY3V0b3IucHlgKSJdCiAgICBDSEFJTlsiU2VxdWVudGlhbCBDaGFpbiBFeGVjdXRvclxuKGBvcmNoZXN0cmF0aW9uL2NoYWluX2V4ZWN1dG9yLnB5YCkiXQogICAgQ09NUFsiUGFyYWxsZWwgQ29tcGFyaXNvblxuKGBvcmNoZXN0cmF0aW9uL2NvbXBhcmlzb24ucHlgKSJdCiAgICBSRVNbIlJlc291cmNlIE1hbmFnZXIgLyBTZW1hcGhvcmVzXG4oYG9yY2hlc3RyYXRpb24vcmVzb3VyY2VfbWFuYWdlci5weWApIl0KICAgIENPU1RbIkNvc3QgVHJhY2tpbmcgKyBQcmljaW5nXG4oYG9yY2hlc3RyYXRpb24vY29zdF90cmFja2VyLnB5YCwgYG9yY2hlc3RyYXRpb24vcHJpY2luZy5weWApIl0KICBlbmQKCiAgc3ViZ3JhcGggU1RSQVRbIlJBRyBTdHJhdGVnaWVzIChgc3RyYXRlZ2llcy9hZ2VudHMvKmApIl0KICAgIFMxWyJTdHJhdGVneSBBIl0KICAgIFMyWyJTdHJhdGVneSBCIl0KICAgIFNOWyIuLi4iXQogIGVuZAoKICBzdWJncmFwaCBEQVRBWyJEYXRhICYgSW5mcmEiXQogICAgUEdbIlBvc3RncmVTUUwgKyBwZ3ZlY3RvclxuKGRvY3VtZW50cywgY2h1bmtzKVxuYHN0cmF0ZWdpZXMvdXRpbHMvc2NoZW1hLnNxbGAiXQogICAgUEdFWFRbIkV2YWx1YXRpb24gU2NoZW1hIEV4dGVuc2lvbnNcbmBldmFsdWF0aW9uL3NjaGVtYV9leHRlbnNpb24uc3FsYCJdCiAgICBSRURJU1siUmVkaXNcbihjYWNoZSArIHJhdGUgbGltaXRpbmcpIl0KICBlbmQKCiAgc3ViZ3JhcGggRVZBTFsiRXZhbHVhdGlvbiAmIEJlbmNobWFya2luZyAoYGV2YWx1YXRpb24vKmApIl0KICAgIE1FVFsiTWV0cmljc1xuKFByZWNpc2lvbi9SZWNhbGwvTVJSL05EQ0cpIl0KICAgIEJFTkNIWyJCZW5jaG1hcmtzXG4obGF0ZW5jeSArIGNvc3QgKyBtZXRyaWNzKSJdCiAgICBSRVBSVFsiUmVwb3J0c1xuKE1hcmtkb3duIC8gSFRNTCkiXQogICAgRFNFVFNbIkRhdGFzZXRzXG4oYGRhdGFzZXRzLypgKSJdCiAgZW5kCgogIHN1YmdyYXBoIElOR0VTVFsiSW5nZXN0aW9uIChXSVApIl0KICAgIERPQ1NbIkRvY3VtZW50c1xuKFBERi9NRC9IVE1ML2V0Yy4pIl0KICAgIENIVU5LWyJDaHVua2luZyArIEVtYmVkZGluZ1xuKFRCRCkiXQogIGVuZAoKICAlJSAtLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tIEluZ2VzdGlvbiBwYXRoIC0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0KICBET0NTIC0tPiBDSFVOSyAtLT4gUEcKICBQR0VYVCAtLT4gUEcKCiAgJSUgLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLSBPbmxpbmUgcmVxdWVzdCBwYXRoIC0tLS0tLS0tLS0tLS0tLS0tLS0tCiAgQ0xJIC0tPiBST1VURVMKICBBUFAgLS0-IFJPVVRFUwoKICBST1VURVMgLS0-IEFVVEggLS0-IFJMIC0tPiBSRUcKCiAgUkVHIC0tPiBFWEVDCiAgUkVHIC0tPiBDSEFJTgogIFJFRyAtLT4gQ09NUAoKICBFWEVDIC0tPiBSRVMgLS0-IFNUUkFUCiAgQ0hBSU4gLS0-IFJFUyAtLT4gU1RSQVQKICBDT01QIC0tPiBSRVMgLS0-IFNUUkFUCgogIFNUUkFUIC0tPiBQRwogIFJMIDwtLT4gUkVESVMKCiAgQ09TVCAtLT4gUk9VVEVTCgogICUlIC0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0gRXZhbHVhdGlvbiBwYXRoIC0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLQogIFJPVVRFUyAtLT4gTUVUCiAgUk9VVEVTIC0tPiBCRU5DSAoKICBEU0VUUyAtLT4gTUVUCiAgRFNFVFMgLS0-IEJFTkNICgogIEJFTkNIIC0tPiBSRVBSVAogIE1FVCAtLT4gUkVQUlQK)" 
    style="width: 100%; max-width: 1200px; height: auto;"
  />
</a>

<details>
<summary>Mermaid source (editable)</summary>

```mermaid
flowchart TB
  %% ============================================================================
  %% High-level workflow: ingest -> store -> retrieve/execute -> evaluate -> report
  %% ============================================================================

  subgraph Client["Clients"]
    CLI["CLI / Notebooks"]
    APP["Frontend / Service"]
  end

  subgraph API["FastAPI API (`api/main.py`)"]
    ROUTES["Routes\n/strategies, /execute, /chain, /compare\n/metrics, /benchmarks"]
    AUTH["API Key Auth\n(`api/auth.py`)"]
    RL["Rate Limiter (Redis)\n(`api/rate_limiter.py`)"]
  end

  subgraph ORCH["Orchestration Layer (`orchestration/*`)"]
    REG["Strategy Registry\n(`orchestration/registry.py`)"]
    EXEC["Single Executor\n(`orchestration/executor.py`)"]
    CHAIN["Sequential Chain Executor\n(`orchestration/chain_executor.py`)"]
    COMP["Parallel Comparison\n(`orchestration/comparison.py`)"]
    RES["Resource Manager / Semaphores\n(`orchestration/resource_manager.py`)"]
    COST["Cost Tracking + Pricing\n(`orchestration/cost_tracker.py`, `orchestration/pricing.py`)"]
  end

  subgraph STRAT["RAG Strategies (`strategies/agents/*`)"]
    S1["Strategy A"]
    S2["Strategy B"]
    SN["..."]
  end

  subgraph DATA["Data & Infra"]
    PG["PostgreSQL + pgvector\n(documents, chunks)\n`strategies/utils/schema.sql`"]
    PGEXT["Evaluation Schema Extensions\n`evaluation/schema_extension.sql`"]
    REDIS["Redis\n(cache + rate limiting)"]
  end

  subgraph EVAL["Evaluation & Benchmarking (`evaluation/*`)"]
    MET["Metrics\n(Precision/Recall/MRR/NDCG)"]
    BENCH["Benchmarks\n(latency + cost + metrics)"]
    REPRT["Reports\n(Markdown / HTML)"]
    DSETS["Datasets\n(`datasets/*`)"]
  end

  subgraph INGEST["Ingestion (WIP)"]
    DOCS["Documents\n(PDF/MD/HTML/etc.)"]
    CHUNK["Chunking + Embedding\n(TBD)"]
  end

  %% ------------------------- Ingestion path -------------------------
  DOCS --> CHUNK --> PG
  PGEXT --> PG

  %% ------------------------- Online request path --------------------
  CLI --> ROUTES
  APP --> ROUTES

  ROUTES --> AUTH --> RL --> REG

  REG --> EXEC
  REG --> CHAIN
  REG --> COMP

  EXEC --> RES --> STRAT
  CHAIN --> RES --> STRAT
  COMP --> RES --> STRAT

  STRAT --> PG
  RL <--> REDIS

  COST --> ROUTES

  %% ------------------------- Evaluation path ------------------------
  ROUTES --> MET
  ROUTES --> BENCH

  DSETS --> MET
  DSETS --> BENCH

  BENCH --> REPRT
  MET --> REPRT
```

</details>

## Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/CuulCat/RAG-Advanced.git
cd RAG-Advanced

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -e .
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your credentials:
# - DATABASE_URL=postgresql://user:pass@localhost:5432/rag_advanced
# - REDIS_URL=redis://localhost:6379
# - OPENAI_API_KEY=sk-...
```

### 3. Setup Database

```bash
# Run base schema
psql $DATABASE_URL < strategies/utils/schema.sql

# Run evaluation extensions
psql $DATABASE_URL < evaluation/schema_extension.sql
```

### 4. Start Services

```bash
# Using Docker Compose (recommended)
docker-compose up -d

# Or start API manually
uvicorn api.main:app --reload
```

### 5. Ingest Documents (Coming Soon)

Document ingestion pipeline is under development. For now, you can manually insert documents:

```sql
-- Insert a document
INSERT INTO documents (title, source, content, metadata)
VALUES ('My Document', 'local/file.pdf', 'Document content...', '{}');

-- Use your preferred method to chunk and embed content
```

## API Usage

### List Available Strategies

```bash
curl http://localhost:8000/strategies
```

### Execute Single Strategy

```bash
curl -X POST http://localhost:8000/execute \
  -H "Content-Type: application/json" \
  -d '{
    "strategy": "reranking",
    "query": "What is machine learning?",
    "config": {
      "initial_k": 20,
      "final_k": 5
    }
  }'
```

### Chain Strategies

```bash
curl -X POST http://localhost:8000/chain \
  -H "Content-Type: application/json" \
  -d '{
    "steps": [
      {"strategy": "multi_query", "config": {"num_variations": 3}},
      {"strategy": "reranking", "config": {"final_k": 5}}
    ],
    "query": "AI ethics best practices"
  }'
```

### Compare Strategies

```bash
curl -X POST http://localhost:8000/compare \
  -H "Content-Type: application/json" \
  -d '{
    "strategies": ["standard", "reranking", "multi_query"],
    "query": "remote work policy"
  }'
```

### Health Check

```bash
curl http://localhost:8000/health
```

### API Documentation

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- OpenAPI JSON: http://localhost:8000/openapi.json

## Strategy Guide

| Strategy | Use Case | Latency | Cost | Precision |
|----------|----------|---------|------|-----------|
| standard | General queries | Fast | Low | Medium |
| reranking | Precision-critical | Medium | Medium | High |
| multi_query | Ambiguous queries | Medium | Medium | Medium-High |
| self_reflective | Complex research | Slow | High | High |
| agentic | Flexible needs | Medium | Medium | High |
| contextual_retrieval | Best accuracy | Slow | High | Highest |

### Recommended Chains

**High Accuracy** (slow, expensive):
```
contextual_retrieval → multi_query → reranking
```

**Balanced** (moderate):
```
query_expansion → standard → reranking
```

**Fast** (cost-effective):
```
standard
```

## Evaluation Metrics

| Metric | Description | Good Score |
|--------|-------------|------------|
| Precision@k | Relevant docs in top-k / k | > 0.7 |
| Recall@k | Retrieved relevant / total relevant | > 0.8 |
| MRR | Reciprocal rank of first relevant | > 0.9 |
| NDCG@k | Ranking quality (graded relevance) | > 0.9 |

## Project Structure

```
RAG-Advanced/
├── api/                       # FastAPI REST API
│   ├── main.py               # Application entry & route registration
│   ├── auth.py               # API key authentication
│   ├── rate_limiter.py       # Redis sliding window rate limiting
│   └── routes/               # API endpoints
│       ├── strategies.py     # Execute, chain, compare
│       ├── evaluation.py     # Metrics endpoints
│       └── benchmarks.py     # Benchmark management
│
├── orchestration/             # Strategy orchestration
│   ├── executor.py           # Single strategy execution
│   ├── chain_executor.py     # Sequential chaining
│   ├── comparison.py         # Parallel comparison
│   ├── pricing.py            # Cost calculation
│   ├── cost_tracker.py       # Per-request cost tracking
│   ├── registry.py           # Strategy registry
│   ├── resource_manager.py   # Semaphore & concurrency control
│   ├── models.py             # ChainContext, configs
│   └── errors.py             # Custom exceptions
│
├── evaluation/                # Metrics and benchmarking
│   ├── metrics.py            # IR metrics (Precision, Recall, MRR, NDCG)
│   ├── benchmarks.py         # Performance benchmarking
│   ├── datasets.py           # Test dataset management
│   ├── reports.py            # Markdown report generation
│   └── html_reports.py       # HTML report generation
│
├── strategies/                # RAG strategies
│   ├── agents/               # Strategy implementations
│   ├── ingestion/            # Document processing (TBD)
│   └── utils/
│       ├── embedding_cache.py # LRU embedding cache
│       ├── result_cache.py   # Query result TTL cache
│       └── schema.sql        # Base database schema
│
├── config/
│   └── pricing.json          # Versioned API pricing
│
├── datasets/                  # Test datasets
│   └── sample/               # Example ground truth
│
├── tests/                     # Test suite (485+ tests)
│
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── .env.example
```

## Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `DATABASE_URL` | Yes | PostgreSQL connection string |
| `REDIS_URL` | Yes | Redis connection string |
| `OPENAI_API_KEY` | Yes | OpenAI API key |
| `ANTHROPIC_API_KEY` | No | For contextual retrieval |

### Pricing Configuration

Edit `config/pricing.json` to update model pricing:

```json
{
  "pricing_history": [
    {
      "effective_date": "2025-01-01T00:00:00Z",
      "currency": "USD",
      "models": {
        "gpt-4o-mini": {"input_per_1k": 0.00015, "output_per_1k": 0.0006},
        "text-embedding-3-small": {"input_per_1k": 0.00002, "output_per_1k": 0.0}
      }
    }
  ]
}
```

## Docker Deployment

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f api

# Stop services
docker-compose down
```

## Migration from all-rag-strategies

If you're migrating from the original all-rag-strategies repository:

### Breaking Changes

- API authentication now required (X-API-Key header)
- Rate limiting enforced per API key
- Async-only (no sync wrappers)

### Code Changes

```python
# Before (all-rag-strategies)
from implementation.agents.rag_agent import execute_rag_agent
result = await execute_rag_agent(query)

# After (RAG-Advanced)
from orchestration.executor import execute_strategy
from orchestration.models import StrategyConfig
result = await execute_strategy("standard", query, StrategyConfig())
```

### Database Migration

```bash
# Backup existing database
pg_dump -U user -d rag_db > backup.sql

# Apply new schema extensions
psql -U user -d rag_db < evaluation/schema_extension.sql
```

See [docs/MIGRATION.md](docs/MIGRATION.md) for complete migration guide.

## Development

### Running Tests

```bash
# All tests
pytest tests/

# With coverage
pytest --cov=. --cov-report=html

# Specific module
pytest tests/test_orchestration/
```

### Coverage Targets

| Module | Target |
|--------|--------|
| orchestration | 85% |
| evaluation | 90% |
| api | 75% |
| strategies | 70% |

## Tech Stack

- **Framework**: Pydantic AI, FastAPI
- **Database**: PostgreSQL + pgvector
- **Cache**: Redis
- **Embeddings**: OpenAI text-embedding-3-small
- **Reranking**: sentence-transformers cross-encoder
- **Evaluation**: ir-measures

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Submit a pull request

## License

MIT License - see LICENSE file

## Acknowledgments

- [**Cole Medin (coleam00)**](https://github.com/coleam00) - Creator of [all-rag-strategies](https://github.com/coleam00/ottomator-agents/tree/main/all-rag-strategies), the foundation for this project
- [Pydantic AI](https://ai.pydantic.dev/) - Agent framework
- [Anthropic](https://www.anthropic.com/news/contextual-retrieval) - Contextual retrieval methodology
- [ir-measures](https://ir-measur.es/) - Information retrieval metrics library
