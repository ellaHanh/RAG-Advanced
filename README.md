# RAG-Advanced

**Strategy Orchestration and Evaluation for Retrieval-Augmented Generation Systems**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-14+-blue.svg)](https://www.postgresql.org/)

Production-ready Retrieval-Augmented Generation (RAG) system with advanced retrieval strategy orchestration, generation, evaluation, and REST API layers.

---

## 📋 Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Quick Start](#-quick-start)
- [API Usage](#-api-usage)
- [Strategy Guide](#-strategy-guide)
- [Evaluation](#-evaluation)
- [Project Structure](#-project-structure)
- [Configuration](#-configuration)
- [Development](#-development)
- [Migration Guide](#-migration-guide)
- [Contributing](#-contributing)
- [Acknowledgments](#-acknowledgments)

---

## 🎯 Features

- **8 runnable RAG strategies** — Standard (baseline), reranking, multi-query, query expansion, self-reflective, agentic, contextual retrieval, context-aware chunking (7 from [all-rag-strategies](https://github.com/coleam00/ottomator-agents/tree/main/all-rag-strategies) + standard)
- **Strategy Chaining** — Execute strategies sequentially (e.g., `contextual → multi-query → reranking`)
- **Parallel Comparison** — Run multiple strategies simultaneously with side-by-side metrics
- **Automated Evaluation** — Precision@k, Recall@k, MRR, NDCG@k with ground truth validation and RAGAS generation evaluation
- **Performance Benchmarking** — Latency (p50/p95/p99), cost tracking, token usage analysis
- **REST API** — FastAPI service with authentication, rate limiting, and OpenAPI docs
- **Cost Tracking** — Per-request API cost calculation with versioned pricing models
- **Resource Management** — Semaphore-based concurrency control for database and API calls

---

## 🏗️ Architecture

### High-Level Workflow

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   API       │────▶│ Orchestration│────▶│ Strategies  │
│  (FastAPI)  │     │   Layer      │     │  & Agents   │
└─────────────┘     └──────────────┘     └─────────────┘
       │                    │                     │
       ▼                    ▼                     ▼
┌─────────────┐     ┌──────────────┐     ┌────────────------------─┐
│  Database   │     │  Generation  │     │  Retrieval              │
│  (pgvector) │     │  (LangChain) │     │ & Generation Evaluation │
└─────────────┘     └──────────────┘     └────────────------------─┘
```

### System Components

<details>
<summary>📊 View Architecture Diagram</summary>

<a href="https://mermaid.ink/svg/Zmxvd2NoYXJ0IFRCCiAgJSUgPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PQogICUlIEhpZ2gtbGV2ZWwgd29ya2Zsb3c6IGluZ2VzdCAtPiBzdG9yZSAtPiByZXRyaWV2ZS9leGVjdXRlIC0-IGV2YWx1YXRlIC0-IHJlcG9ydAogICUlID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT0KCiAgc3ViZ3JhcGggQ2xpZW50WyJDbGllbnRzIl0KICAgIENMSVsiQ0xJIC8gTm90ZWJvb2tzIl0KICAgIEFQUFsiRnJvbnRlbmQgLyBTZXJ2aWNlIl0KICBlbmQKCiAgc3ViZ3JhcGggQVBJWyJGYXN0QVBJIEFQSSAoYGFwaS9tYWluLnB5YCkiXQogICAgUk9VVEVTWyJSb3V0ZXNcbi9zdHJhdGVnaWVzLCAvZXhlY3V0ZSwgL2NoYWluLCAvY29tcGFyZVxuL21ldHJpY3MsIC9iZW5jaG1hcmtzIl0KICAgIEFVVEhbIkFQSSBLZXkgQXV0aFxuKGBhcGkvYXV0aC5weWApIl0KICAgIFJMWyJSYXRlIExpbWl0ZXIgKFJlZGlzKVxuKGBhcGkvcmF0ZV9saW1pdGVyLnB5YCkiXQogIGVuZAoKICBzdWJncmFwaCBPUkNIWyJPcmNoZXN0cmF0aW9uIExheWVyIChgb3JjaGVzdHJhdGlvbi8qYCkiXQogICAgUkVHWyJTdHJhdGVneSBSZWdpc3RyeVxuKGBvcmNoZXN0cmF0aW9uL3JlZ2lzdHJ5LnB5YCkiXQogICAgRVhFQ1siU2luZ2xlIEV4ZWN1dG9yXG4oYG9yY2hlc3RyYXRpb24vZXhlY3V0b3IucHlgKSJdCiAgICBDSEFJTlsiU2VxdWVudGlhbCBDaGFpbiBFeGVjdXRvclxuKGBvcmNoZXN0cmF0aW9uL2NoYWluX2V4ZWN1dG9yLnB5YCkiXQogICAgQ09NUFsiUGFyYWxsZWwgQ29tcGFyaXNvblxuKGBvcmNoZXN0cmF0aW9uL2NvbXBhcmlzb24ucHlgKSJdCiAgICBSRVNbIlJlc291cmNlIE1hbmFnZXIgLyBTZW1hcGhvcmVzXG4oYG9yY2hlc3RyYXRpb24vcmVzb3VyY2VfbWFuYWdlci5weWApIl0KICAgIENPU1RbIkNvc3QgVHJhY2tpbmcgKyBQcmljaW5nXG4oYG9yY2hlc3RyYXRpb24vY29zdF90cmFja2VyLnB5YCwgYG9yY2hlc3RyYXRpb24vcHJpY2luZy5weWApIl0KICBlbmQKCiAgc3ViZ3JhcGggU1RSQVRbIlJBRyBTdHJhdGVnaWVzIChgc3RyYXRlZ2llcy9hZ2VudHMvKmApIl0KICAgIFMxWyJTdHJhdGVneSBBIl0KICAgIFMyWyJTdHJhdGVneSBCIl0KICAgIFNOWyIuLi4iXQogIGVuZAoKICBzdWJncmFwaCBEQVRBWyJEYXRhICYgSW5mcmEiXQogICAgUEdbIlBvc3RncmVTUUwgKyBwZ3ZlY3RvclxuKGRvY3VtZW50cywgY2h1bmtzKVxuYHN0cmF0ZWdpZXMvdXRpbHMvc2NoZW1hLnNxbGAiXQogICAgUEdFWFRbIkV2YWx1YXRpb24gU2NoZW1hIEV4dGVuc2lvbnNcbmBldmFsdWF0aW9uL3NjaGVtYV9leHRlbnNpb24uc3FsYCJdCiAgICBSRURJU1siUmVkaXNcbihjYWNoZSArIHJhdGUgbGltaXRpbmcpIl0KICBlbmQKCiAgc3ViZ3JhcGggRVZBTFsiRXZhbHVhdGlvbiAmIEJlbmNobWFya2luZyAoYGV2YWx1YXRpb24vKmApIl0KICAgIE1FVFsiTWV0cmljc1xuKFByZWNpc2lvbi9SZWNhbGwvTVJSL05EQ0cpIl0KICAgIEJFTkNIWyJCZW5jaG1hcmtzXG4obGF0ZW5jeSArIGNvc3QgKyBtZXRyaWNzKSJdCiAgICBSRVBSVFsiUmVwb3J0c1xuKE1hcmtkb3duIC8gSFRNTCkiXQogICAgRFNFVFNbIkRhdGFzZXRzXG4oYGRhdGFzZXRzLypgKSJdCiAgZW5kCgogIHN1YmdyYXBoIElOR0VTVFsiSW5nZXN0aW9uIChgc3RyYXRlZ2llcy9pbmdlc3Rpb24vYCkiXQogICAgRE9DU1siRG9jdW1lbnRzXG4oUERGL01EL0RPQ1gvYXVkaW8pIl0KICAgIENIVU5LWyJDaHVua2luZyArIEVtYmVkZGluZ1xuKGBweXRob24gLW0gc3RyYXRlZ2llcy5pbmdlc3Rpb24uaW5nZXN0YCkiXQogIGVuZAoKICAlJSAtLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tIEluZ2VzdGlvbiBwYXRoIC0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0KICBET0NTIC0tPiBDSFVOSyAtLT4gUEcKICBQR0VYVCAtLT4gUEcKCiAgJSUgLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLSBPbmxpbmUgcmVxdWVzdCBwYXRoIC0tLS0tLS0tLS0tLS0tLS0tLS0tCiAgQ0xJIC0tPiBST1VURVMKICBBUFAgLS0-IFJPVVRFUwoKICBST1VURVMgLS0-IEFVVEggLS0-IFJMIC0tPiBSRUcKCiAgUkVHIC0tPiBFWEVDCiAgUkVHIC0tPiBDSEFJTgogIFJFRyAtLT4gQ09NUAoKICBFWEVDIC0tPiBSRVMgLS0-IFNUUkFUCiAgQ0hBSU4gLS0-IFJFUyAtLT4gU1RSQVQKICBDT01QIC0tPiBSRVMgLS0-IFNUUkFUCgogIFNUUkFUIC0tPiBQRwogIFJMIDwtLT4gUkVESVMKCiAgQ09TVCAtLT4gUk9VVEVTCgogICUlIC0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0gRXZhbHVhdGlvbiBwYXRoIC0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLQogIFJPVVRFUyAtLT4gTUVUCiAgUk9VVEVTIC0tPiBCRU5DSAoKICBEU0VUUyAtLT4gTUVUCiAgRFNFVFMgLS0-IEJFTkNICgogIEJFTkNIIC0tPiBSRVBSVAogIE1FVCAtLT4gUkVQUlQK" target="_blank">
  <img
    alt="RAG-Advanced Architecture Diagram"
    src="https://mermaid.ink/svg/Zmxvd2NoYXJ0IFRCCiAgJSUgPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PQogICUlIEhpZ2gtbGV2ZWwgd29ya2Zsb3c6IGluZ2VzdCAtPiBzdG9yZSAtPiByZXRyaWV2ZS9leGVjdXRlIC0-IGV2YWx1YXRlIC0-IHJlcG9ydAogICUlID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT0KCiAgc3ViZ3JhcGggQ2xpZW50WyJDbGllbnRzIl0KICAgIENMSVsiQ0xJIC8gTm90ZWJvb2tzIl0KICAgIEFQUFsiRnJvbnRlbmQgLyBTZXJ2aWNlIl0KICBlbmQKCiAgc3ViZ3JhcGggQVBJWyJGYXN0QVBJIEFQSSAoYGFwaS9tYWluLnB5YCkiXQogICAgUk9VVEVTWyJSb3V0ZXNcbi9zdHJhdGVnaWVzLCAvZXhlY3V0ZSwgL2NoYWluLCAvY29tcGFyZVxuL21ldHJpY3MsIC9iZW5jaG1hcmtzIl0KICAgIEFVVEhbIkFQSSBLZXkgQXV0aFxuKGBhcGkvYXV0aC5weWApIl0KICAgIFJMWyJSYXRlIExpbWl0ZXIgKFJlZGlzKVxuKGBhcGkvcmF0ZV9saW1pdGVyLnB5YCkiXQogIGVuZAoKICBzdWJncmFwaCBPUkNIWyJPcmNoZXN0cmF0aW9uIExheWVyIChgb3JjaGVzdHJhdGlvbi8qYCkiXQogICAgUkVHWyJTdHJhdGVneSBSZWdpc3RyeVxuKGBvcmNoZXN0cmF0aW9uL3JlZ2lzdHJ5LnB5YCkiXQogICAgRVhFQ1siU2luZ2xlIEV4ZWN1dG9yXG4oYG9yY2hlc3RyYXRpb24vZXhlY3V0b3IucHlgKSJdCiAgICBDSEFJTlsiU2VxdWVudGlhbCBDaGFpbiBFeGVjdXRvclxuKGBvcmNoZXN0cmF0aW9uL2NoYWluX2V4ZWN1dG9yLnB5YCkiXQogICAgQ09NUFsiUGFyYWxsZWwgQ29tcGFyaXNvblxuKGBvcmNoZXN0cmF0aW9uL2NvbXBhcmlzb24ucHlgKSJdCiAgICBSRVNbIlJlc291cmNlIE1hbmFnZXIgLyBTZW1hcGhvcmVzXG4oYG9yY2hlc3RyYXRpb24vcmVzb3VyY2VfbWFuYWdlci5weWApIl0KICAgIENPU1RbIkNvc3QgVHJhY2tpbmcgKyBQcmljaW5nXG4oYG9yY2hlc3RyYXRpb24vY29zdF90cmFja2VyLnB5YCwgYG9yY2hlc3RyYXRpb24vcHJpY2luZy5weWApIl0KICBlbmQKCiAgc3ViZ3JhcGggU1RSQVRbIlJBRyBTdHJhdGVnaWVzIChgc3RyYXRlZ2llcy9hZ2VudHMvKmApIl0KICAgIFMxWyJTdHJhdGVneSBBIl0KICAgIFMyWyJTdHJhdGVneSBCIl0KICAgIFNOWyIuLi4iXQogIGVuZAoKICBzdWJncmFwaCBEQVRBWyJEYXRhICYgSW5mcmEiXQogICAgUEdbIlBvc3RncmVTUUwgKyBwZ3ZlY3RvclxuKGRvY3VtZW50cywgY2h1bmtzKVxuYHN0cmF0ZWdpZXMvdXRpbHMvc2NoZW1hLnNxbGAiXQogICAgUEdFWFRbIkV2YWx1YXRpb24gU2NoZW1hIEV4dGVuc2lvbnNcbmBldmFsdWF0aW9uL3NjaGVtYV9leHRlbnNpb24uc3FsYCJdCiAgICBSRURJU1siUmVkaXNcbihjYWNoZSArIHJhdGUgbGltaXRpbmcpIl0KICBlbmQKCiAgc3ViZ3JhcGggRVZBTFsiRXZhbHVhdGlvbiAmIEJlbmNobWFya2luZyAoYGV2YWx1YXRpb24vKmApIl0KICAgIE1FVFsiTWV0cmljc1xuKFByZWNpc2lvbi9SZWNhbGwvTVJSL05EQ0cpIl0KICAgIEJFTkNIWyJCZW5jaG1hcmtzXG4obGF0ZW5jeSArIGNvc3QgKyBtZXRyaWNzKSJdCiAgICBSRVBSVFsiUmVwb3J0c1xuKE1hcmtkb3duIC8gSFRNTCkiXQogICAgRFNFVFNbIkRhdGFzZXRzXG4oYGRhdGFzZXRzLypgKSJdCiAgZW5kCgogIHN1YmdyYXBoIElOR0VTVFsiSW5nZXN0aW9uIChgc3RyYXRlZ2llcy9pbmdlc3Rpb24vYCkiXQogICAgRE9DU1siRG9jdW1lbnRzXG4oUERGL01EL0RPQ1gvYXVkaW8pIl0KICAgIENIVU5LWyJDaHVua2luZyArIEVtYmVkZGluZ1xuKGBweXRob24gLW0gc3RyYXRlZ2llcy5pbmdlc3Rpb24uaW5nZXN0YCkiXQogIGVuZAoKICAlJSAtLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tIEluZ2VzdGlvbiBwYXRoIC0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0KICBET0NTIC0tPiBDSFVOSyAtLT4gUEcKICBQR0VYVCAtLT4gUEcKCiAgJSUgLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLSBPbmxpbmUgcmVxdWVzdCBwYXRoIC0tLS0tLS0tLS0tLS0tLS0tLS0tCiAgQ0xJIC0tPiBST1VURVMKICBBUFAgLS0-IFJPVVRFUwoKICBST1VURVMgLS0-IEFVVEggLS0-IFJMIC0tPiBSRUcKCiAgUkVHIC0tPiBFWEVDCiAgUkVHIC0tPiBDSEFJTgogIFJFRyAtLT4gQ09NUAoKICBFWEVDIC0tPiBSRVMgLS0-IFNUUkFUCiAgQ0hBSU4gLS0-IFJFUyAtLT4gU1RSQVQKICBDT01QIC0tPiBSRVMgLS0-IFNUUkFUCgogIFNUUkFUIC0tPiBQRwogIFJMIDwtLT4gUkVESVMKCiAgQ09TVCAtLT4gUk9VVEVTCgogICUlIC0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0gRXZhbHVhdGlvbiBwYXRoIC0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLQogIFJPVVRFUyAtLT4gTUVUCiAgUk9VVEVTIC0tPiBCRU5DSAoKICBEU0VUUyAtLT4gTUVUCiAgRFNFVFMgLS0-IEJFTkNICgogIEJFTkNIIC0tPiBSRVBSVAogIE1FVCAtLT4gUkVQUlQK"
    style="width: 100%; max-width: 1200px; height: auto;"
  />
</a>

</details>

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- PostgreSQL 14+ with pgvector
- Redis 6+
- OpenAI API key

### Installation

```bash
git clone https://github.com/EllaHanh/RAG-Advanced.git
cd RAG-Advanced

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -e .
```

### Configuration

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```env
DATABASE_URL=postgresql://user:pass@localhost:5432/rag_advanced
REDIS_URL=redis://localhost:6379
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...  # Optional: for contextual retrieval
```

### Database Setup

**Option A: Docker Compose** (automatically applies schema)

```bash
docker-compose up -d
```

**Option B: Existing PostgreSQL**

```bash
# With psql installed
psql $DATABASE_URL < strategies/utils/schema.sql
psql $DATABASE_URL < evaluation/schema_extension.sql

# Without psql (Python script)
python scripts/run_schema.py
```

### Start Services

```bash
# All services (PostgreSQL + Redis + API)
docker-compose up -d

# Or: Only infrastructure, run API locally
docker-compose up -d postgres redis
uvicorn api.main:app --reload
```

### Verify Installation

```bash
# Health check
curl http://localhost:8000/health

# List strategies
curl http://localhost:8000/strategies

# API documentation
open http://localhost:8000/docs
```

### Ingest Documents

```bash
# Default: ./documents folder
python -m strategies.ingestion.ingest

# Custom folder
python -m strategies.ingestion.ingest --documents /path/to/docs

# With contextual retrieval
python -m strategies.ingestion.ingest --contextual
```

**Supported formats**: PDF, Markdown, HTML, DOCX, audio (transcription via Whisper)

See [strategies/ingestion/README.md](strategies/ingestion/README.md) for options.

---

## 🔌 API Usage

### List Available Strategies

```bash
curl http://localhost:8000/strategies
```

<details>
<summary>Response Example</summary>

```json
{
  "strategies": [
    {
      "name": "standard",
      "description": "Basic vector similarity search",
      "avg_latency_ms": 150,
      "avg_cost_usd": 0.0002
    },
    {
      "name": "reranking",
      "description": "Two-stage retrieval with cross-encoder reranking",
      "avg_latency_ms": 320,
      "avg_cost_usd": 0.0005
    }
  ]
}
```

</details>

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

<details>
<summary>Response Example</summary>

```json
{
  "strategy": "reranking",
  "documents": [
    {
      "chunk_id": "chunk-123",
      "content": "Machine learning is...",
      "score": 0.92,
      "metadata": {"source": "ml_guide.pdf", "page": 1}
    }
  ],
  "latency_ms": 285,
  "cost_usd": 0.00048,
  "timestamp": "2025-01-15T10:30:00Z"
}
```

</details>

### Chain Strategies

Execute strategies **sequentially** (same query, last step's documents returned):

```bash
curl -X POST http://localhost:8000/chain \
  -H "Content-Type: application/json" \
  -d '{
    "steps": [
      {"strategy": "multi_query", "config": {"limit": 10, "num_variations": 3}},
      {"strategy": "reranking", "config": {"limit": 5, "initial_k": 20, "final_k": 5}}
    ],
    "query": "AI ethics best practices",
    "continue_on_error": false
  }'
```

**Recommended chains**:
- **Recall → Precision**: `multi_query → reranking`
- **Fast → Precise**: `standard → reranking`
- **Contextual → Enhanced**: `contextual_retrieval → multi_query → reranking`

See [strategies/docs/README.md](strategies/docs/README.md) for all combinations.

### Compare Strategies

Run strategies in **parallel** with side-by-side metrics:

```bash
curl -X POST http://localhost:8000/compare \
  -H "Content-Type: application/json" \
  -d '{
    "strategies": ["standard", "reranking", "multi_query"],
    "query": "remote work policy",
    "ground_truth_ids": ["doc-1", "doc-2"]
  }'
```

<details>
<summary>Response Example</summary>

```json
{
  "results": [
    {
      "strategy": "reranking",
      "documents": [...],
      "latency_ms": 310,
      "cost_usd": 0.00052,
      "metrics": {
        "precision@5": 0.8,
        "recall@5": 0.75,
        "mrr": 0.95,
        "ndcg@5": 0.88
      }
    }
  ],
  "ranking": ["reranking", "multi_query", "standard"]
}
```

</details>

### API Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

---

## 📚 Strategy Guide

### Strategy Comparison

| Strategy | Use Case | Latency | Cost | Precision | Code |
|----------|----------|---------|------|-----------|------|
| **standard** | General queries, baseline | Fast | Low | Medium | [agent.py](strategies/agents/standard/agent.py) |
| **reranking** | Precision-critical tasks | Medium | Medium | High | [agent.py](strategies/agents/reranking/agent.py) |
| **multi_query** | Ambiguous/broad queries | Medium | Medium | Med-High | [agent.py](strategies/agents/multi_query/agent.py) |
| **query_expansion** | Single expanded search | Medium | Low-Med | Medium | [agent.py](strategies/agents/query_expansion/agent.py) |
| **self_reflective** | Complex research | Slow | High | High | [agent.py](strategies/agents/self_reflective/agent.py) |
| **agentic** | Full doc + chunks | Medium | Low | High | [agent.py](strategies/agents/agentic/agent.py) |
| **contextual_retrieval** | With contextual ingestion | Fast | Low | Med-High | [agent.py](strategies/agents/contextual_retrieval/agent.py) |
| **context_aware_chunking** | Docling ingestion | Fast | Low | Medium | [agent.py](strategies/agents/context_aware_chunking/agent.py) |

### When to Use Each Strategy

<details>
<summary><b>standard</b> — Baseline vector search</summary>

**Best for**: Quick prototyping, low-latency requirements, cost-sensitive applications

**Example**: FAQ chatbot, simple document search

```bash
curl -X POST http://localhost:8000/execute \
  -H "Content-Type: application/json" \
  -d '{"strategy": "standard", "query": "What is RAG?", "config": {"limit": 5}}'
```

</details>

<details>
<summary><b>reranking</b> — Two-stage retrieval with cross-encoder</summary>

**Best for**: Precision-critical applications, when initial retrieval is noisy

**Example**: Legal document search, medical Q&A

```bash
curl -X POST http://localhost:8000/execute \
  -H "Content-Type: application/json" \
  -d '{
    "strategy": "reranking",
    "query": "contract termination clauses",
    "config": {"initial_k": 20, "final_k": 5}
  }'
```

</details>

<details>
<summary><b>multi_query</b> — Generate query variations for better recall</summary>

**Best for**: Ambiguous queries, maximizing recall, diverse result sets

**Example**: Research queries, exploratory search

```bash
curl -X POST http://localhost:8000/execute \
  -H "Content-Type: application/json" \
  -d '{
    "strategy": "multi_query",
    "query": "climate change solutions",
    "config": {"num_variations": 3, "limit": 10}
  }'
```

</details>

<details>
<summary><b>self_reflective</b> — Self-correcting retrieval with quality checks</summary>

**Best for**: Complex research, when accuracy is critical

**Example**: Academic research, fact-checking

```bash
curl -X POST http://localhost:8000/execute \
  -H "Content-Type: application/json" \
  -d '{
    "strategy": "self_reflective",
    "query": "quantum computing algorithms",
    "config": {"max_iterations": 3, "limit": 5}
  }'
```

</details>

### Recommended Chains

```bash
# High recall → High precision
multi_query → reranking

# Balanced performance
query_expansion → reranking

# Maximum quality (higher cost/latency)
contextual_retrieval → multi_query → reranking

# Fast baseline
standard
```

See [docs/ALL_RAG_STRATEGIES_7_VS_RAG_ADVANCED.md](docs/ALL_RAG_STRATEGIES_7_VS_RAG_ADVANCED.md) for full strategy mapping.

---

## 📊 Evaluation

### Calculate IR Metrics (Single Query)

```bash
curl -X POST http://localhost:8000/metrics \
  -H "Content-Type: application/json" \
  -d '{
    "retrieved_ids": ["chunk-1", "chunk-2", "chunk-3"],
    "ground_truth_ids": ["chunk-1", "chunk-3"],
    "k_values": [1, 3, 5, 10]
  }'
```

<details>
<summary>Response Example</summary>

```json
{
  "precision": {"@1": 1.0, "@3": 0.67, "@5": 0.4},
  "recall": {"@1": 0.5, "@3": 1.0, "@5": 1.0},
  "ndcg": {"@1": 1.0, "@3": 0.82, "@5": 0.73},
  "mrr": 1.0
}
```

</details>

### Calculate IR Metrics (Batch)

```bash
curl -X POST "http://localhost:8000/metrics/batch?include_per_query=true" \
  -H "Content-Type: application/json" \
  -d '{
    "queries": [
      {
        "retrieved_ids": ["c1", "c2"],
        "ground_truth_ids": ["c1", "c3"],
        "k_values": [1, 3, 5]
      },
      {
        "retrieved_ids": ["c4", "c5"],
        "ground_truth_ids": ["c4", "c5"],
        "k_values": [1, 3, 5]
      }
    ]
  }'
```

### Run Benchmark (Async)

```bash
# 1. Start benchmark
BENCHMARK_ID=$(curl -X POST http://localhost:8000/benchmarks \
  -H "Content-Type: application/json" \
  -d '{
    "strategies": ["standard", "reranking", "multi_query"],
    "queries": [
      {
        "query_id": "q1",
        "query": "What is RAG?",
        "ground_truth_ids": ["doc-1", "doc-2"]
      }
    ],
    "iterations": 3,
    "timeout_seconds": 30
  }' | jq -r '.benchmark_id')

# 2. Check status
curl http://localhost:8000/benchmarks/$BENCHMARK_ID

# 3. Get results (when status == "completed")
curl http://localhost:8000/benchmarks/$BENCHMARK_ID/results

# 4. Cancel (optional)
curl -X DELETE http://localhost:8000/benchmarks/$BENCHMARK_ID
```

<details>
<summary>Benchmark Results Example</summary>

```json
{
  "benchmark_id": "bench-123",
  "status": "completed",
  "results": {
    "strategies": {
      "reranking": {
        "latency": {"p50": 285, "p95": 320, "p99": 350},
        "cost_usd": {"mean": 0.00048, "total": 0.00144},
        "metrics": {
          "precision@5": {"mean": 0.8, "std": 0.05},
          "recall@5": {"mean": 0.75, "std": 0.08},
          "mrr": {"mean": 0.95, "std": 0.02}
        }
      }
    },
    "ranking": ["reranking", "multi_query", "standard"]
  }
}
```

</details>

### RAGAS generation evaluation

Evaluate end-to-end answers (faithfulness, answer relevancy, context precision) via [RAGAS](https://docs.ragas.io/). Use when you have **question**, **contexts**, **answer**, and **ground_truth** per sample.

- **API**: `POST /evaluate/generation` with body `{ "samples": [ { "question", "contexts", "answer", "ground_truth" }, ... ] }`. Returns aggregate and optional per-sample scores plus `ragas_llm_usage` / `ragas_embedding_usage`.
- **Pipeline**: Run retrieval benchmarks then generation eval with `--run-generation-eval`; gold dataset must include an answer column (see `evaluation/config/bioasq_v1.json` for column mapping). Outputs `generation_metrics.json` in the report directory.

See [evaluation/README.md](evaluation/README.md) for RAGAS usage, Python API (`evaluation.ragas_eval.evaluate_generation`), and pipeline options.

### Metric Reference (retrieval)

| Metric | Description | Good Score | When to Use |
|--------|-------------|------------|-------------|
| **Precision@k** | Relevant docs in top-k / k | > 0.7 | When false positives are costly |
| **Recall@k** | Retrieved relevant / total relevant | > 0.8 | When missing docs is costly |
| **MRR** | Reciprocal rank of first relevant | > 0.9 | When first result matters most |
| **NDCG@k** | Ranking quality (graded relevance) | > 0.9 | When ranking order matters |

See [evaluation/README.md](evaluation/README.md) for full evaluation guide (IR metrics, benchmarks, pipeline, RAGAS).

---

## 📁 Project Structure

```
RAG-Advanced/
├── api/                    # FastAPI REST API
│   ├── main.py            # API routes and server
│   ├── auth.py            # API key authentication
│   ├── rate_limiter.py    # Redis-based rate limiting
│   └── README.md          # API documentation
├── orchestration/         # Strategy orchestration layer
│   ├── registry.py        # Strategy registration
│   ├── executor.py        # Single strategy execution
│   ├── chain_executor.py  # Sequential chaining
│   ├── comparison.py      # Parallel comparison
│   ├── cost_tracker.py    # Cost calculation
│   ├── pricing.py         # Model pricing data
│   └── README.md          # Orchestration guide
├── evaluation/            # Metrics and benchmarking
│   ├── metrics.py         # IR metrics (Precision, Recall, MRR, NDCG)
│   ├── benchmark.py       # Async benchmarking
│   ├── pipeline.py        # Evaluation pipeline
│   └── README.md          # Evaluation guide
├── strategies/            # RAG strategies (from all-rag-strategies)
│   ├── agents/            # Strategy implementations
│   │   ├── standard/
│   │   ├── reranking/
│   │   ├── multi_query/
│   │   └── ...
│   ├── ingestion/         # Document ingestion pipeline
│   ├── utils/             # Shared utilities
│   └── docs/              # Strategy documentation
├── generation/            # RAG generation (query + docs → answer)
├── config/                # Configuration files
│   └── pricing.json       # Model pricing data
├── datasets/              # Evaluation datasets
├── documents/             # Default ingestion folder
├── tests/                 # Pytest test suite
└── docs/                  # Additional documentation
```

### Key Documentation

| Module | Description | Guide |
|--------|-------------|-------|
| **api** | FastAPI REST API | [api/README.md](api/README.md) |
| **orchestration** | Strategy execution, chaining, comparison | [orchestration/README.md](orchestration/README.md) |
| **evaluation** | IR metrics, benchmarks, pipeline | [evaluation/README.md](evaluation/README.md) |
| **strategies/ingestion** | Document ingestion (PDF, DOCX, audio) | [strategies/ingestion/README.md](strategies/ingestion/README.md) |
| **generation** | RAG generation (query + docs → answer) | [generation/README.md](generation/README.md) |
| **config** | Pricing and configuration | [config/README.md](config/README.md) |
| **datasets** | Evaluation dataset format | [datasets/README.md](datasets/README.md) |
| **documents** | Default ingestion folder | [documents/README.md](documents/README.md) |
| **scripts** | run_schema, evaluation pipeline | [scripts/README.md](scripts/README.md) |
| **tests** | Pytest layout and running tests | [tests/README.md](tests/README.md) |

**Terminology**: [docs/README_terminology.md](docs/README_terminology.md)

---

## ⚙️ Configuration

### Environment Variables

| Variable | Required | Description | Default |
|----------|----------|-------------|---------|
| `DATABASE_URL` | ✅ | PostgreSQL connection string | - |
| `REDIS_URL` | ✅ | Redis connection string | - |
| `OPENAI_API_KEY` | ✅ | OpenAI API key | - |
| `ANTHROPIC_API_KEY` | ⬜ | For contextual retrieval | - |
| `LOG_LEVEL` | ⬜ | Logging level | `INFO` |
| `MAX_WORKERS` | ⬜ | Concurrent strategy workers | `4` |

### Pricing Configuration

Edit `config/pricing.json` to update model pricing:

```json
{
  "pricing_history": [
    {
      "effective_date": "2025-01-01T00:00:00Z",
      "currency": "USD",
      "models": {
        "gpt-4o-mini": {
          "input_per_1k": 0.00015,
          "output_per_1k": 0.0006
        },
        "text-embedding-3-small": {
          "input_per_1k": 0.00002,
          "output_per_1k": 0.0
        }
      }
    }
  ]
}
```

### Docker Configuration

```yaml
# docker-compose.yml
services:
  api:
    environment:
      - MAX_WORKERS=8
      - LOG_LEVEL=DEBUG
  postgres:
    environment:
      - POSTGRES_MAX_CONNECTIONS=100
  redis:
    command: redis-server --maxmemory 256mb
```

---

## 🛠️ Development

### Running Tests

```bash
# All tests
pytest tests/

# With coverage
pytest --cov=. --cov-report=html

# Specific module
pytest tests/test_orchestration/

# Parallel execution
pytest -n auto
```

### Coverage Targets

| Module | Target | Current |
|--------|--------|---------|
| orchestration | 85% | 87% |
| evaluation | 90% | 92% |
| api | 75% | 78% |
| strategies | 70% | 73% |

### Code Quality

```bash
# Format code
black .
isort .

# Lint
flake8 .
mypy .

# Pre-commit hooks
pre-commit install
pre-commit run --all-files
```

### Local Development

```bash
# Start dependencies only
docker-compose up -d postgres redis

# Run API with hot reload
uvicorn api.main:app --reload --log-level debug

# Run tests in watch mode
pytest-watch
```

---

## 🔄 Migration Guide

### From all-rag-strategies

**Breaking Changes**:
- API authentication required (`X-API-Key` header)
- Rate limiting enforced per API key
- Async-only (no sync wrappers)

**Code Changes**:

```python
# Before (all-rag-strategies)
from implementation.agents.rag_agent import execute_rag_agent
result = await execute_rag_agent(query)

# After (RAG-Advanced)
from orchestration.executor import execute_strategy
from orchestration.models import StrategyConfig
result = await execute_strategy("standard", query, StrategyConfig())
```

**Database Migration**:

```bash
# Backup existing database
pg_dump -U user -d rag_db > backup.sql

# Apply new schema extensions
psql -U user -d rag_db < evaluation/schema_extension.sql
```

See [docs/MIGRATION.md](docs/MIGRATION.md) for complete guide.

### Reference Material

- **Strategy docs**: `strategies/docs/` (11 markdown files)
- **Folder comparison**: [docs/ALL_RAG_STRATEGIES_VS_RAG_ADVANCED_COMPARISON.md](docs/ALL_RAG_STRATEGIES_VS_RAG_ADVANCED_COMPARISON.md)

---

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

### Development Process

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Make** changes with tests
4. **Run** tests: `pytest tests/`
5. **Format** code: `black . && isort .`
6. **Commit**: `git commit -m 'Add amazing feature'`
7. **Push**: `git push origin feature/amazing-feature`
8. **Open** a Pull Request

### Pull Request Guidelines

- Include tests for new features
- Update documentation
- Follow existing code style
- Add entry to CHANGELOG.md

### Bug Reports

Use the [issue tracker](https://github.com/EllaHanh/RAG-Advanced/issues) and include:
- Python version
- Steps to reproduce
- Expected vs actual behavior
- Error messages/logs

---

## 📊 Tech Stack

| Category | Technology | Purpose |
|----------|-----------|---------|
| **Framework** | Pydantic AI, FastAPI | Agent framework, REST API |
| **Database** | PostgreSQL + pgvector | Vector storage |
| **Cache** | Redis | Rate limiting, caching |
| **Embeddings** | OpenAI text-embedding-3-small | Semantic search |
| **Reranking** | sentence-transformers | Cross-encoder reranking |
| **Evaluation** | ir-measures | IR metrics (Precision, Recall, NDCG) |
| **Ingestion** | Docling | Document conversion, chunking |

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **[Cole Medin (coleam00)](https://github.com/coleam00)** — Creator of [all-rag-strategies](https://github.com/coleam00/ottomator-agents/tree/main/all-rag-strategies), the foundation for RAG-Advanced
- **[Pydantic AI](https://ai.pydantic.dev/)** — Agent framework
- **[Anthropic](https://www.anthropic.com/news/contextual-retrieval)** — Contextual retrieval methodology
- **[ir-measures](https://ir-measur.es/)** — Information retrieval metrics library

### Based On

This project extends the original [all-rag-strategies](https://github.com/coleam00/ottomator-agents/tree/main/all-rag-strategies) repository by Cole Medin with:
- **Orchestration Layer**: Strategy registry, chaining, parallel comparison
- **Evaluation Framework**: IR metrics (Precision, Recall, MRR, NDCG) and benchmarking
- **REST API**: FastAPI service with authentication and rate limiting
- **Cost Tracking**: Per-request API cost calculation with versioned pricing

---

## 📞 Support

- **Documentation**: https://github.com/EllaHanh/RAG-Advanced
- **Issues**: https://github.com/EllaHanh/RAG-Advanced/issues
- **Discussions**: https://github.com/EllaHanh/RAG-Advanced/discussions

---

<div align="center">

**[⬆ Back to Top](#rag-advanced)**

Made with ❤️ by the RAG-Advanced community

</div>