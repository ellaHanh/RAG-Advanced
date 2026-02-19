# RAG Generation Module

Overview of the **generation** stage: turning (query + retrieved documents) into a natural-language answer using a LangChain-based “stuff documents” chain.

---

## Purpose

RAG-Advanced supports three pipeline stages:

1. **Ingestion** – Index documents (chunks) into the vector store.
2. **Retrieval** – Run a strategy (or chain) to get relevant documents (`POST /execute`, `POST /chain`).
3. **Generation** – Turn the query and those documents into an answer string (`POST /generate`).

The generation module implements stage 3: it takes a list of `orchestration.models.Document` and a query, builds a prompt (context + question), calls an LLM, and returns the answer plus token counts and cost.

---

## Design

- **LangChain usage**: Only for the generation step. Uses `ChatPromptTemplate` with placeholders `{context}` and `{input}`, `create_stuff_documents_chain`, and `ChatOpenAI`. Retrieval stays as-is (existing strategies and chain executor).
- **Empty retrieval**: If no documents are returned, the API can return a fixed message (e.g. “I couldn’t find relevant information…”) or call the LLM with no context when `no_context_fallback` is true.
- **Context length**: Optional truncation so context + query fit the model’s context window; `context_truncated` is set in the result when truncation occurs.
- **Cost**: Token usage is captured via a LangChain callback; cost is computed with `orchestration.pricing` (e.g. `config/pricing.json`).

References:

- [Codecademy: Build RAG pipelines](https://www.codecademy.com/article/build-rag-pipelines-in-ai-applications)
- [ApX: Combining Retrieval and Generation](https://apxml.com/courses/getting-started-rag/chapter-5-building-basic-rag-pipeline/combining-retrieval-generation)

---

## Module layout

| File | Role |
|------|------|
| `generation/__init__.py` | Exports `generate_answer`, `GenerationResult`, `DEFAULT_MODEL`, `DEFAULT_PROMPT_TEMPLATE`. |
| `generation/chain.py` | Builds the stuff-documents chain, `generate_answer()`, truncation, token/cost handling. |
| `api/routes/generate.py` | `POST /generate`: runs retrieval then `generate_answer()`, maps to `GenerateResponse`. |

---

## Usage

### Programmatic

```python
from generation import generate_answer, GenerationResult
from orchestration.models import Document

docs = [
    Document(id="1", content="RAG combines retrieval and generation.", title="", source="", similarity=0.9),
]
result = generate_answer("What is RAG?", docs, model="gpt-4o-mini")
print(result.answer)
print(result.input_tokens, result.output_tokens, result.cost_usd)
```

### API

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"query": "What is RAG?", "strategy": "standard", "limit": 5}'
```

Response includes `answer`, `documents`, `model`, `input_tokens`, `output_tokens`, `cost_usd`, `retrieval_latency_ms`, `generation_latency_ms`, `context_truncated`.

---

## Configuration

- **Environment**: `OPENAI_API_KEY` (required). Optional: `GENERATION_MODEL` (default `gpt-4o-mini`), `GENERATION_PROMPT_TEMPLATE`.
- **Pricing**: The generation model should exist in `config/pricing.json` so cost is accurate; `gpt-4o-mini` is included by default.

---

## Inputs and outputs

- **Input**: `query` (str), `documents` (list of `orchestration.models.Document`), optional `model`, `prompt_template`, `empty_context_fallback`, `max_context_tokens`, `pricing_provider`.
- **Output**: `GenerationResult`: `answer`, `input_tokens`, `output_tokens`, `cost_usd`, `context_truncated`, `latency_ms`.
