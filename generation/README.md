# Generation

RAG stage 3: (query + retrieved documents) → natural-language answer via a LangChain “stuff documents” chain.

## Quickstart

```bash
# Via API (retrieval + generation)
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" -H "X-API-Key: your-key" \
  -d '{"query": "What is RAG?", "strategy": "standard", "limit": 5}'
```

Requires `OPENAI_API_KEY`. Optional: `GENERATION_MODEL`, `GENERATION_PROMPT_TEMPLATE`.

## Features

- **Prompt** — `ChatPromptTemplate` with `{context}` and `{input}`; `create_stuff_documents_chain`, `ChatOpenAI`.
- **Empty retrieval** — Fixed message or LLM with no context when `no_context_fallback` is true.
- **Context truncation** — Optional so context + query fit model window; `context_truncated` in result.
- **Cost** — Token usage via LangChain callback; cost from `orchestration.pricing` (`config/pricing.json`).

## Usage

```python
from generation import generate_answer, GenerationResult
from orchestration.models import Document

docs = [
    Document(id="1", content="RAG combines retrieval and generation.", title="", source="", similarity=0.9),
]
result = generate_answer("What is RAG?", docs, model="gpt-4o-mini")
print(result.answer, result.input_tokens, result.cost_usd)
```

| File | Role |
|------|------|
| `generation/chain.py` | Chain build, `generate_answer()`, truncation, token/cost |
| `api/routes/generate.py` | `POST /generate` |

## Dependencies

Project deps; LangChain, OpenAI. Generation model should be in `config/pricing.json` for cost.

## Related

- [Root README](../README.md)
- [api/](../api/README.md) — `POST /generate`
- [orchestration/](../orchestration/README.md) — Document type, pricing
