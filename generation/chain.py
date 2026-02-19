"""
RAG-Advanced Generation Chain (LangChain).

Builds a "stuff documents" chain: prompt template (context + question) and LLM to generate
an answer from retrieved documents. Supports token/cost tracking and optional context truncation.

Key features:
- ChatPromptTemplate with {context} and {input}; create_stuff_documents_chain + ChatOpenAI.
- Converts orchestration.models.Document to LangChain Document format.
- Returns answer, input_tokens, output_tokens, cost_usd; optional context_truncated.

Usage:
    from generation import generate_answer, GenerationResult
    from orchestration.models import Document

    docs = [Document(id="1", content="RAG combines retrieval and generation.")]
    result = generate_answer("What is RAG?", docs, model="gpt-4o-mini")
    print(result.answer, result.cost_usd)
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any

try:
    from langchain_classic.chains.combine_documents import create_stuff_documents_chain
except ModuleNotFoundError:
    from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document as LCDocument
from langchain_core.outputs import LLMResult
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from orchestration.models import Document as OrchDocument
from orchestration.pricing import get_pricing_provider_sync

logger = logging.getLogger(__name__)

# Default prompt: context + question (Codecademy-style RAG)
DEFAULT_PROMPT_TEMPLATE = (
    "Answer the question based only on the following context.\n\n"
    "Context:\n{context}\n\n"
    "Question: {input}\n\n"
    "Answer:"
)

# Default model for generation (cost-effective)
DEFAULT_MODEL = "gpt-4o-mini"

# Approximate chars per token for truncation (conservative)
CHARS_PER_TOKEN_ESTIMATE = 4

# Typical context windows (tokens) for common models (leave margin for prompt + response)
DEFAULT_CONTEXT_WINDOW = 128_000
MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    "gpt-4o-mini": 128_000,
    "gpt-4o": 128_000,
    "gpt-4-turbo": 128_000,
    "gpt-3.5-turbo": 16_385,
}


class _TokenUsageCallbackHandler(BaseCallbackHandler):
    """Captures input/output token counts from LLMResult in on_llm_end."""

    input_tokens: int = 0
    output_tokens: int = 0

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        if not response.generations:
            return
        gen = response.generations[0][0]
        msg = getattr(gen, "message", None)
        if msg is None:
            return
        usage = getattr(msg, "usage_metadata", None) or {}
        self.input_tokens = usage.get("input_tokens") or usage.get("prompt_tokens", 0)
        self.output_tokens = usage.get("output_tokens") or usage.get("completion_tokens", 0)


@dataclass(frozen=True)
class GenerationResult:
    """
    Result of RAG generation.

    Attributes:
        answer: Generated answer text.
        input_tokens: LLM input tokens used.
        output_tokens: LLM output tokens used.
        cost_usd: Estimated cost in USD.
        context_truncated: True if context was truncated to fit model window.
        latency_ms: Generation latency in milliseconds.
    """

    answer: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    context_truncated: bool = False
    latency_ms: int = 0


def _orch_docs_to_langchain(docs: list[OrchDocument]) -> list[LCDocument]:
    """Convert orchestration Documents to LangChain Documents (page_content + metadata)."""
    return [
        LCDocument(
            page_content=d.content,
            metadata={"id": d.id, "title": d.title, "source": d.source, **d.metadata},
        )
        for d in docs
    ]


def _get_context_window(model: str) -> int:
    """Return context window size in tokens for the model."""
    return MODEL_CONTEXT_WINDOWS.get(model, DEFAULT_CONTEXT_WINDOW)


def _truncate_context_to_fit(
    docs: list[OrchDocument],
    query: str,
    prompt_template: str,
    model: str,
    reserve_output: int = 2048,
) -> tuple[list[OrchDocument], bool]:
    """
    Truncate documents so context + query + template fit within model context window.

    Drops lowest-ranked (last) documents until fit. Returns (truncated_docs, was_truncated).
    """
    window = _get_context_window(model)
    available = window - reserve_output  # reserve for system/response
    # Approximate template size (no docs)
    template_only = prompt_template.replace("{context}", "").replace("{input}", query)
    used = len(template_only) // CHARS_PER_TOKEN_ESTIMATE + len(query) // CHARS_PER_TOKEN_ESTIMATE
    remaining = available - used
    if remaining <= 0:
        return [], True

    truncated = False
    total_chars = 0
    result: list[OrchDocument] = []
    for doc in docs:
        need = len(doc.content) // CHARS_PER_TOKEN_ESTIMATE
        if total_chars // CHARS_PER_TOKEN_ESTIMATE + need <= remaining:
            result.append(doc)
            total_chars += len(doc.content)
        else:
            truncated = True
            break
    return result, truncated


def _build_chain(
    model: str,
    prompt_template: str,
    temperature: float = 0.0,
) -> Any:
    """Build the stuff-documents chain (LLM + prompt)."""
    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    prompt = ChatPromptTemplate.from_template(prompt_template)
    return create_stuff_documents_chain(llm, prompt)


def _cost_from_tokens(model: str, input_tokens: int, output_tokens: int) -> float:
    """Compute cost in USD using pricing provider."""
    provider = get_pricing_provider_sync()
    if not provider:
        return 0.0
    pricing = provider.get_model_pricing(model)
    return pricing.calculate_cost(input_tokens=input_tokens, output_tokens=output_tokens)


def generate_answer(
    query: str,
    documents: list[OrchDocument],
    *,
    model: str | None = None,
    prompt_template: str | None = None,
    no_context_message: str = "I couldn't find relevant information to answer your query.",
    empty_context_fallback: bool = False,
    max_context_tokens: bool = True,
    pricing_provider: Any = None,
) -> GenerationResult:
    """
    Generate an answer from a query and retrieved documents using a LangChain stuff-documents chain.

    Args:
        query: User question.
        documents: Retrieved documents (orchestration.models.Document).
        model: LLM model name (default from env GENERATION_MODEL or gpt-4o-mini).
        prompt_template: Prompt with {context} and {input} (default RAG template).
        no_context_message: Returned as answer when documents is empty and empty_context_fallback is False.
        empty_context_fallback: If True and documents is empty, call LLM with empty context.
        max_context_tokens: If True, truncate context to fit model window and set context_truncated.
        pricing_provider: Optional PricingProvider for cost; uses get_pricing_provider_sync if None.

    Returns:
        GenerationResult with answer, token counts, cost_usd, and optional context_truncated.

    Raises:
        Exception: On LLM or chain invocation errors (caller should handle).
    """
    model = model or os.getenv("GENERATION_MODEL", DEFAULT_MODEL)
    prompt_template = prompt_template or os.getenv(
        "GENERATION_PROMPT_TEMPLATE", DEFAULT_PROMPT_TEMPLATE
    )

    if not documents and not empty_context_fallback:
        return GenerationResult(
            answer=no_context_message,
            input_tokens=0,
            output_tokens=0,
            cost_usd=0.0,
            context_truncated=False,
            latency_ms=0,
        )

    docs = list(documents) if documents else []
    context_truncated = False
    if docs and max_context_tokens:
        docs, context_truncated = _truncate_context_to_fit(
            docs, query, prompt_template, model
        )
        if not docs:
            return GenerationResult(
                answer=no_context_message,
                input_tokens=0,
                output_tokens=0,
                cost_usd=0.0,
                context_truncated=True,
                latency_ms=0,
            )

    lc_docs = _orch_docs_to_langchain(docs) if docs else []
    chain = _build_chain(model, prompt_template)
    usage_handler = _TokenUsageCallbackHandler()
    start = time.perf_counter()
    try:
        result = chain.invoke(
            {"context": lc_docs, "input": query},
            config={"callbacks": [usage_handler]},
        )
    finally:
        latency_ms = int((time.perf_counter() - start) * 1000)

    answer = result if isinstance(result, str) else str(result)
    input_tokens = usage_handler.input_tokens
    output_tokens = usage_handler.output_tokens
    if input_tokens == 0 and output_tokens == 0:
        input_tokens = sum(len(d.content) for d in docs) // CHARS_PER_TOKEN_ESTIMATE
        input_tokens += len(query) // CHARS_PER_TOKEN_ESTIMATE
        output_tokens = len(answer) // CHARS_PER_TOKEN_ESTIMATE

    cost_usd = 0.0
    if pricing_provider:
        try:
            pricing = pricing_provider.get_model_pricing(model)
            cost_usd = pricing.calculate_cost(
                input_tokens=input_tokens, output_tokens=output_tokens
            )
        except Exception as e:
            logger.warning("Generation cost calculation failed: %s", e)
    else:
        cost_usd = _cost_from_tokens(model, input_tokens, output_tokens)

    return GenerationResult(
        answer=answer,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
        context_truncated=context_truncated,
        latency_ms=latency_ms,
    )
