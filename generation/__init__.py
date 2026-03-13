"""
RAG-Advanced Generation Module.

LangChain-based generation stage: turn (query + retrieved documents) into a natural-language
answer via a stuff-documents chain (prompt template + LLM).

Exports:
    - generate_answer: Generate answer from query and documents; returns GenerationResult.
    - GenerationResult: Dataclass with answer, input_tokens, output_tokens, cost_usd, etc.
    - DEFAULT_MODEL: Default LLM model name (gpt-4o-mini).
    - DEFAULT_PROMPT_TEMPLATE: Default RAG prompt with {context} and {input}.
"""

from generation.chain import (
    DEFAULT_MODEL,
    DEFAULT_PROMPT_TEMPLATE,
    GenerationResult,
    generate_answer,
)

__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_PROMPT_TEMPLATE",
    "GenerationResult",
    "generate_answer",
]
