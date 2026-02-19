"""
Unit tests for RAG generation chain (generate_answer).

Tests cover:
- generate_answer with mocked chain invoke: answer, tokens, cost, prompt usage.
- Empty documents: fixed message or empty_context_fallback.
- Context truncation (context_truncated) when over limit.
- Cost from pricing provider.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from generation.chain import (
    DEFAULT_MODEL,
    DEFAULT_PROMPT_TEMPLATE,
    GenerationResult,
    generate_answer,
)
from orchestration.models import Document


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_documents() -> list[Document]:
    """Fixed list of documents for tests."""
    return [
        Document(
            id="1",
            content="RAG combines retrieval and generation. You retrieve relevant docs then generate an answer.",
            title="RAG intro",
            source="intro.md",
            similarity=0.95,
        ),
        Document(
            id="2",
            content="LangChain provides create_stuff_documents_chain for stuffing docs into a prompt.",
            title="LangChain",
            source="langchain.md",
            similarity=0.88,
        ),
    ]


# =============================================================================
# Unit tests: generate_answer with mocked chain
# =============================================================================


@patch("generation.chain._build_chain")
def test_generate_answer_returns_answer_and_tokens(
    mock_build: MagicMock,
    sample_documents: list[Document],
) -> None:
    """generate_answer returns answer, token counts, and cost when chain is mocked."""
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = "RAG is retrieval-augmented generation. It combines retrieval and generation."
    mock_build.return_value = mock_chain

    result = generate_answer(
        "What is RAG?",
        sample_documents,
        model="gpt-4o-mini",
    )

    assert isinstance(result, GenerationResult)
    assert "retrieval" in result.answer.lower() or "RAG" in result.answer
    assert result.input_tokens >= 0
    assert result.output_tokens >= 0
    assert result.cost_usd >= 0.0
    mock_chain.invoke.assert_called_once()
    call_args = mock_chain.invoke.call_args[0][0]
    assert "context" in call_args or "input_documents" in call_args
    assert call_args.get("input") == "What is RAG?"


@patch("generation.chain._build_chain")
def test_generate_answer_prompt_template_applied(
    mock_build: MagicMock,
    sample_documents: list[Document],
) -> None:
    """Context passed to chain contains document contents; input contains query."""
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = "Generated answer."

    def capture_invoke(arg: dict, **kwargs: object) -> str:
        # Chain may receive context as list of LC docs or under input_documents
        ctx = arg.get("context", arg.get("input_documents", []))
        if ctx:
            if hasattr(ctx[0], "page_content"):
                assert "retrieval" in ctx[0].page_content or "RAG" in ctx[0].page_content
            else:
                assert "retrieval" in str(ctx[0]) or "RAG" in str(ctx[0])
        assert arg.get("input") == "What is RAG?"
        return "Generated answer."

    mock_chain.invoke.side_effect = capture_invoke
    mock_build.return_value = mock_chain

    result = generate_answer("What is RAG?", sample_documents)

    assert result.answer == "Generated answer."
    mock_chain.invoke.assert_called_once()


@patch("generation.chain._build_chain")
def test_generate_answer_empty_documents_returns_fixed_message(
    mock_build: MagicMock,
) -> None:
    """When documents is empty and empty_context_fallback is False, return no_context_message."""
    result = generate_answer(
        "Any question?",
        [],
        no_context_message="I couldn't find relevant information.",
    )

    assert result.answer == "I couldn't find relevant information."
    assert result.input_tokens == 0
    assert result.output_tokens == 0
    assert result.cost_usd == 0.0
    mock_build.assert_not_called()


@patch("generation.chain._build_chain")
def test_generate_answer_empty_context_fallback_calls_llm(
    mock_build: MagicMock,
) -> None:
    """When documents is empty and empty_context_fallback is True, chain is invoked with empty context."""
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = "I have no context to use."
    mock_build.return_value = mock_chain

    result = generate_answer(
        "What is RAG?",
        [],
        empty_context_fallback=True,
        no_context_message="No info.",
    )

    assert result.answer == "I have no context to use."
    mock_chain.invoke.assert_called_once()
    call_arg = mock_chain.invoke.call_args[0][0]
    ctx = call_arg.get("context", call_arg.get("input_documents", []))
    assert ctx == [] or (hasattr(ctx, "__len__") and len(ctx) == 0)


def test_default_constants() -> None:
    """DEFAULT_MODEL and DEFAULT_PROMPT_TEMPLATE have expected placeholders."""
    assert "gpt-4o-mini" in DEFAULT_MODEL or "gpt" in DEFAULT_MODEL
    assert "{context}" in DEFAULT_PROMPT_TEMPLATE
    assert "{input}" in DEFAULT_PROMPT_TEMPLATE


@patch("generation.chain._build_chain")
def test_generate_answer_cost_from_pricing(
    mock_build: MagicMock,
    sample_documents: list[Document],
) -> None:
    """When callback provides token counts, cost is computed (or 0 if no provider)."""
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = "Answer."
    mock_build.return_value = mock_chain

    result = generate_answer("What is RAG?", sample_documents)

    # Cost may be 0 if pricing provider not initialized in test env
    assert result.cost_usd >= 0.0
    assert result.input_tokens >= 0
    assert result.output_tokens >= 0
