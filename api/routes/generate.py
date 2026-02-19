"""
RAG-Advanced Generate API Route.

POST /generate: run retrieval (single strategy) then LangChain-based generation to return
an answer string plus documents, tokens, and cost. Handles empty retrieval and separates
retrieval vs generation errors with clear user-facing messages.

Usage:
    POST /generate with body: { "query": "...", "strategy": "standard", "limit": 5, ... }
    Response: { "answer": "...", "documents": [...], "model": "...", "input_tokens": ..., ... }
"""

from __future__ import annotations

import logging
import os
from typing import Any

from fastapi import status
from pydantic import BaseModel, ConfigDict, Field

from api.routes.strategies import (
    DocumentResponse,
    _document_to_response,
    execute_strategy_endpoint,
)
from api.routes.strategies import ExecuteRequest as StrategyExecuteRequest
from generation import GenerationResult, generate_answer
from orchestration.errors import StrategyExecutionError
from orchestration.models import Document

logger = logging.getLogger(__name__)

# User-facing messages for retrieval vs generation failures (ApX-style)
RETRIEVAL_ERROR_MESSAGE = "An error occurred during retrieval."
GENERATION_ERROR_MESSAGE = "An error occurred while generating the response."
NO_CONTEXT_MESSAGE = "I couldn't find relevant information to answer your query."


# =============================================================================
# Request / Response Models
# =============================================================================


class GenerateRequest(BaseModel):
    """
    Request for RAG generate (retrieval + generation).

    Attributes:
        query: User question (required).
        strategy: Retrieval strategy name (default standard).
        limit: Max documents to retrieve.
        model: LLM model for generation (optional override).
        prompt_template: Prompt template with {context} and {input} (optional override).
        no_context_fallback: If true and no documents, call LLM with no context; else return fixed message.
        timeout_seconds: Timeout for retrieval step.
    """

    model_config = ConfigDict(
        frozen=True,
        json_schema_extra={
            "example": {
                "query": "What is retrieval-augmented generation?",
                "strategy": "standard",
                "limit": 5,
                "model": "gpt-4o-mini",
                "no_context_fallback": False,
            }
        },
    )

    query: str = Field(..., min_length=1, description="Search query")
    strategy: str = Field(default="standard", description="Retrieval strategy name")
    limit: int = Field(default=5, ge=1, le=50, description="Max documents to retrieve")
    model: str | None = Field(default=None, description="Generation model override")
    prompt_template: str | None = Field(default=None, description="Prompt template override")
    no_context_fallback: bool = Field(
        default=False,
        description="If true, call LLM with no context when retrieval returns no documents",
    )
    timeout_seconds: float = Field(default=30.0, ge=1.0, description="Retrieval timeout")
    initial_k: int | None = Field(default=None, ge=1, le=100, description="Reranking: candidate count")
    final_k: int | None = Field(default=None, ge=1, le=50, description="Reranking: results after rerank")


class GenerateResponse(BaseModel):
    """
    Response from POST /generate.

    Attributes:
        answer: Generated answer text.
        documents: Retrieved documents (sources).
        model: Model used for generation.
        input_tokens: LLM input tokens.
        output_tokens: LLM output tokens.
        cost_usd: Total cost (retrieval + generation).
        retrieval_latency_ms: Retrieval step latency.
        generation_latency_ms: Generation step latency.
        context_truncated: True if context was truncated to fit model window.
    """

    model_config = ConfigDict(frozen=True)

    answer: str = Field(..., description="Generated answer")
    documents: list[DocumentResponse] = Field(..., description="Retrieved documents (sources)")
    model: str = Field(..., description="Model used for generation")
    input_tokens: int = Field(..., description="LLM input tokens")
    output_tokens: int = Field(..., description="LLM output tokens")
    cost_usd: float = Field(..., description="Total cost in USD")
    retrieval_latency_ms: int = Field(..., description="Retrieval latency in ms")
    generation_latency_ms: int = Field(..., description="Generation latency in ms")
    context_truncated: bool = Field(default=False, description="Whether context was truncated")


# =============================================================================
# Endpoint
# =============================================================================


async def generate_endpoint(
    request: GenerateRequest,
    pricing_provider: Any = None,
) -> tuple[GenerateResponse | None, str | None, int]:
    """
    Run retrieval then generation; return response or (None, error_message, status_code).

    Caller (e.g. main.py) should return JSONResponse with error_message and status_code
    when first element is None.
    """
    # 1) Retrieval
    try:
        exec_req = StrategyExecuteRequest(
            strategy=request.strategy,
            query=request.query,
            limit=request.limit,
            timeout_seconds=request.timeout_seconds,
            initial_k=request.initial_k,
            final_k=request.final_k,
        )
        exec_response = await execute_strategy_endpoint(exec_req)
    except StrategyExecutionError as e:
        logger.warning("Generate: retrieval failed: %s", e)
        return None, RETRIEVAL_ERROR_MESSAGE, status.HTTP_503_SERVICE_UNAVAILABLE
    except Exception as e:
        logger.exception("Generate: retrieval error: %s", e)
        return None, RETRIEVAL_ERROR_MESSAGE, status.HTTP_502_BAD_GATEWAY

    retrieval_latency_ms = exec_response.latency_ms
    documents = [
        Document(
            id=d.id,
            content=d.content,
            title=d.title or "",
            source=d.source or "",
            similarity=d.similarity or 0.0,
            metadata=d.metadata,
        )
        for d in exec_response.documents
    ]

    # 2) Empty retrieval: fixed message or LLM with no context
    no_context_message = NO_CONTEXT_MESSAGE
    if not documents:
        if request.no_context_fallback:
            # Call LLM with empty context (fallback to standard LLM behavior)
            gen_result = generate_answer(
                request.query,
                [],
                model=request.model or None,
                prompt_template=request.prompt_template or None,
                no_context_message=no_context_message,
                empty_context_fallback=True,
                pricing_provider=pricing_provider,
            )
            return (
                GenerateResponse(
                    answer=gen_result.answer,
                    documents=[],
                    model=request.model or os.getenv("GENERATION_MODEL", "gpt-4o-mini"),
                    input_tokens=gen_result.input_tokens,
                    output_tokens=gen_result.output_tokens,
                    cost_usd=exec_response.cost_usd + gen_result.cost_usd,
                    retrieval_latency_ms=retrieval_latency_ms,
                    generation_latency_ms=gen_result.latency_ms,
                    context_truncated=gen_result.context_truncated,
                ),
                None,
                status.HTTP_200_OK,
            )
        # Return fixed message without calling LLM
        return (
            GenerateResponse(
                answer=no_context_message,
                documents=[],
                model=request.model or "gpt-4o-mini",
                input_tokens=0,
                output_tokens=0,
                cost_usd=exec_response.cost_usd,
                retrieval_latency_ms=retrieval_latency_ms,
                generation_latency_ms=0,
                context_truncated=False,
            ),
            None,
            status.HTTP_200_OK,
        )

    # 3) Generation
    try:
        gen_result = generate_answer(
            request.query,
            documents,
            model=request.model or None,
            prompt_template=request.prompt_template or None,
            no_context_message=no_context_message,
            pricing_provider=pricing_provider,
        )
    except Exception as e:
        logger.exception("Generate: generation error: %s", e)
        return None, GENERATION_ERROR_MESSAGE, status.HTTP_502_BAD_GATEWAY

    model_used = request.model or os.getenv("GENERATION_MODEL", "gpt-4o-mini")

    return (
        GenerateResponse(
            answer=gen_result.answer,
            documents=[_document_to_response(d) for d in documents],
            model=model_used,
            input_tokens=gen_result.input_tokens,
            output_tokens=gen_result.output_tokens,
            cost_usd=exec_response.cost_usd + gen_result.cost_usd,
            retrieval_latency_ms=retrieval_latency_ms,
            generation_latency_ms=gen_result.latency_ms,
            context_truncated=gen_result.context_truncated,
        ),
        None,
        status.HTTP_200_OK,
    )
