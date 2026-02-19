"""
RAG-Advanced Pydantic Models.

Core data models for strategy orchestration, execution, and evaluation.

This module provides:
- StrategyConfig: Configuration for strategy execution
- StrategyMetadata: Metadata about a registered strategy
- ExecutionResult: Results from strategy execution
- ChainContext: Immutable state passed between chain steps
- TokenCounts: Token usage tracking
- Document: Retrieved document representation

Usage:
    from orchestration.models import StrategyConfig, ExecutionResult

    config = StrategyConfig(limit=5, initial_k=20)
    result = await execute_strategy("reranking", query, config)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from types import MappingProxyType
from typing import Any, Callable, Coroutine, TypeVar

from pydantic import BaseModel, ConfigDict, Field


# =============================================================================
# Type Aliases
# =============================================================================

# Strategy function type: async function that takes query and config, returns result
StrategyFunc = Callable[..., Coroutine[Any, Any, "ExecutionResult"]]

T = TypeVar("T")


# =============================================================================
# Enums
# =============================================================================


class StrategyType(str, Enum):
    """Types of RAG strategies available."""

    STANDARD = "standard"
    RERANKING = "reranking"
    MULTI_QUERY = "multi_query"
    QUERY_EXPANSION = "query_expansion"
    SELF_REFLECTIVE = "self_reflective"
    AGENTIC = "agentic"
    CONTEXTUAL_RETRIEVAL = "contextual_retrieval"
    HIERARCHICAL = "hierarchical"
    LATE_CHUNKING = "late_chunking"
    KNOWLEDGE_GRAPH = "knowledge_graph"
    FINE_TUNED = "fine_tuned"


class ResourceType(str, Enum):
    """Types of resources a strategy may require."""

    DATABASE = "database"
    EMBEDDING_API = "embedding_api"
    LLM_API = "llm_api"
    RERANKER = "reranker"
    CACHE = "cache"


# =============================================================================
# Configuration Models
# =============================================================================


class StrategyConfig(BaseModel):
    """
    Configuration for strategy execution.

    This model contains all configurable parameters for RAG strategies.
    Not all parameters apply to all strategies - each strategy uses
    the relevant subset.

    Attributes:
        limit: Maximum number of final results to return.
        initial_k: Number of initial candidates for reranking strategies.
        final_k: Number of results after reranking.
        num_variations: Number of query variations for multi-query strategies.
        max_iterations: Maximum iterations for self-reflective strategies.
        relevance_threshold: Minimum relevance score (0-5) for self-reflective.
        timeout_seconds: Timeout for strategy execution.
        use_cache: Whether to use result caching.
        metadata: Additional strategy-specific configuration.
    """

    model_config = ConfigDict(
        frozen=False,
        extra="allow",
        validate_assignment=True,
    )

    limit: int = Field(default=5, ge=1, le=100, description="Maximum results to return")
    initial_k: int = Field(default=20, ge=1, le=100, description="Initial candidates for reranking")
    final_k: int = Field(default=5, ge=1, le=50, description="Results after reranking")
    num_variations: int = Field(default=3, ge=1, le=10, description="Query variations count")
    max_iterations: int = Field(default=2, ge=1, le=5, description="Max self-reflection iterations")
    relevance_threshold: float = Field(
        default=3.0, ge=0.0, le=5.0, description="Minimum relevance score"
    )
    timeout_seconds: float = Field(default=30.0, ge=1.0, le=300.0, description="Execution timeout")
    use_cache: bool = Field(default=True, description="Whether to use caching")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional config")


class StrategyMetadata(BaseModel):
    """
    Metadata about a registered strategy.

    This model describes a strategy's characteristics and requirements,
    used for documentation and intelligent strategy selection.

    Attributes:
        name: Unique identifier for the strategy.
        description: Human-readable description of what the strategy does.
        strategy_type: The type/category of the strategy.
        version: Version string for the strategy implementation.
        required_resources: List of resources the strategy needs.
        estimated_latency_ms: Typical latency range (min, max).
        estimated_cost_per_query: Typical cost in USD per query.
        precision_rating: Expected precision (1-5 scale).
        tags: Additional searchable tags.
    """

    model_config = ConfigDict(frozen=True)

    name: str = Field(..., description="Unique strategy identifier")
    description: str = Field(..., description="What the strategy does")
    strategy_type: StrategyType = Field(..., description="Strategy category")
    version: str = Field(default="1.0.0", description="Implementation version")
    required_resources: list[ResourceType] = Field(
        default_factory=list, description="Required resources"
    )
    estimated_latency_ms: tuple[int, int] = Field(
        default=(100, 500), description="Latency range (min, max)"
    )
    estimated_cost_per_query: float = Field(
        default=0.001, ge=0.0, description="Estimated cost in USD"
    )
    precision_rating: int = Field(default=3, ge=1, le=5, description="Precision 1-5")
    tags: list[str] = Field(default_factory=list, description="Searchable tags")


# =============================================================================
# Token and Cost Tracking
# =============================================================================


class TokenCounts(BaseModel):
    """
    Token usage tracking for cost calculation.

    Attributes:
        embedding_tokens: Tokens used for embedding generation.
        llm_input_tokens: Tokens sent to LLM.
        llm_output_tokens: Tokens received from LLM.
        total_tokens: Sum of all tokens.
    """

    model_config = ConfigDict(frozen=True)

    embedding_tokens: int = Field(default=0, ge=0, description="Embedding tokens used")
    llm_input_tokens: int = Field(default=0, ge=0, description="LLM input tokens")
    llm_output_tokens: int = Field(default=0, ge=0, description="LLM output tokens")

    @property
    def total_tokens(self) -> int:
        """Calculate total tokens used."""
        return self.embedding_tokens + self.llm_input_tokens + self.llm_output_tokens

    def __add__(self, other: "TokenCounts") -> "TokenCounts":
        """Add two TokenCounts together."""
        return TokenCounts(
            embedding_tokens=self.embedding_tokens + other.embedding_tokens,
            llm_input_tokens=self.llm_input_tokens + other.llm_input_tokens,
            llm_output_tokens=self.llm_output_tokens + other.llm_output_tokens,
        )


# =============================================================================
# Document Models
# =============================================================================


class Document(BaseModel):
    """
    Retrieved document representation.

    Attributes:
        id: Unique document identifier.
        content: Document text content.
        title: Document title.
        source: Source path or URL.
        similarity: Similarity score from vector search.
        metadata: Additional document metadata.
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(..., description="Document ID")
    content: str = Field(..., description="Document content")
    title: str = Field(default="", description="Document title")
    source: str = Field(default="", description="Source path/URL")
    similarity: float = Field(default=0.0, ge=0.0, le=1.0, description="Similarity score")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


# =============================================================================
# Execution Results
# =============================================================================


class ExecutionResult(BaseModel):
    """
    Result from strategy execution.

    Contains retrieved documents, performance metrics, and cost information.

    Attributes:
        documents: List of retrieved documents.
        query: The original query that was executed.
        strategy_name: Name of the strategy that was executed.
        latency_ms: Execution time in milliseconds.
        cost_usd: Estimated cost in USD.
        token_counts: Detailed token usage.
        timestamp: When the execution completed.
        cached: Whether result was from cache.
        metadata: Additional execution metadata.
    """

    model_config = ConfigDict(frozen=True)

    documents: list[Document] = Field(default_factory=list, description="Retrieved documents")
    query: str = Field(..., description="Original query")
    strategy_name: str = Field(..., description="Strategy that was executed")
    latency_ms: int = Field(default=0, ge=0, description="Execution time in ms")
    cost_usd: float = Field(default=0.0, ge=0.0, description="Estimated cost")
    token_counts: TokenCounts = Field(
        default_factory=TokenCounts, description="Token usage breakdown"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Execution timestamp"
    )
    cached: bool = Field(default=False, description="Was result cached")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    @property
    def document_ids(self) -> list[str]:
        """Get list of IDs from result documents.

        In RAG-Advanced retrieval, strategies return chunks; each Document.id
        is the chunk primary key (chunks.id). So document_ids are chunk IDs
        when used for evaluation. See docs/README_terminology.md.
        """
        return [doc.id for doc in self.documents]

    @property
    def document_count(self) -> int:
        """Get number of documents retrieved."""
        return len(self.documents)


# =============================================================================
# Chain Context (Immutable)
# =============================================================================


@dataclass(frozen=True)
class ChainContext:
    """
    Immutable state passed between strategy steps in a chain.

    This class uses a frozen dataclass to ensure immutability.
    All updates return new instances via with_* methods.

    Attributes:
        query: Current query (may be modified by strategies).
        original_query: The original unmodified query.
        input_documents: Documents from the previous step (for the next step to consume).
        intermediate_results: Results from each completed step.
        metadata: Additional context metadata.
        total_cost: Accumulated cost across all steps.
        total_latency_ms: Accumulated latency across all steps.
        error_log: Log of any errors encountered.
        step_index: Current step index (0-based).
    """

    query: str
    original_query: str
    input_documents: tuple[Document, ...] = field(default_factory=tuple)
    intermediate_results: tuple[MappingProxyType[str, Any], ...] = field(default_factory=tuple)
    metadata: MappingProxyType[str, Any] = field(
        default_factory=lambda: MappingProxyType({})
    )
    total_cost: float = 0.0
    total_latency_ms: int = 0
    error_log: tuple[str, ...] = field(default_factory=tuple)
    step_index: int = 0

    @classmethod
    def create(
        cls,
        query: str,
        metadata: dict[str, Any] | None = None,
    ) -> "ChainContext":
        """
        Create a new ChainContext.

        Args:
            query: The initial query.
            metadata: Optional metadata dictionary.

        Returns:
            A new ChainContext instance.
        """
        return cls(
            query=query,
            original_query=query,
            metadata=MappingProxyType(metadata or {}),
        )

    def with_step_result(
        self,
        result: ExecutionResult,
    ) -> "ChainContext":
        """
        Create a new context with a step result added.

        Args:
            result: The execution result from the completed step.

        Returns:
            A new ChainContext with the result appended.
        """
        step_data = MappingProxyType({
            "step_index": self.step_index,
            "strategy_name": result.strategy_name,
            "document_count": result.document_count,
            "latency_ms": result.latency_ms,
            "cost_usd": result.cost_usd,
            "document_ids": result.document_ids,
        })
        return ChainContext(
            query=self.query,
            original_query=self.original_query,
            input_documents=tuple(result.documents),
            intermediate_results=self.intermediate_results + (step_data,),
            metadata=self.metadata,
            total_cost=self.total_cost + result.cost_usd,
            total_latency_ms=self.total_latency_ms + result.latency_ms,
            error_log=self.error_log,
            step_index=self.step_index + 1,
        )

    def with_query(self, new_query: str) -> "ChainContext":
        """
        Create a new context with an updated query.

        Args:
            new_query: The new query string.

        Returns:
            A new ChainContext with the updated query.
        """
        return ChainContext(
            query=new_query,
            original_query=self.original_query,
            input_documents=self.input_documents,
            intermediate_results=self.intermediate_results,
            metadata=self.metadata,
            total_cost=self.total_cost,
            total_latency_ms=self.total_latency_ms,
            error_log=self.error_log,
            step_index=self.step_index,
        )

    def with_error(self, error_message: str) -> "ChainContext":
        """
        Create a new context with an error logged.

        Args:
            error_message: The error message to log.

        Returns:
            A new ChainContext with the error appended.
        """
        return ChainContext(
            query=self.query,
            original_query=self.original_query,
            input_documents=self.input_documents,
            intermediate_results=self.intermediate_results,
            metadata=self.metadata,
            total_cost=self.total_cost,
            total_latency_ms=self.total_latency_ms,
            error_log=self.error_log + (error_message,),
            step_index=self.step_index,
        )

    def with_metadata(self, key: str, value: Any) -> "ChainContext":
        """
        Create a new context with additional metadata.

        Args:
            key: Metadata key.
            value: Metadata value.

        Returns:
            A new ChainContext with the metadata added.
        """
        new_metadata = dict(self.metadata)
        new_metadata[key] = value
        return ChainContext(
            query=self.query,
            original_query=self.original_query,
            input_documents=self.input_documents,
            intermediate_results=self.intermediate_results,
            metadata=MappingProxyType(new_metadata),
            total_cost=self.total_cost,
            total_latency_ms=self.total_latency_ms,
            error_log=self.error_log,
            step_index=self.step_index,
        )


# =============================================================================
# Chain Step Configuration
# =============================================================================


class ChainStep(BaseModel):
    """
    Configuration for a single step in a strategy chain.

    Attributes:
        strategy: Name of the strategy to execute.
        config: Configuration for this step.
        fallback_strategy: Optional fallback if this step fails.
        continue_on_error: Whether to continue chain if this step fails.
    """

    model_config = ConfigDict(frozen=True)

    strategy: str = Field(..., description="Strategy name")
    config: StrategyConfig = Field(default_factory=StrategyConfig, description="Step config")
    fallback_strategy: str | None = Field(default=None, description="Fallback strategy name")
    continue_on_error: bool = Field(default=False, description="Continue on failure")


class ChainConfig(BaseModel):
    """
    Configuration for a strategy chain.

    Attributes:
        steps: List of chain steps to execute.
        timeout_seconds: Overall chain timeout.
        stop_on_empty_results: Stop chain if a step returns no results.
    """

    model_config = ConfigDict(frozen=True)

    steps: list[ChainStep] = Field(..., min_length=1, description="Chain steps")
    timeout_seconds: float = Field(default=60.0, ge=1.0, description="Chain timeout")
    stop_on_empty_results: bool = Field(default=False, description="Stop on empty results")
