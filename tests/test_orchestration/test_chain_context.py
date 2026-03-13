"""
Unit tests for ChainContext immutability and functional updates.

Tests cover:
- Immutability (frozen dataclass)
- Factory method (create)
- Functional updates (with_* methods)
- Accumulated metrics
- Error logging
- Metadata management
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import MappingProxyType

import pytest

from orchestration.models import (
    ChainContext,
    Document,
    ExecutionResult,
    TokenCounts,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_context() -> ChainContext:
    """Create a sample ChainContext for testing."""
    return ChainContext.create(
        query="What is machine learning?",
        metadata={"source": "test", "user_id": "user123"},
    )


@pytest.fixture
def sample_result() -> ExecutionResult:
    """Create a sample ExecutionResult for testing."""
    return ExecutionResult(
        documents=[
            Document(
                id="doc_1",
                content="Machine learning is a subset of AI.",
                title="ML Intro",
                source="test.md",
                similarity=0.95,
            ),
            Document(
                id="doc_2",
                content="Deep learning uses neural networks.",
                title="DL Overview",
                source="test2.md",
                similarity=0.85,
            ),
        ],
        query="What is machine learning?",
        strategy_name="standard",
        latency_ms=150,
        cost_usd=0.001,
        token_counts=TokenCounts(
            embedding_tokens=10,
            llm_input_tokens=50,
            llm_output_tokens=20,
        ),
    )


# =============================================================================
# Test: Creation
# =============================================================================


class TestChainContextCreation:
    """Tests for ChainContext creation."""

    def test_create_basic(self):
        """Test basic context creation."""
        ctx = ChainContext.create("test query")

        assert ctx.query == "test query"
        assert ctx.original_query == "test query"
        assert ctx.step_index == 0
        assert ctx.total_cost == 0.0
        assert ctx.total_latency_ms == 0
        assert len(ctx.intermediate_results) == 0
        assert len(ctx.error_log) == 0
        assert ctx.input_documents == ()

    def test_create_with_metadata(self):
        """Test context creation with metadata."""
        ctx = ChainContext.create(
            "test query",
            metadata={"key": "value", "number": 42},
        )

        assert ctx.metadata["key"] == "value"
        assert ctx.metadata["number"] == 42

    def test_create_metadata_is_immutable(self):
        """Test that created metadata is immutable."""
        ctx = ChainContext.create("test query", metadata={"key": "value"})

        assert isinstance(ctx.metadata, MappingProxyType)
        with pytest.raises(TypeError):
            ctx.metadata["new_key"] = "new_value"  # type: ignore

    def test_direct_creation(self):
        """Test direct dataclass creation."""
        ctx = ChainContext(
            query="current query",
            original_query="original query",
            total_cost=0.5,
            total_latency_ms=200,
            step_index=2,
        )

        assert ctx.query == "current query"
        assert ctx.original_query == "original query"
        assert ctx.total_cost == 0.5
        assert ctx.total_latency_ms == 200
        assert ctx.step_index == 2


# =============================================================================
# Test: Immutability
# =============================================================================


class TestChainContextImmutability:
    """Tests for ChainContext immutability."""

    def test_cannot_modify_query(self, sample_context: ChainContext):
        """Test that query cannot be modified."""
        with pytest.raises(FrozenInstanceError):
            sample_context.query = "new query"  # type: ignore

    def test_cannot_modify_original_query(self, sample_context: ChainContext):
        """Test that original_query cannot be modified."""
        with pytest.raises(FrozenInstanceError):
            sample_context.original_query = "new query"  # type: ignore

    def test_cannot_modify_step_index(self, sample_context: ChainContext):
        """Test that step_index cannot be modified."""
        with pytest.raises(FrozenInstanceError):
            sample_context.step_index = 5  # type: ignore

    def test_cannot_modify_total_cost(self, sample_context: ChainContext):
        """Test that total_cost cannot be modified."""
        with pytest.raises(FrozenInstanceError):
            sample_context.total_cost = 1.0  # type: ignore

    def test_cannot_modify_total_latency(self, sample_context: ChainContext):
        """Test that total_latency_ms cannot be modified."""
        with pytest.raises(FrozenInstanceError):
            sample_context.total_latency_ms = 500  # type: ignore

    def test_intermediate_results_is_tuple(self, sample_context: ChainContext):
        """Test that intermediate_results is immutable tuple."""
        assert isinstance(sample_context.intermediate_results, tuple)

    def test_error_log_is_tuple(self, sample_context: ChainContext):
        """Test that error_log is immutable tuple."""
        assert isinstance(sample_context.error_log, tuple)

    def test_metadata_is_mapping_proxy(self, sample_context: ChainContext):
        """Test that metadata is immutable MappingProxyType."""
        assert isinstance(sample_context.metadata, MappingProxyType)


# =============================================================================
# Test: with_step_result
# =============================================================================


class TestWithStepResult:
    """Tests for with_step_result method."""

    def test_adds_result_to_intermediate(
        self,
        sample_context: ChainContext,
        sample_result: ExecutionResult,
    ):
        """Test that step result is added to intermediate_results."""
        new_ctx = sample_context.with_step_result(sample_result)

        assert len(new_ctx.intermediate_results) == 1
        step_data = new_ctx.intermediate_results[0]
        assert step_data["strategy_name"] == "standard"
        assert step_data["document_count"] == 2

    def test_accumulates_cost(
        self,
        sample_context: ChainContext,
        sample_result: ExecutionResult,
    ):
        """Test that cost is accumulated."""
        new_ctx = sample_context.with_step_result(sample_result)

        assert new_ctx.total_cost == sample_result.cost_usd

    def test_accumulates_latency(
        self,
        sample_context: ChainContext,
        sample_result: ExecutionResult,
    ):
        """Test that latency is accumulated."""
        new_ctx = sample_context.with_step_result(sample_result)

        assert new_ctx.total_latency_ms == sample_result.latency_ms

    def test_increments_step_index(
        self,
        sample_context: ChainContext,
        sample_result: ExecutionResult,
    ):
        """Test that step_index is incremented."""
        new_ctx = sample_context.with_step_result(sample_result)

        assert new_ctx.step_index == 1

    def test_original_unchanged(
        self,
        sample_context: ChainContext,
        sample_result: ExecutionResult,
    ):
        """Test that original context is unchanged."""
        original_step_index = sample_context.step_index
        original_cost = sample_context.total_cost

        sample_context.with_step_result(sample_result)

        assert sample_context.step_index == original_step_index
        assert sample_context.total_cost == original_cost
        assert len(sample_context.intermediate_results) == 0

    def test_multiple_steps(
        self,
        sample_context: ChainContext,
        sample_result: ExecutionResult,
    ):
        """Test adding multiple step results."""
        ctx1 = sample_context.with_step_result(sample_result)

        result2 = ExecutionResult(
            documents=[],
            query="test",
            strategy_name="reranking",
            latency_ms=200,
            cost_usd=0.002,
        )
        ctx2 = ctx1.with_step_result(result2)

        assert len(ctx2.intermediate_results) == 2
        assert ctx2.step_index == 2
        assert ctx2.total_cost == pytest.approx(0.003)
        assert ctx2.total_latency_ms == 350

    def test_preserves_query(
        self,
        sample_context: ChainContext,
        sample_result: ExecutionResult,
    ):
        """Test that query is preserved."""
        new_ctx = sample_context.with_step_result(sample_result)

        assert new_ctx.query == sample_context.query
        assert new_ctx.original_query == sample_context.original_query

    def test_sets_input_documents_for_next_step(
        self,
        sample_context: ChainContext,
        sample_result: ExecutionResult,
    ):
        """Test that with_step_result sets input_documents to the step's documents."""
        new_ctx = sample_context.with_step_result(sample_result)

        assert len(new_ctx.input_documents) == 2
        assert new_ctx.input_documents[0].id == sample_result.documents[0].id
        assert new_ctx.input_documents[1].content == sample_result.documents[1].content

    def test_preserves_metadata(
        self,
        sample_context: ChainContext,
        sample_result: ExecutionResult,
    ):
        """Test that metadata is preserved."""
        new_ctx = sample_context.with_step_result(sample_result)

        assert new_ctx.metadata["source"] == "test"
        assert new_ctx.metadata["user_id"] == "user123"


# =============================================================================
# Test: with_query
# =============================================================================


class TestWithQuery:
    """Tests for with_query method."""

    def test_updates_query(self, sample_context: ChainContext):
        """Test that query is updated."""
        new_ctx = sample_context.with_query("new query")

        assert new_ctx.query == "new query"

    def test_preserves_original_query(self, sample_context: ChainContext):
        """Test that original_query is preserved."""
        new_ctx = sample_context.with_query("new query")

        assert new_ctx.original_query == sample_context.original_query

    def test_original_unchanged(self, sample_context: ChainContext):
        """Test that original context is unchanged."""
        original_query = sample_context.query

        sample_context.with_query("new query")

        assert sample_context.query == original_query

    def test_preserves_all_other_fields(
        self,
        sample_context: ChainContext,
        sample_result: ExecutionResult,
    ):
        """Test that all other fields are preserved."""
        ctx_with_result = sample_context.with_step_result(sample_result)
        ctx_with_error = ctx_with_result.with_error("test error")
        new_ctx = ctx_with_error.with_query("new query")

        assert new_ctx.step_index == ctx_with_error.step_index
        assert new_ctx.total_cost == ctx_with_error.total_cost
        assert new_ctx.total_latency_ms == ctx_with_error.total_latency_ms
        assert len(new_ctx.intermediate_results) == len(ctx_with_error.intermediate_results)
        assert len(new_ctx.error_log) == 1


# =============================================================================
# Test: with_error
# =============================================================================


class TestWithError:
    """Tests for with_error method."""

    def test_adds_error(self, sample_context: ChainContext):
        """Test that error is added to log."""
        new_ctx = sample_context.with_error("Something went wrong")

        assert len(new_ctx.error_log) == 1
        assert new_ctx.error_log[0] == "Something went wrong"

    def test_multiple_errors(self, sample_context: ChainContext):
        """Test adding multiple errors."""
        ctx1 = sample_context.with_error("Error 1")
        ctx2 = ctx1.with_error("Error 2")
        ctx3 = ctx2.with_error("Error 3")

        assert len(ctx3.error_log) == 3
        assert ctx3.error_log == ("Error 1", "Error 2", "Error 3")

    def test_original_unchanged(self, sample_context: ChainContext):
        """Test that original context is unchanged."""
        sample_context.with_error("test error")

        assert len(sample_context.error_log) == 0

    def test_preserves_all_other_fields(self, sample_context: ChainContext):
        """Test that all other fields are preserved."""
        new_ctx = sample_context.with_error("test error")

        assert new_ctx.query == sample_context.query
        assert new_ctx.original_query == sample_context.original_query
        assert new_ctx.step_index == sample_context.step_index
        assert new_ctx.total_cost == sample_context.total_cost


# =============================================================================
# Test: with_metadata
# =============================================================================


class TestWithMetadata:
    """Tests for with_metadata method."""

    def test_adds_metadata(self, sample_context: ChainContext):
        """Test that metadata is added."""
        new_ctx = sample_context.with_metadata("new_key", "new_value")

        assert new_ctx.metadata["new_key"] == "new_value"

    def test_preserves_existing_metadata(self, sample_context: ChainContext):
        """Test that existing metadata is preserved."""
        new_ctx = sample_context.with_metadata("new_key", "new_value")

        assert new_ctx.metadata["source"] == "test"
        assert new_ctx.metadata["user_id"] == "user123"

    def test_original_unchanged(self, sample_context: ChainContext):
        """Test that original context is unchanged."""
        sample_context.with_metadata("new_key", "new_value")

        assert "new_key" not in sample_context.metadata

    def test_overwrites_existing_key(self, sample_context: ChainContext):
        """Test that existing key can be overwritten."""
        new_ctx = sample_context.with_metadata("source", "updated")

        assert new_ctx.metadata["source"] == "updated"
        assert sample_context.metadata["source"] == "test"

    def test_metadata_is_immutable(self, sample_context: ChainContext):
        """Test that new metadata is still immutable."""
        new_ctx = sample_context.with_metadata("key", "value")

        assert isinstance(new_ctx.metadata, MappingProxyType)
        with pytest.raises(TypeError):
            new_ctx.metadata["another"] = "test"  # type: ignore


# =============================================================================
# Test: Chaining Operations
# =============================================================================


class TestChainingOperations:
    """Tests for chaining multiple operations."""

    def test_chain_multiple_operations(
        self,
        sample_context: ChainContext,
        sample_result: ExecutionResult,
    ):
        """Test chaining multiple with_* operations."""
        final_ctx = (
            sample_context
            .with_metadata("phase", "started")
            .with_step_result(sample_result)
            .with_query("refined query")
            .with_metadata("phase", "refined")
            .with_error("minor warning")
        )

        assert final_ctx.query == "refined query"
        assert final_ctx.original_query == "What is machine learning?"
        assert final_ctx.metadata["phase"] == "refined"
        assert len(final_ctx.intermediate_results) == 1
        assert len(final_ctx.error_log) == 1
        assert final_ctx.step_index == 1

    def test_original_context_completely_unchanged(
        self,
        sample_context: ChainContext,
        sample_result: ExecutionResult,
    ):
        """Test that original context is completely unchanged after chain."""
        original_query = sample_context.query
        original_metadata = dict(sample_context.metadata)
        original_step = sample_context.step_index

        # Perform many operations
        (
            sample_context
            .with_metadata("a", 1)
            .with_step_result(sample_result)
            .with_query("changed")
            .with_error("error")
            .with_metadata("b", 2)
        )

        # Original should be unchanged
        assert sample_context.query == original_query
        assert dict(sample_context.metadata) == original_metadata
        assert sample_context.step_index == original_step
        assert len(sample_context.intermediate_results) == 0
        assert len(sample_context.error_log) == 0


# =============================================================================
# Test: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_query(self):
        """Test context with empty query."""
        ctx = ChainContext.create("")

        assert ctx.query == ""
        assert ctx.original_query == ""

    def test_very_long_error_log(self, sample_context: ChainContext):
        """Test accumulating many errors."""
        ctx = sample_context
        for i in range(100):
            ctx = ctx.with_error(f"Error {i}")

        assert len(ctx.error_log) == 100
        assert ctx.error_log[0] == "Error 0"
        assert ctx.error_log[99] == "Error 99"

    def test_many_steps(self, sample_context: ChainContext, sample_result: ExecutionResult):
        """Test accumulating many step results."""
        ctx = sample_context
        for _ in range(50):
            ctx = ctx.with_step_result(sample_result)

        assert ctx.step_index == 50
        assert len(ctx.intermediate_results) == 50
        assert ctx.total_cost == pytest.approx(0.001 * 50)
        assert ctx.total_latency_ms == 150 * 50

    def test_metadata_with_complex_values(self, sample_context: ChainContext):
        """Test metadata with complex nested values."""
        ctx = sample_context.with_metadata("nested", {"a": [1, 2, 3], "b": {"c": "d"}})

        assert ctx.metadata["nested"]["a"] == [1, 2, 3]
        assert ctx.metadata["nested"]["b"]["c"] == "d"
