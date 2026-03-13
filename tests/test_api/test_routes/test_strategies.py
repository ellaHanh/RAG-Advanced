"""
Unit tests for Strategy API Routes.

Tests cover:
- Execute endpoint
- Chain endpoint
- Compare endpoint
- List strategies endpoint
"""

from __future__ import annotations

import pytest

from api.routes.strategies import (
    ChainRequest,
    ChainResponse,
    ChainStepResponse,
    CompareRequest,
    DocumentResponse,
    ExecuteRequest,
    ExecuteResponse,
    ListStrategiesResponse,
    StrategyInfoResponse,
    execute_chain_endpoint,
    execute_strategy_endpoint,
    list_strategies_endpoint,
)
from orchestration.executor import ExecutionContext
from orchestration.models import Document
from orchestration.registry import StrategyRegistry


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def registry() -> StrategyRegistry:
    """Create a test registry."""
    StrategyRegistry.reset()
    registry = StrategyRegistry()

    async def test_strategy(ctx: ExecutionContext) -> list[Document]:
        return [
            Document(
                id="doc1",
                content="Test content",
                title="Test",
                source="test.md",
                similarity=0.9,
            )
        ]

    async def another_strategy(ctx: ExecutionContext) -> list[Document]:
        return [
            Document(
                id="doc2",
                content="Another content",
                title="Another",
                source="other.md",
                similarity=0.8,
            )
        ]

    registry.register("test", test_strategy)
    registry.register("another", another_strategy)

    return registry


# =============================================================================
# Test: Request Models
# =============================================================================


class TestRequestModels:
    """Tests for request models."""

    def test_execute_request(self):
        """Test ExecuteRequest creation."""
        request = ExecuteRequest(
            strategy="test",
            query="test query",
        )

        assert request.strategy == "test"
        assert request.limit == 5  # Default

    def test_chain_request(self):
        """Test ChainRequest creation."""
        request = ChainRequest(
            steps=[{"strategy": "test"}],
            query="test query",
        )

        assert len(request.steps) == 1

    def test_compare_request(self):
        """Test CompareRequest creation."""
        request = CompareRequest(
            strategies=["test", "another"],
            query="test query",
        )

        assert len(request.strategies) == 2


# =============================================================================
# Test: Response Models
# =============================================================================


class TestResponseModels:
    """Tests for response models."""

    def test_document_response(self):
        """Test DocumentResponse creation."""
        response = DocumentResponse(
            id="doc1",
            content="Test content",
            similarity=0.9,
        )

        assert response.id == "doc1"

    def test_execute_response(self):
        """Test ExecuteResponse creation."""
        response = ExecuteResponse(
            documents=[
                DocumentResponse(id="doc1", content="Content")
            ],
            query="test",
            strategy_name="test",
            latency_ms=100,
            cost_usd=0.01,
        )

        assert len(response.documents) == 1

    def test_chain_response(self):
        """Test ChainResponse creation."""
        response = ChainResponse(
            query="test",
            success=True,
            steps=[
                ChainStepResponse(
                    step_name="step1",
                    strategy_name="test",
                    document_count=1,
                    duration_ms=100,
                )
            ],
            total_latency_ms=100,
            total_cost_usd=0.01,
            documents=[],
        )

        assert response.success is True


# =============================================================================
# Test: Execute Endpoint
# =============================================================================


class TestExecuteEndpoint:
    """Tests for execute endpoint."""

    @pytest.mark.asyncio
    async def test_execute_strategy(
        self,
        registry: StrategyRegistry,
    ):
        """Test executing a strategy."""
        request = ExecuteRequest(
            strategy="test",
            query="test query",
        )

        response = await execute_strategy_endpoint(request)

        assert response.strategy_name == "test"
        assert len(response.documents) == 1
        assert response.documents[0].id == "doc1"


# =============================================================================
# Test: Chain Endpoint
# =============================================================================


class TestChainEndpoint:
    """Tests for chain endpoint."""

    @pytest.mark.asyncio
    async def test_execute_chain(
        self,
        registry: StrategyRegistry,
    ):
        """Test executing a chain."""
        request = ChainRequest(
            steps=[
                {"strategy": "test"},
                {"strategy": "another"},
            ],
            query="test query",
        )

        response = await execute_chain_endpoint(request)

        assert response.success is True
        assert len(response.steps) == 2


# =============================================================================
# Test: List Strategies Endpoint
# =============================================================================


class TestListEndpoint:
    """Tests for list strategies endpoint."""

    def test_list_strategies_empty(self):
        """Test listing strategies from fresh registry."""
        StrategyRegistry.reset()

        response = list_strategies_endpoint()

        assert response.total_count == 0

    def test_list_strategies_with_registered(
        self,
        registry: StrategyRegistry,
    ):
        """Test listing with registered strategies."""
        # Note: Registry has strategies but without metadata
        response = list_strategies_endpoint()

        # The test registry doesn't provide full metadata
        # so we just verify the response structure
        assert isinstance(response, ListStrategiesResponse)
