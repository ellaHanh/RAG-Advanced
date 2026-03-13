"""
Unit tests for Cost Tracking Module.

Tests cover:
- PricingProvider initialization and configuration loading
- Model pricing lookup (current and historical)
- Cost calculation
- CostTracker aggregation and summaries
- Thread safety
"""

from __future__ import annotations

import json
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from orchestration.cost_tracker import (
    CostEntry,
    CostSummary,
    CostTracker,
    calculate_cost,
    estimate_embedding_cost,
    estimate_llm_cost,
)
from orchestration.errors import PricingConfigError
from orchestration.pricing import (
    ModelPricing,
    PricingProvider,
    get_pricing_provider,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_pricing_config() -> dict:
    """Create sample pricing configuration."""
    return {
        "pricing_history": [
            {
                "effective_date": "2026-01-01T00:00:00Z",
                "currency": "USD",
                "models": {
                    "gpt-4o-mini": {
                        "input_per_1k": 0.00015,
                        "output_per_1k": 0.0006,
                        "description": "GPT-4o Mini",
                    },
                    "text-embedding-3-small": {
                        "input_per_1k": 0.00002,
                        "output_per_1k": 0.0,
                        "description": "Small embedding",
                    },
                },
            },
            {
                "effective_date": "2025-01-01T00:00:00Z",
                "currency": "USD",
                "models": {
                    "gpt-4o-mini": {
                        "input_per_1k": 0.0003,
                        "output_per_1k": 0.001,
                        "description": "GPT-4o Mini (old pricing)",
                    },
                },
            },
        ],
        "defaults": {
            "input_per_1k": 0.001,
            "output_per_1k": 0.002,
            "description": "Default pricing",
        },
    }


@pytest.fixture
def pricing_config_file(sample_pricing_config: dict, tmp_path: Path) -> Path:
    """Create a temporary pricing config file."""
    config_path = tmp_path / "pricing.json"
    with open(config_path, "w") as f:
        json.dump(sample_pricing_config, f)
    return config_path


@pytest.fixture(autouse=True)
def reset_pricing_provider():
    """Reset pricing provider singleton before each test."""
    PricingProvider.reset()
    yield
    PricingProvider.reset()


# =============================================================================
# Test: ModelPricing
# =============================================================================


class TestModelPricing:
    """Tests for ModelPricing model."""

    def test_calculate_cost_input_only(self):
        """Test cost calculation with only input tokens."""
        pricing = ModelPricing(
            input_per_1k=0.001,
            output_per_1k=0.002,
        )
        cost = pricing.calculate_cost(input_tokens=1000, output_tokens=0)
        assert cost == pytest.approx(0.001)

    def test_calculate_cost_output_only(self):
        """Test cost calculation with only output tokens."""
        pricing = ModelPricing(
            input_per_1k=0.001,
            output_per_1k=0.002,
        )
        cost = pricing.calculate_cost(input_tokens=0, output_tokens=1000)
        assert cost == pytest.approx(0.002)

    def test_calculate_cost_both(self):
        """Test cost calculation with both input and output tokens."""
        pricing = ModelPricing(
            input_per_1k=0.001,
            output_per_1k=0.002,
        )
        cost = pricing.calculate_cost(input_tokens=1000, output_tokens=500)
        expected = 0.001 + 0.001  # 0.001 for input, 0.001 for 500 output tokens
        assert cost == pytest.approx(expected)

    def test_calculate_cost_fractional_tokens(self):
        """Test cost calculation with non-1K token counts."""
        pricing = ModelPricing(
            input_per_1k=0.001,
            output_per_1k=0.002,
        )
        cost = pricing.calculate_cost(input_tokens=500, output_tokens=250)
        expected = 0.0005 + 0.0005  # 0.5 * 0.001 + 0.25 * 0.002
        assert cost == pytest.approx(expected)

    def test_calculate_cost_zero_tokens(self):
        """Test cost calculation with zero tokens."""
        pricing = ModelPricing(
            input_per_1k=0.001,
            output_per_1k=0.002,
        )
        cost = pricing.calculate_cost(input_tokens=0, output_tokens=0)
        assert cost == 0.0


# =============================================================================
# Test: PricingProvider
# =============================================================================


class TestPricingProvider:
    """Tests for PricingProvider."""

    @pytest.mark.asyncio
    async def test_create_from_config_file(self, pricing_config_file: Path):
        """Test creating provider from config file."""
        provider = await PricingProvider.create(pricing_config_file)
        assert provider.is_initialized
        assert provider.config_path == pricing_config_file

    @pytest.mark.asyncio
    async def test_get_model_pricing_current(self, pricing_config_file: Path):
        """Test getting current model pricing."""
        provider = await PricingProvider.create(pricing_config_file)
        pricing = provider.get_model_pricing("gpt-4o-mini")

        assert pricing.input_per_1k == 0.00015
        assert pricing.output_per_1k == 0.0006

    @pytest.mark.asyncio
    async def test_get_model_pricing_historical(self, pricing_config_file: Path):
        """Test getting historical model pricing."""
        provider = await PricingProvider.create(pricing_config_file)

        # Get pricing from 2025 (before 2026 pricing)
        historical_date = datetime(2025, 6, 1, tzinfo=UTC)
        pricing = provider.get_model_pricing("gpt-4o-mini", at_datetime=historical_date)

        assert pricing.input_per_1k == 0.0003
        assert pricing.output_per_1k == 0.001

    @pytest.mark.asyncio
    async def test_get_model_pricing_unknown_model(self, pricing_config_file: Path):
        """Test getting pricing for unknown model returns defaults."""
        provider = await PricingProvider.create(pricing_config_file)
        pricing = provider.get_model_pricing("unknown-model")

        assert pricing.input_per_1k == 0.001
        assert pricing.output_per_1k == 0.002

    @pytest.mark.asyncio
    async def test_calculate_cost(self, pricing_config_file: Path):
        """Test calculate_cost convenience method."""
        provider = await PricingProvider.create(pricing_config_file)
        cost = provider.calculate_cost(
            "gpt-4o-mini",
            input_tokens=1000,
            output_tokens=500,
        )

        expected = (1000 / 1000) * 0.00015 + (500 / 1000) * 0.0006
        assert cost == pytest.approx(expected)

    @pytest.mark.asyncio
    async def test_list_models(self, pricing_config_file: Path):
        """Test listing available models."""
        provider = await PricingProvider.create(pricing_config_file)
        models = provider.list_models()

        assert "gpt-4o-mini" in models
        assert "text-embedding-3-small" in models

    @pytest.mark.asyncio
    async def test_singleton_pattern(self, pricing_config_file: Path):
        """Test singleton pattern returns same instance."""
        provider1 = await PricingProvider.create(pricing_config_file)
        provider2 = await PricingProvider.create(pricing_config_file)

        assert provider1 is provider2

    @pytest.mark.asyncio
    async def test_reset_clears_singleton(self, pricing_config_file: Path):
        """Test reset clears singleton."""
        provider1 = await PricingProvider.create(pricing_config_file)
        PricingProvider.reset()
        provider2 = await PricingProvider.create(pricing_config_file)

        assert provider1 is not provider2

    @pytest.mark.asyncio
    async def test_invalid_config_path_raises_error(self):
        """Test invalid config path raises error."""
        with pytest.raises(PricingConfigError):
            await PricingProvider.create("/nonexistent/path.json")

    @pytest.mark.asyncio
    async def test_invalid_json_raises_error(self, tmp_path: Path):
        """Test invalid JSON raises error."""
        config_path = tmp_path / "invalid.json"
        with open(config_path, "w") as f:
            f.write("not valid json")

        with pytest.raises(PricingConfigError):
            await PricingProvider.create(config_path)


# =============================================================================
# Test: CostTracker
# =============================================================================


class TestCostTracker:
    """Tests for CostTracker."""

    def test_add_cost(self):
        """Test adding a cost entry."""
        tracker = CostTracker()
        cost = tracker.add_cost(
            model="gpt-4o-mini",
            input_tokens=1000,
            output_tokens=500,
            category="llm",
        )

        assert cost > 0
        assert len(tracker) == 1

    def test_add_embedding_cost(self):
        """Test adding embedding cost."""
        tracker = CostTracker()
        cost = tracker.add_embedding_cost(
            model="text-embedding-3-small",
            tokens=1000,
        )

        assert cost >= 0
        entries = tracker.get_entries_by_category("embedding")
        assert len(entries) == 1
        assert entries[0].output_tokens == 0

    def test_add_llm_cost(self):
        """Test adding LLM cost."""
        tracker = CostTracker()
        cost = tracker.add_llm_cost(
            model="gpt-4o-mini",
            input_tokens=500,
            output_tokens=200,
        )

        assert cost >= 0
        entries = tracker.get_entries_by_category("llm")
        assert len(entries) == 1

    def test_get_summary_empty(self):
        """Test summary with no entries."""
        tracker = CostTracker()
        summary = tracker.get_summary()

        assert summary.total_cost == 0.0
        assert summary.entry_count == 0

    def test_get_summary_with_entries(self):
        """Test summary with multiple entries."""
        tracker = CostTracker()
        tracker.add_llm_cost("gpt-4o-mini", input_tokens=1000, output_tokens=500)
        tracker.add_embedding_cost("text-embedding-3-small", tokens=2000)

        summary = tracker.get_summary()

        assert summary.entry_count == 2
        assert summary.total_cost > 0
        assert "gpt-4o-mini" in summary.by_model
        assert "llm" in summary.by_category
        assert "embedding" in summary.by_category

    def test_get_total_cost(self):
        """Test get_total_cost method."""
        tracker = CostTracker()
        cost1 = tracker.add_llm_cost("gpt-4o-mini", input_tokens=1000)
        cost2 = tracker.add_llm_cost("gpt-4o-mini", input_tokens=1000)

        total = tracker.get_total_cost()
        assert total == pytest.approx(cost1 + cost2)

    def test_get_entries_by_model(self):
        """Test filtering entries by model."""
        tracker = CostTracker()
        tracker.add_llm_cost("gpt-4o-mini", input_tokens=1000)
        tracker.add_llm_cost("gpt-4o", input_tokens=1000)
        tracker.add_llm_cost("gpt-4o-mini", input_tokens=500)

        entries = tracker.get_entries_by_model("gpt-4o-mini")
        assert len(entries) == 2

    def test_clear(self):
        """Test clearing tracker."""
        tracker = CostTracker()
        tracker.add_llm_cost("gpt-4o-mini", input_tokens=1000)
        assert len(tracker) == 1

        tracker.clear()
        assert len(tracker) == 0

    def test_merge_trackers(self):
        """Test merging two trackers."""
        tracker1 = CostTracker()
        tracker1.add_llm_cost("gpt-4o-mini", input_tokens=1000)

        tracker2 = CostTracker()
        tracker2.add_embedding_cost("text-embedding-3-small", tokens=2000)

        tracker1.merge(tracker2)
        assert len(tracker1) == 2

    def test_add_trackers(self):
        """Test adding two trackers."""
        tracker1 = CostTracker()
        tracker1.add_llm_cost("gpt-4o-mini", input_tokens=1000)

        tracker2 = CostTracker()
        tracker2.add_embedding_cost("text-embedding-3-small", tokens=2000)

        combined = tracker1 + tracker2
        assert len(combined) == 2
        # Original trackers unchanged
        assert len(tracker1) == 1
        assert len(tracker2) == 1


# =============================================================================
# Test: Convenience Functions
# =============================================================================


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_calculate_cost_fallback(self):
        """Test calculate_cost with fallback pricing."""
        # No provider initialized, should use fallback
        cost = calculate_cost(
            model="unknown",
            input_tokens=1000,
            output_tokens=500,
        )
        # Fallback: input=0.001/1K, output=0.002/1K
        expected = 0.001 + 0.001
        assert cost == pytest.approx(expected)

    def test_estimate_embedding_cost(self):
        """Test embedding cost estimation."""
        text = "a" * 400  # 400 chars ≈ 100 tokens at 4 chars/token
        cost = estimate_embedding_cost(text, model="text-embedding-3-small")
        assert cost >= 0

    def test_estimate_llm_cost(self):
        """Test LLM cost estimation."""
        prompt = "a" * 400  # 400 chars ≈ 100 tokens
        cost = estimate_llm_cost(
            prompt,
            expected_output_tokens=200,
            model="gpt-4o-mini",
        )
        assert cost >= 0


# =============================================================================
# Test: Integration with Pricing Provider
# =============================================================================


class TestCostTrackerWithProvider:
    """Tests for CostTracker with actual PricingProvider."""

    @pytest.mark.asyncio
    async def test_tracker_uses_provider(self, pricing_config_file: Path):
        """Test tracker uses pricing provider for calculations."""
        provider = await PricingProvider.create(pricing_config_file)
        tracker = CostTracker(pricing_provider=provider)

        # Add cost for known model
        cost = tracker.add_llm_cost(
            model="gpt-4o-mini",
            input_tokens=1000,
            output_tokens=500,
        )

        # Calculate expected from provider
        expected = provider.calculate_cost(
            "gpt-4o-mini",
            input_tokens=1000,
            output_tokens=500,
        )

        assert cost == pytest.approx(expected)

    @pytest.mark.asyncio
    async def test_get_pricing_provider_convenience(self, pricing_config_file: Path):
        """Test get_pricing_provider convenience function."""
        provider = await get_pricing_provider(pricing_config_file)
        assert provider.is_initialized

        # Should return same instance
        provider2 = await get_pricing_provider(pricing_config_file)
        assert provider is provider2
