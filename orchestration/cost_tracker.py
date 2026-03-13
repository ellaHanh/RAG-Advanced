"""
RAG-Advanced Cost Tracker.

Thread-safe cost tracking and aggregation for API usage across strategy executions.
Provides per-model, per-strategy, and aggregate cost tracking.

Usage:
    from orchestration.cost_tracker import CostTracker

    # Create a tracker for a session/request
    tracker = CostTracker()

    # Track costs
    tracker.add_embedding_cost("text-embedding-3-small", tokens=1000)
    tracker.add_llm_cost("gpt-4o-mini", input_tokens=500, output_tokens=100)

    # Get summary
    summary = tracker.get_summary()
    print(f"Total cost: ${summary.total_cost:.6f}")
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from orchestration.pricing import PricingProvider, get_pricing_provider_sync


logger = logging.getLogger(__name__)


# =============================================================================
# Cost Entry Models
# =============================================================================


@dataclass
class CostEntry:
    """
    A single cost entry representing one API call.

    Attributes:
        model: Model name used.
        input_tokens: Number of input tokens.
        output_tokens: Number of output tokens.
        cost_usd: Calculated cost in USD.
        timestamp: When the cost was incurred.
        category: Category of the cost (embedding, llm, reranking).
        metadata: Additional metadata.
    """

    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    category: str = "llm"
    metadata: dict[str, Any] = field(default_factory=dict)


class CostSummary(BaseModel):
    """
    Summary of tracked costs.

    Attributes:
        total_cost: Total cost in USD.
        total_input_tokens: Total input tokens across all calls.
        total_output_tokens: Total output tokens across all calls.
        by_model: Cost breakdown by model.
        by_category: Cost breakdown by category.
        entry_count: Number of cost entries.
        start_time: When tracking started.
        end_time: Most recent entry time.
    """

    model_config = ConfigDict(frozen=True)

    total_cost: float = Field(default=0.0, description="Total cost in USD")
    total_input_tokens: int = Field(default=0, description="Total input tokens")
    total_output_tokens: int = Field(default=0, description="Total output tokens")
    by_model: dict[str, float] = Field(default_factory=dict, description="Cost by model")
    by_category: dict[str, float] = Field(default_factory=dict, description="Cost by category")
    entry_count: int = Field(default=0, description="Number of entries")
    start_time: datetime | None = Field(default=None, description="First entry time")
    end_time: datetime | None = Field(default=None, description="Last entry time")


# =============================================================================
# Cost Tracker
# =============================================================================


class CostTracker:
    """
    Thread-safe cost tracker for aggregating API costs.

    Tracks costs per API call and provides aggregation by model,
    category, and total. Uses PricingProvider for cost calculation.

    Thread-safe for concurrent tracking from multiple coroutines.

    Example:
        >>> tracker = CostTracker()
        >>> tracker.add_embedding_cost("text-embedding-3-small", tokens=1000)
        >>> tracker.add_llm_cost("gpt-4o-mini", input_tokens=500, output_tokens=100)
        >>> summary = tracker.get_summary()
        >>> print(f"Total: ${summary.total_cost:.6f}")
    """

    def __init__(
        self,
        pricing_provider: PricingProvider | None = None,
    ) -> None:
        """
        Initialize the cost tracker.

        Args:
            pricing_provider: Optional pricing provider.
                            Uses global provider if not specified.
        """
        self._entries: list[CostEntry] = []
        self._lock = threading.Lock()
        self._pricing_provider = pricing_provider

    def _get_provider(self) -> PricingProvider | None:
        """Get the pricing provider (lazy initialization)."""
        if self._pricing_provider is None:
            self._pricing_provider = get_pricing_provider_sync()
        return self._pricing_provider

    def _calculate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """
        Calculate cost using pricing provider.

        Falls back to default pricing if provider not available.

        Args:
            model: Model name.
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.

        Returns:
            Cost in USD.
        """
        provider = self._get_provider()
        if provider:
            return provider.calculate_cost(model, input_tokens, output_tokens)

        # Fallback default pricing
        default_input_per_1k = 0.001
        default_output_per_1k = 0.002
        return (
            (input_tokens / 1000) * default_input_per_1k
            + (output_tokens / 1000) * default_output_per_1k
        )

    def add_cost(
        self,
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        category: str = "llm",
        metadata: dict[str, Any] | None = None,
    ) -> float:
        """
        Add a cost entry.

        Args:
            model: Model name used.
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.
            category: Cost category (llm, embedding, reranking).
            metadata: Additional metadata.

        Returns:
            The cost in USD for this entry.
        """
        cost = self._calculate_cost(model, input_tokens, output_tokens)

        entry = CostEntry(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            category=category,
            metadata=metadata or {},
        )

        with self._lock:
            self._entries.append(entry)

        logger.debug(
            f"Added cost entry",
            extra={
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": cost,
                "category": category,
            },
        )

        return cost

    def add_embedding_cost(
        self,
        model: str = "text-embedding-3-small",
        tokens: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> float:
        """
        Add an embedding cost entry.

        Embeddings only have input tokens.

        Args:
            model: Embedding model name.
            tokens: Number of tokens embedded.
            metadata: Additional metadata.

        Returns:
            The cost in USD.
        """
        return self.add_cost(
            model=model,
            input_tokens=tokens,
            output_tokens=0,
            category="embedding",
            metadata=metadata,
        )

    def add_llm_cost(
        self,
        model: str = "gpt-4o-mini",
        input_tokens: int = 0,
        output_tokens: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> float:
        """
        Add an LLM cost entry.

        Args:
            model: LLM model name.
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.
            metadata: Additional metadata.

        Returns:
            The cost in USD.
        """
        return self.add_cost(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            category="llm",
            metadata=metadata,
        )

    def add_reranking_cost(
        self,
        model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        tokens: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> float:
        """
        Add a reranking cost entry.

        Note: Local reranking models typically have zero cost.
        This method exists for tracking purposes.

        Args:
            model: Reranking model name.
            tokens: Number of tokens processed.
            metadata: Additional metadata.

        Returns:
            The cost in USD (typically 0 for local models).
        """
        return self.add_cost(
            model=model,
            input_tokens=tokens,
            output_tokens=0,
            category="reranking",
            metadata=metadata,
        )

    def get_summary(self) -> CostSummary:
        """
        Get a summary of all tracked costs.

        Returns:
            CostSummary with aggregated data.
        """
        with self._lock:
            entries = list(self._entries)

        if not entries:
            return CostSummary()

        total_cost = sum(e.cost_usd for e in entries)
        total_input = sum(e.input_tokens for e in entries)
        total_output = sum(e.output_tokens for e in entries)

        # Aggregate by model
        by_model: dict[str, float] = {}
        for entry in entries:
            by_model[entry.model] = by_model.get(entry.model, 0.0) + entry.cost_usd

        # Aggregate by category
        by_category: dict[str, float] = {}
        for entry in entries:
            by_category[entry.category] = (
                by_category.get(entry.category, 0.0) + entry.cost_usd
            )

        # Time range
        timestamps = [e.timestamp for e in entries]
        start_time = min(timestamps)
        end_time = max(timestamps)

        return CostSummary(
            total_cost=total_cost,
            total_input_tokens=total_input,
            total_output_tokens=total_output,
            by_model=by_model,
            by_category=by_category,
            entry_count=len(entries),
            start_time=start_time,
            end_time=end_time,
        )

    def get_total_cost(self) -> float:
        """
        Get the total cost.

        Returns:
            Total cost in USD.
        """
        with self._lock:
            return sum(e.cost_usd for e in self._entries)

    def get_entries(self) -> list[CostEntry]:
        """
        Get all cost entries.

        Returns:
            List of CostEntry objects (copy).
        """
        with self._lock:
            return list(self._entries)

    def get_entries_by_category(self, category: str) -> list[CostEntry]:
        """
        Get entries filtered by category.

        Args:
            category: Category to filter by.

        Returns:
            List of matching CostEntry objects.
        """
        with self._lock:
            return [e for e in self._entries if e.category == category]

    def get_entries_by_model(self, model: str) -> list[CostEntry]:
        """
        Get entries filtered by model.

        Args:
            model: Model name to filter by.

        Returns:
            List of matching CostEntry objects.
        """
        model = model.lower()
        with self._lock:
            return [e for e in self._entries if e.model.lower() == model]

    def clear(self) -> None:
        """Clear all tracked entries."""
        with self._lock:
            self._entries.clear()
        logger.debug("Cost tracker cleared")

    def merge(self, other: "CostTracker") -> None:
        """
        Merge entries from another tracker.

        Args:
            other: Another CostTracker to merge from.
        """
        with self._lock, other._lock:
            self._entries.extend(other._entries)

    def __len__(self) -> int:
        """Return number of entries."""
        with self._lock:
            return len(self._entries)

    def __add__(self, other: "CostTracker") -> "CostTracker":
        """
        Combine two trackers into a new one.

        Args:
            other: Another CostTracker.

        Returns:
            New CostTracker with combined entries.
        """
        new_tracker = CostTracker(pricing_provider=self._pricing_provider)
        with self._lock:
            new_tracker._entries.extend(self._entries)
        with other._lock:
            new_tracker._entries.extend(other._entries)
        return new_tracker


# =============================================================================
# Convenience Functions
# =============================================================================


def calculate_cost(
    model: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> float:
    """
    Calculate cost for a single API call.

    Convenience function that uses the global pricing provider.

    Args:
        model: Model name.
        input_tokens: Number of input tokens.
        output_tokens: Number of output tokens.

    Returns:
        Cost in USD.
    """
    provider = get_pricing_provider_sync()
    if provider:
        return provider.calculate_cost(model, input_tokens, output_tokens)

    # Fallback
    default_input_per_1k = 0.001
    default_output_per_1k = 0.002
    return (
        (input_tokens / 1000) * default_input_per_1k
        + (output_tokens / 1000) * default_output_per_1k
    )


def estimate_embedding_cost(
    text: str,
    model: str = "text-embedding-3-small",
    chars_per_token: float = 4.0,
) -> float:
    """
    Estimate embedding cost based on text length.

    Uses a simple character-to-token ratio for estimation.

    Args:
        text: Text to embed.
        model: Embedding model name.
        chars_per_token: Average characters per token (default 4).

    Returns:
        Estimated cost in USD.
    """
    estimated_tokens = int(len(text) / chars_per_token)
    return calculate_cost(model, input_tokens=estimated_tokens)


def estimate_llm_cost(
    prompt: str,
    expected_output_tokens: int = 500,
    model: str = "gpt-4o-mini",
    chars_per_token: float = 4.0,
) -> float:
    """
    Estimate LLM cost based on prompt length.

    Args:
        prompt: Input prompt text.
        expected_output_tokens: Expected output length in tokens.
        model: LLM model name.
        chars_per_token: Average characters per token (default 4).

    Returns:
        Estimated cost in USD.
    """
    estimated_input_tokens = int(len(prompt) / chars_per_token)
    return calculate_cost(
        model,
        input_tokens=estimated_input_tokens,
        output_tokens=expected_output_tokens,
    )
