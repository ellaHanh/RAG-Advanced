"""
RAG-Advanced Pricing Provider.

Async pricing provider that loads versioned pricing data from JSON configuration.
Supports historical pricing lookup for accurate cost calculation over time.

Usage:
    from orchestration.pricing import PricingProvider, get_pricing_provider

    # Initialize (typically at application startup)
    provider = await PricingProvider.create()

    # Get current pricing for a model
    pricing = provider.get_model_pricing("gpt-4o-mini")
    cost = pricing.calculate_cost(input_tokens=100, output_tokens=50)

    # Get historical pricing
    historical = provider.get_model_pricing(
        "gpt-4o",
        at_datetime=datetime(2025, 6, 1, tzinfo=UTC)
    )
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiofiles
from pydantic import BaseModel, ConfigDict, Field

from orchestration.errors import PricingConfigError


logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

# Default path to pricing configuration
DEFAULT_PRICING_PATH = Path(__file__).parent.parent / "config" / "pricing.json"


# =============================================================================
# Pricing Models
# =============================================================================


class ModelPricing(BaseModel):
    """
    Pricing information for a single model.

    Attributes:
        input_per_1k: Cost per 1000 input tokens in USD.
        output_per_1k: Cost per 1000 output tokens in USD.
        description: Optional description of the model.
    """

    model_config = ConfigDict(frozen=True)

    input_per_1k: float = Field(..., ge=0.0, description="Cost per 1K input tokens")
    output_per_1k: float = Field(..., ge=0.0, description="Cost per 1K output tokens")
    description: str = Field(default="", description="Model description")

    def calculate_cost(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> float:
        """
        Calculate total cost for given token counts.

        Args:
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.

        Returns:
            Total cost in USD.
        """
        input_cost = (input_tokens / 1000) * self.input_per_1k
        output_cost = (output_tokens / 1000) * self.output_per_1k
        return input_cost + output_cost


class PricingPeriod(BaseModel):
    """
    Pricing data for a specific time period.

    Attributes:
        effective_date: When this pricing became effective.
        currency: Currency code (always USD).
        models: Dictionary of model name to pricing.
    """

    model_config = ConfigDict(frozen=True)

    effective_date: datetime = Field(..., description="When pricing became effective")
    currency: str = Field(default="USD", description="Currency code")
    models: dict[str, ModelPricing] = Field(default_factory=dict, description="Model pricing")


class PricingConfig(BaseModel):
    """
    Complete pricing configuration.

    Attributes:
        pricing_history: List of pricing periods, ordered by effective_date descending.
        defaults: Default pricing for unknown models.
    """

    model_config = ConfigDict(frozen=True)

    pricing_history: list[PricingPeriod] = Field(..., description="Historical pricing periods")
    defaults: ModelPricing = Field(..., description="Default pricing for unknown models")


# =============================================================================
# Pricing Provider
# =============================================================================


class PricingProvider:
    """
    Async pricing provider with historical pricing support.

    Loads pricing data from JSON configuration and provides
    model-specific pricing lookup with support for historical dates.

    Thread-safe singleton implementation.

    Example:
        >>> provider = await PricingProvider.create()
        >>> pricing = provider.get_model_pricing("gpt-4o-mini")
        >>> cost = pricing.calculate_cost(input_tokens=1000, output_tokens=500)
        >>> print(f"Cost: ${cost:.6f}")
    """

    _instance: "PricingProvider | None" = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        """Initialize the provider (use create() for async initialization)."""
        self._config: PricingConfig | None = None
        self._config_path: Path | None = None
        self._initialized: bool = False

    @classmethod
    async def create(
        cls,
        config_path: Path | str | None = None,
    ) -> "PricingProvider":
        """
        Create and initialize a PricingProvider.

        Args:
            config_path: Path to pricing JSON file. Uses default if not specified.

        Returns:
            Initialized PricingProvider instance.

        Raises:
            PricingConfigError: If configuration cannot be loaded or parsed.
        """
        with cls._lock:
            if cls._instance is None or not cls._instance._initialized:
                instance = cls()
                await instance._initialize(config_path)
                cls._instance = instance
            return cls._instance

    @classmethod
    def get_instance(cls) -> "PricingProvider | None":
        """
        Get the current singleton instance without initialization.

        Returns:
            The singleton instance if initialized, None otherwise.
        """
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance (for testing)."""
        with cls._lock:
            cls._instance = None

    async def _initialize(self, config_path: Path | str | None = None) -> None:
        """
        Initialize the provider by loading pricing configuration.

        Args:
            config_path: Path to pricing JSON file.

        Raises:
            PricingConfigError: If configuration cannot be loaded.
        """
        self._config_path = Path(config_path) if config_path else DEFAULT_PRICING_PATH

        try:
            self._config = await self._load_config()
            self._initialized = True
            logger.info(
                f"Pricing provider initialized",
                extra={
                    "config_path": str(self._config_path),
                    "periods": len(self._config.pricing_history),
                    "models": self._get_all_model_names(),
                },
            )
        except Exception as e:
            raise PricingConfigError(f"Failed to load pricing config: {e}") from e

    async def _load_config(self) -> PricingConfig:
        """
        Load pricing configuration from JSON file.

        Returns:
            Parsed PricingConfig.

        Raises:
            PricingConfigError: If file cannot be read or parsed.
        """
        if not self._config_path or not self._config_path.exists():
            raise PricingConfigError(f"Pricing config not found: {self._config_path}")

        try:
            async with aiofiles.open(self._config_path, "r") as f:
                content = await f.read()
                data = json.loads(content)
        except json.JSONDecodeError as e:
            raise PricingConfigError(f"Invalid JSON in pricing config: {e}") from e
        except IOError as e:
            raise PricingConfigError(f"Cannot read pricing config: {e}") from e

        return self._parse_config(data)

    def _parse_config(self, data: dict[str, Any]) -> PricingConfig:
        """
        Parse raw JSON data into PricingConfig.

        Args:
            data: Raw JSON data.

        Returns:
            Parsed PricingConfig.
        """
        # Parse pricing history
        pricing_history = []
        for period_data in data.get("pricing_history", []):
            # Parse effective date
            effective_date = datetime.fromisoformat(
                period_data["effective_date"].replace("Z", "+00:00")
            )

            # Parse model pricing
            models = {}
            for model_name, model_data in period_data.get("models", {}).items():
                models[model_name] = ModelPricing(
                    input_per_1k=model_data["input_per_1k"],
                    output_per_1k=model_data["output_per_1k"],
                    description=model_data.get("description", ""),
                )

            pricing_history.append(
                PricingPeriod(
                    effective_date=effective_date,
                    currency=period_data.get("currency", "USD"),
                    models=models,
                )
            )

        # Sort by effective date descending (most recent first)
        pricing_history.sort(key=lambda p: p.effective_date, reverse=True)

        # Parse defaults
        defaults_data = data.get("defaults", {})
        defaults = ModelPricing(
            input_per_1k=defaults_data.get("input_per_1k", 0.001),
            output_per_1k=defaults_data.get("output_per_1k", 0.002),
            description=defaults_data.get("description", "Default pricing"),
        )

        return PricingConfig(
            pricing_history=pricing_history,
            defaults=defaults,
        )

    def _get_all_model_names(self) -> list[str]:
        """Get all unique model names across all pricing periods."""
        if not self._config:
            return []
        models = set()
        for period in self._config.pricing_history:
            models.update(period.models.keys())
        return sorted(models)

    def get_model_pricing(
        self,
        model_name: str,
        at_datetime: datetime | None = None,
    ) -> ModelPricing:
        """
        Get pricing for a model, optionally at a specific date.

        Args:
            model_name: Name of the model (e.g., "gpt-4o-mini").
            at_datetime: Optional datetime for historical pricing.
                        Uses current time if not specified.

        Returns:
            ModelPricing for the specified model.
            Returns default pricing if model not found.

        Example:
            >>> pricing = provider.get_model_pricing("gpt-4o-mini")
            >>> historical = provider.get_model_pricing(
            ...     "gpt-4o",
            ...     at_datetime=datetime(2025, 6, 1, tzinfo=UTC)
            ... )
        """
        if not self._config:
            logger.warning("Pricing provider not initialized, using defaults")
            return ModelPricing(
                input_per_1k=0.001,
                output_per_1k=0.002,
                description="Fallback default",
            )

        # Normalize model name
        model_name = model_name.lower().strip()

        # Use current time if not specified
        if at_datetime is None:
            at_datetime = datetime.now(UTC)
        elif at_datetime.tzinfo is None:
            # Assume UTC for naive datetimes
            at_datetime = at_datetime.replace(tzinfo=UTC)

        # Find applicable pricing period
        for period in self._config.pricing_history:
            if period.effective_date <= at_datetime:
                # Found applicable period, check for model
                if model_name in period.models:
                    return period.models[model_name]
                # Model not in this period, check for partial match
                for name, pricing in period.models.items():
                    if model_name in name.lower() or name.lower() in model_name:
                        logger.debug(f"Fuzzy match: {model_name} -> {name}")
                        return pricing
                break

        # Model not found, use defaults
        logger.debug(f"Model '{model_name}' not found, using default pricing")
        return self._config.defaults

    def calculate_cost(
        self,
        model_name: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        at_datetime: datetime | None = None,
    ) -> float:
        """
        Calculate cost for a model with given token counts.

        Convenience method that combines get_model_pricing and calculate_cost.

        Args:
            model_name: Name of the model.
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.
            at_datetime: Optional datetime for historical pricing.

        Returns:
            Total cost in USD.

        Example:
            >>> cost = provider.calculate_cost(
            ...     "gpt-4o-mini",
            ...     input_tokens=1000,
            ...     output_tokens=500
            ... )
        """
        pricing = self.get_model_pricing(model_name, at_datetime)
        return pricing.calculate_cost(input_tokens, output_tokens)

    def list_models(self, at_datetime: datetime | None = None) -> list[str]:
        """
        List all available models at a given time.

        Args:
            at_datetime: Optional datetime for historical lookup.

        Returns:
            List of model names.
        """
        if not self._config:
            return []

        if at_datetime is None:
            at_datetime = datetime.now(UTC)
        elif at_datetime.tzinfo is None:
            at_datetime = at_datetime.replace(tzinfo=UTC)

        for period in self._config.pricing_history:
            if period.effective_date <= at_datetime:
                return list(period.models.keys())

        return []

    def get_all_pricing(
        self,
        at_datetime: datetime | None = None,
    ) -> dict[str, ModelPricing]:
        """
        Get pricing for all models at a given time.

        Args:
            at_datetime: Optional datetime for historical lookup.

        Returns:
            Dictionary of model name to pricing.
        """
        if not self._config:
            return {}

        if at_datetime is None:
            at_datetime = datetime.now(UTC)
        elif at_datetime.tzinfo is None:
            at_datetime = at_datetime.replace(tzinfo=UTC)

        for period in self._config.pricing_history:
            if period.effective_date <= at_datetime:
                return dict(period.models)

        return {}

    @property
    def is_initialized(self) -> bool:
        """Check if provider is initialized."""
        return self._initialized

    @property
    def config_path(self) -> Path | None:
        """Get the configuration file path."""
        return self._config_path


# =============================================================================
# Module-level Convenience Functions
# =============================================================================


async def get_pricing_provider(
    config_path: Path | str | None = None,
) -> PricingProvider:
    """
    Get or create the global pricing provider.

    Args:
        config_path: Optional path to pricing config.

    Returns:
        Initialized PricingProvider.
    """
    return await PricingProvider.create(config_path)


def get_pricing_provider_sync() -> PricingProvider | None:
    """
    Get the pricing provider if already initialized.

    Returns:
        PricingProvider if initialized, None otherwise.
    """
    return PricingProvider.get_instance()
