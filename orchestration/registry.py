"""
RAG-Advanced Strategy Registry.

A central registry for type-safe strategy lookup and metadata management.
Provides decorator-based registration, validation, and strategy discovery.

Usage:
    from orchestration.registry import StrategyRegistry, register_strategy

    # Register a strategy using decorator
    @register_strategy(
        name="reranking",
        description="Cross-encoder reranking for precision",
        strategy_type=StrategyType.RERANKING,
        required_resources=[ResourceType.DATABASE, ResourceType.RERANKER],
    )
    async def reranking_strategy(
        query: str,
        config: StrategyConfig,
    ) -> ExecutionResult:
        ...

    # Or register manually
    registry = StrategyRegistry()
    registry.register("custom", custom_func, metadata)

    # Get and execute strategy
    strategy = registry.get("reranking")
    result = await strategy(query, config)

    # List available strategies
    strategies = registry.list_strategies()
"""

from __future__ import annotations

import inspect
import logging
import threading
from functools import wraps
from typing import Any, Callable, TypeVar

from orchestration.errors import (
    InvalidStrategyError,
    StrategyAlreadyRegisteredError,
    StrategyNotFoundError,
)
from orchestration.models import (
    ExecutionResult,
    ResourceType,
    StrategyConfig,
    StrategyFunc,
    StrategyMetadata,
    StrategyType,
)


logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


class StrategyRegistry:
    """
    Central registry for RAG strategies.

    Provides type-safe strategy registration, lookup, and discovery.
    Supports decorator-based and manual registration with validation.

    Thread-safe for concurrent access.

    Attributes:
        _strategies: Internal dictionary of registered strategies.
        _metadata: Internal dictionary of strategy metadata.

    Example:
        >>> registry = StrategyRegistry()
        >>> registry.register("standard", standard_search, metadata)
        >>> strategy = registry.get("standard")
        >>> result = await strategy(query, config)
    """

    # Singleton instance for global registry
    _instance: "StrategyRegistry | None" = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "StrategyRegistry":
        """Ensure singleton pattern for global registry."""
        if cls._instance is None:
            with cls._lock:
                # Double-check locking pattern
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._strategies = {}
                    instance._metadata = {}
                    instance._initialized = True
                    cls._instance = instance
        return cls._instance

    def __init__(self) -> None:
        """Initialize the registry (only on first creation)."""
        # Skip re-initialization on singleton access
        pass

    @classmethod
    def reset(cls) -> None:
        """
        Reset the singleton instance.

        Primarily useful for testing to start with a clean registry.
        """
        with cls._lock:
            if cls._instance is not None:
                cls._instance._strategies.clear()
                cls._instance._metadata.clear()
            cls._instance = None

    def register(
        self,
        name: str,
        func: StrategyFunc,
        metadata: StrategyMetadata | None = None,
        *,
        allow_override: bool = False,
    ) -> None:
        """
        Register a strategy function with the registry.

        Args:
            name: Unique identifier for the strategy.
            func: Async function implementing the strategy.
            metadata: Optional metadata describing the strategy.
            allow_override: If True, allow overwriting existing registration.

        Raises:
            StrategyAlreadyRegisteredError: If name exists and allow_override is False.
            InvalidStrategyError: If the function doesn't meet requirements.

        Example:
            >>> registry.register(
            ...     "standard",
            ...     standard_search,
            ...     StrategyMetadata(
            ...         name="standard",
            ...         description="Basic semantic search",
            ...         strategy_type=StrategyType.STANDARD,
            ...     ),
            ... )
        """
        # Normalize name
        name = name.lower().strip()

        # Check for duplicate registration
        if name in self._strategies and not allow_override:
            raise StrategyAlreadyRegisteredError(name)

        # Validate the strategy function
        self._validate_strategy(name, func)

        # Create default metadata if not provided
        if metadata is None:
            metadata = StrategyMetadata(
                name=name,
                description=f"Strategy: {name}",
                strategy_type=StrategyType.STANDARD,
            )

        # Ensure metadata name matches registration name
        if metadata.name != name:
            metadata = StrategyMetadata(
                name=name,
                description=metadata.description,
                strategy_type=metadata.strategy_type,
                version=metadata.version,
                required_resources=metadata.required_resources,
                estimated_latency_ms=metadata.estimated_latency_ms,
                estimated_cost_per_query=metadata.estimated_cost_per_query,
                precision_rating=metadata.precision_rating,
                tags=metadata.tags,
            )

        # Register the strategy
        self._strategies[name] = func
        self._metadata[name] = metadata

        logger.info(
            f"Registered strategy: {name}",
            extra={
                "strategy_name": name,
                "strategy_type": metadata.strategy_type.value,
                "version": metadata.version,
            },
        )

    def _validate_strategy(self, name: str, func: StrategyFunc) -> None:
        """
        Validate that a function meets strategy requirements.

        Requirements:
        - Must be callable
        - Must be a coroutine function (async)
        - Should accept query (str) and config (StrategyConfig) parameters

        Args:
            name: Strategy name (for error messages).
            func: Function to validate.

        Raises:
            InvalidStrategyError: If validation fails.
        """
        # Must be callable
        if not callable(func):
            raise InvalidStrategyError(name, "Strategy must be callable")

        # Must be async
        if not inspect.iscoroutinefunction(func):
            raise InvalidStrategyError(name, "Strategy must be an async function (coroutine)")

        # Check signature for expected parameters
        sig = inspect.signature(func)
        params = list(sig.parameters.keys())

        # We expect at least query parameter, config is optional
        if len(params) < 1:
            raise InvalidStrategyError(
                name,
                "Strategy must accept at least a 'query' parameter",
            )

        logger.debug(f"Validated strategy: {name}", extra={"parameters": params})

    def get(self, name: str) -> StrategyFunc:
        """
        Get a registered strategy by name.

        Args:
            name: The strategy name to look up.

        Returns:
            The registered strategy function.

        Raises:
            StrategyNotFoundError: If the strategy is not registered.

        Example:
            >>> strategy = registry.get("reranking")
            >>> result = await strategy(query, config)
        """
        name = name.lower().strip()

        if name not in self._strategies:
            raise StrategyNotFoundError(
                strategy_name=name,
                available_strategies=list(self._strategies.keys()),
            )

        return self._strategies[name]

    def get_metadata(self, name: str) -> StrategyMetadata:
        """
        Get metadata for a registered strategy.

        Args:
            name: The strategy name to look up.

        Returns:
            The strategy metadata.

        Raises:
            StrategyNotFoundError: If the strategy is not registered.
        """
        name = name.lower().strip()

        if name not in self._metadata:
            raise StrategyNotFoundError(
                strategy_name=name,
                available_strategies=list(self._strategies.keys()),
            )

        return self._metadata[name]

    def list_strategies(self) -> list[StrategyMetadata]:
        """
        List all registered strategies with their metadata.

        Returns:
            List of StrategyMetadata for all registered strategies.

        Example:
            >>> strategies = registry.list_strategies()
            >>> for s in strategies:
            ...     print(f"{s.name}: {s.description}")
        """
        return list(self._metadata.values())

    def list_strategy_names(self) -> list[str]:
        """
        List names of all registered strategies.

        Returns:
            List of strategy names.
        """
        return list(self._strategies.keys())

    def is_registered(self, name: str) -> bool:
        """
        Check if a strategy is registered.

        Args:
            name: The strategy name to check.

        Returns:
            True if registered, False otherwise.
        """
        return name.lower().strip() in self._strategies

    def filter_by_type(self, strategy_type: StrategyType) -> list[StrategyMetadata]:
        """
        Filter strategies by type.

        Args:
            strategy_type: The type to filter by.

        Returns:
            List of matching strategy metadata.
        """
        return [m for m in self._metadata.values() if m.strategy_type == strategy_type]

    def filter_by_resource(self, resource: ResourceType) -> list[StrategyMetadata]:
        """
        Filter strategies by required resource.

        Args:
            resource: The resource type to filter by.

        Returns:
            List of strategies that require the specified resource.
        """
        return [m for m in self._metadata.values() if resource in m.required_resources]

    def filter_by_tag(self, tag: str) -> list[StrategyMetadata]:
        """
        Filter strategies by tag.

        Args:
            tag: The tag to filter by (case-insensitive).

        Returns:
            List of strategies with the specified tag.
        """
        tag = tag.lower()
        return [m for m in self._metadata.values() if tag in [t.lower() for t in m.tags]]

    def unregister(self, name: str) -> bool:
        """
        Remove a strategy from the registry.

        Args:
            name: The strategy name to remove.

        Returns:
            True if removed, False if not found.
        """
        name = name.lower().strip()

        if name in self._strategies:
            del self._strategies[name]
            del self._metadata[name]
            logger.info(f"Unregistered strategy: {name}")
            return True

        return False

    def __len__(self) -> int:
        """Return number of registered strategies."""
        return len(self._strategies)

    def __contains__(self, name: str) -> bool:
        """Check if strategy is registered using 'in' operator."""
        return self.is_registered(name)

    def __iter__(self):
        """Iterate over strategy names."""
        return iter(self._strategies.keys())


# =============================================================================
# Global Registry Instance
# =============================================================================


def get_registry() -> StrategyRegistry:
    """
    Get the global strategy registry instance.

    Returns:
        The global StrategyRegistry singleton.
    """
    return StrategyRegistry()


# =============================================================================
# Decorator for Strategy Registration
# =============================================================================


def register_strategy(
    name: str,
    description: str,
    strategy_type: StrategyType,
    *,
    version: str = "1.0.0",
    required_resources: list[ResourceType] | None = None,
    estimated_latency_ms: tuple[int, int] = (100, 500),
    estimated_cost_per_query: float = 0.001,
    precision_rating: int = 3,
    tags: list[str] | None = None,
    allow_override: bool = False,
) -> Callable[[F], F]:
    """
    Decorator to register a strategy function with the global registry.

    This decorator registers the function and returns it unchanged,
    allowing it to be used both through the registry and directly.

    Args:
        name: Unique identifier for the strategy.
        description: Human-readable description.
        strategy_type: The category of strategy.
        version: Version string (default "1.0.0").
        required_resources: List of required resources.
        estimated_latency_ms: Latency range tuple (min, max).
        estimated_cost_per_query: Estimated cost in USD.
        precision_rating: Precision rating 1-5.
        tags: Searchable tags.
        allow_override: Allow overwriting existing registration.

    Returns:
        Decorator function that registers and returns the strategy.

    Example:
        >>> @register_strategy(
        ...     name="reranking",
        ...     description="Cross-encoder reranking for high precision",
        ...     strategy_type=StrategyType.RERANKING,
        ...     required_resources=[ResourceType.DATABASE, ResourceType.RERANKER],
        ...     precision_rating=5,
        ...     tags=["precision", "cross-encoder"],
        ... )
        ... async def reranking_strategy(
        ...     query: str,
        ...     config: StrategyConfig,
        ... ) -> ExecutionResult:
        ...     ...
    """

    def decorator(func: F) -> F:
        metadata = StrategyMetadata(
            name=name,
            description=description,
            strategy_type=strategy_type,
            version=version,
            required_resources=required_resources or [],
            estimated_latency_ms=estimated_latency_ms,
            estimated_cost_per_query=estimated_cost_per_query,
            precision_rating=precision_rating,
            tags=tags or [],
        )

        # Register with global registry
        registry = get_registry()
        registry.register(name, func, metadata, allow_override=allow_override)

        # Add metadata to function for introspection
        func._strategy_metadata = metadata  # type: ignore[attr-defined]

        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> ExecutionResult:
            return await func(*args, **kwargs)

        return func  # Return original function, not wrapper

    return decorator


# =============================================================================
# Convenience Functions
# =============================================================================


def get_strategy(name: str) -> StrategyFunc:
    """
    Get a strategy from the global registry.

    Convenience function for get_registry().get(name).

    Args:
        name: Strategy name to look up.

    Returns:
        The strategy function.

    Raises:
        StrategyNotFoundError: If not found.
    """
    return get_registry().get(name)


def get_strategy_metadata(name: str) -> StrategyMetadata:
    """
    Get strategy metadata from the global registry.

    Convenience function for get_registry().get_metadata(name).

    Args:
        name: Strategy name to look up.

    Returns:
        The strategy metadata.

    Raises:
        StrategyNotFoundError: If not found.
    """
    return get_registry().get_metadata(name)


def list_strategies() -> list[StrategyMetadata]:
    """
    List all registered strategies.

    Convenience function for get_registry().list_strategies().

    Returns:
        List of all strategy metadata.
    """
    return get_registry().list_strategies()


def list_strategy_names() -> list[str]:
    """
    List all registered strategy names.

    Convenience function for get_registry().list_strategy_names().

    Returns:
        List of strategy names.
    """
    return get_registry().list_strategy_names()
