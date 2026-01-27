"""
RAG-Advanced Exception Hierarchy.

Provides a comprehensive set of exceptions for all modules in the system.
All exceptions inherit from RAGAdvancedError for easy catching.

Usage:
    from orchestration.errors import StrategyNotFoundError, StrategyExecutionError

    try:
        result = await execute_strategy("unknown", query, config)
    except StrategyNotFoundError as e:
        logger.error(f"Strategy not found: {e.strategy_name}")
    except StrategyExecutionError as e:
        logger.error(f"Execution failed: {e}")
"""

from __future__ import annotations

from typing import Any


class RAGAdvancedError(Exception):
    """
    Base exception for all RAG-Advanced errors.

    All custom exceptions in this project inherit from this class,
    allowing callers to catch all project-specific exceptions with
    a single except clause if needed.

    Attributes:
        message: Human-readable error description.
        details: Optional dictionary with additional error context.
    """

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        """
        Initialize the exception.

        Args:
            message: Human-readable error description.
            details: Optional dictionary with additional error context.
        """
        self.message = message
        self.details = details or {}
        super().__init__(message)

    def __str__(self) -> str:
        """Return string representation with details if present."""
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


# =============================================================================
# Strategy Registry Errors
# =============================================================================


class RegistryError(RAGAdvancedError):
    """Base exception for strategy registry errors."""

    pass


class StrategyNotFoundError(RegistryError):
    """
    Raised when a requested strategy is not found in the registry.

    Attributes:
        strategy_name: The name of the strategy that was not found.
        available_strategies: List of available strategy names.
    """

    def __init__(
        self,
        strategy_name: str,
        available_strategies: list[str] | None = None,
    ) -> None:
        """
        Initialize the exception.

        Args:
            strategy_name: The name of the strategy that was not found.
            available_strategies: List of available strategy names.
        """
        self.strategy_name = strategy_name
        self.available_strategies = available_strategies or []
        message = f"Strategy '{strategy_name}' not found in registry"
        details = {
            "strategy_name": strategy_name,
            "available_strategies": self.available_strategies,
        }
        super().__init__(message, details)


class StrategyAlreadyRegisteredError(RegistryError):
    """
    Raised when attempting to register a strategy with a name that already exists.

    Attributes:
        strategy_name: The name of the strategy that already exists.
    """

    def __init__(self, strategy_name: str) -> None:
        """
        Initialize the exception.

        Args:
            strategy_name: The name of the strategy that already exists.
        """
        self.strategy_name = strategy_name
        message = f"Strategy '{strategy_name}' is already registered"
        details = {"strategy_name": strategy_name}
        super().__init__(message, details)


class InvalidStrategyError(RegistryError):
    """
    Raised when a strategy does not conform to the expected interface.

    Attributes:
        strategy_name: The name of the invalid strategy.
        reason: Description of why the strategy is invalid.
    """

    def __init__(self, strategy_name: str, reason: str) -> None:
        """
        Initialize the exception.

        Args:
            strategy_name: The name of the invalid strategy.
            reason: Description of why the strategy is invalid.
        """
        self.strategy_name = strategy_name
        self.reason = reason
        message = f"Invalid strategy '{strategy_name}': {reason}"
        details = {"strategy_name": strategy_name, "reason": reason}
        super().__init__(message, details)


# =============================================================================
# Strategy Execution Errors
# =============================================================================


class ExecutionError(RAGAdvancedError):
    """Base exception for strategy execution errors."""

    pass


class StrategyExecutionError(ExecutionError):
    """
    Raised when a strategy execution fails.

    Attributes:
        strategy_name: The name of the strategy that failed.
        query: The query that was being processed.
        original_error: The original exception that caused the failure.
    """

    def __init__(
        self,
        strategy_name: str,
        query: str,
        original_error: Exception | None = None,
    ) -> None:
        """
        Initialize the exception.

        Args:
            strategy_name: The name of the strategy that failed.
            query: The query that was being processed.
            original_error: The original exception that caused the failure.
        """
        self.strategy_name = strategy_name
        self.query = query
        self.original_error = original_error
        message = f"Strategy '{strategy_name}' execution failed"
        details = {
            "strategy_name": strategy_name,
            "query": query[:100] + "..." if len(query) > 100 else query,
            "original_error": str(original_error) if original_error else None,
        }
        super().__init__(message, details)


class StrategyTimeoutError(ExecutionError):
    """
    Raised when a strategy execution times out.

    Attributes:
        strategy_name: The name of the strategy that timed out.
        timeout_seconds: The timeout value that was exceeded.
    """

    def __init__(self, strategy_name: str, timeout_seconds: float) -> None:
        """
        Initialize the exception.

        Args:
            strategy_name: The name of the strategy that timed out.
            timeout_seconds: The timeout value that was exceeded.
        """
        self.strategy_name = strategy_name
        self.timeout_seconds = timeout_seconds
        message = f"Strategy '{strategy_name}' timed out after {timeout_seconds}s"
        details = {"strategy_name": strategy_name, "timeout_seconds": timeout_seconds}
        super().__init__(message, details)


# =============================================================================
# Chain Execution Errors
# =============================================================================


class ChainError(RAGAdvancedError):
    """Base exception for chain execution errors."""

    pass


class ChainExecutionError(ChainError):
    """
    Raised when a chain execution fails.

    Attributes:
        step_index: The index of the step that failed.
        strategy_name: The name of the strategy that failed.
        original_error: The original exception that caused the failure.
    """

    def __init__(
        self,
        step_index: int,
        strategy_name: str,
        original_error: Exception | None = None,
    ) -> None:
        """
        Initialize the exception.

        Args:
            step_index: The index of the step that failed.
            strategy_name: The name of the strategy that failed.
            original_error: The original exception that caused the failure.
        """
        self.step_index = step_index
        self.strategy_name = strategy_name
        self.original_error = original_error
        message = f"Chain execution failed at step {step_index} ({strategy_name})"
        details = {
            "step_index": step_index,
            "strategy_name": strategy_name,
            "original_error": str(original_error) if original_error else None,
        }
        super().__init__(message, details)


class ChainConfigurationError(ChainError):
    """
    Raised when a chain configuration is invalid.

    Attributes:
        reason: Description of why the configuration is invalid.
    """

    def __init__(self, reason: str) -> None:
        """
        Initialize the exception.

        Args:
            reason: Description of why the configuration is invalid.
        """
        self.reason = reason
        message = f"Invalid chain configuration: {reason}"
        details = {"reason": reason}
        super().__init__(message, details)


# =============================================================================
# Evaluation Errors
# =============================================================================


class EvaluationError(RAGAdvancedError):
    """Base exception for evaluation errors."""

    pass


class InvalidInputError(EvaluationError):
    """
    Raised when input to an evaluation function is invalid.

    Attributes:
        parameter_name: The name of the invalid parameter.
        reason: Description of why the input is invalid.
    """

    def __init__(self, parameter_name: str, reason: str) -> None:
        """
        Initialize the exception.

        Args:
            parameter_name: The name of the invalid parameter.
            reason: Description of why the input is invalid.
        """
        self.parameter_name = parameter_name
        self.reason = reason
        message = f"Invalid input for '{parameter_name}': {reason}"
        details = {"parameter_name": parameter_name, "reason": reason}
        super().__init__(message, details)


class DatasetError(EvaluationError):
    """
    Raised when there's an error with a test dataset.

    Attributes:
        dataset_name: Optional name of the dataset.
        reason: Description of the error.
    """

    def __init__(
        self,
        message: str,
        dataset_name: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize the exception.

        Args:
            message: Error message or reason.
            dataset_name: Optional name of the dataset.
            details: Optional additional details dictionary.
        """
        self.dataset_name = dataset_name
        self.reason = message
        if dataset_name:
            full_message = f"Dataset error for '{dataset_name}': {message}"
        else:
            full_message = message
        final_details = details or {}
        if dataset_name:
            final_details["dataset_name"] = dataset_name
        super().__init__(full_message, final_details)


# =============================================================================
# Configuration Errors
# =============================================================================


class ConfigurationError(RAGAdvancedError):
    """Base exception for configuration errors."""

    pass


class PricingConfigError(ConfigurationError):
    """
    Raised when there's an error with pricing configuration.

    Attributes:
        reason: Description of the error.
    """

    def __init__(self, reason: str) -> None:
        """
        Initialize the exception.

        Args:
            reason: Description of the error.
        """
        self.reason = reason
        message = f"Pricing configuration error: {reason}"
        details = {"reason": reason}
        super().__init__(message, details)


# =============================================================================
# Database Errors
# =============================================================================


class DatabaseError(RAGAdvancedError):
    """Base exception for database errors."""

    pass


class ConnectionPoolError(DatabaseError):
    """
    Raised when there's an error with the database connection pool.

    Attributes:
        reason: Description of the error.
    """

    def __init__(self, reason: str) -> None:
        """
        Initialize the exception.

        Args:
            reason: Description of the error.
        """
        self.reason = reason
        message = f"Connection pool error: {reason}"
        details = {"reason": reason}
        super().__init__(message, details)


# =============================================================================
# API Errors
# =============================================================================


class APIError(RAGAdvancedError):
    """Base exception for API errors."""

    pass


class AuthenticationError(APIError):
    """
    Raised when authentication fails.

    Attributes:
        reason: Description of why authentication failed.
    """

    def __init__(self, reason: str = "Invalid or missing API key") -> None:
        """
        Initialize the exception.

        Args:
            reason: Description of why authentication failed.
        """
        self.reason = reason
        message = f"Authentication failed: {reason}"
        details = {"reason": reason}
        super().__init__(message, details)


class RateLimitError(APIError):
    """
    Raised when rate limit is exceeded.

    Attributes:
        limit: The rate limit that was exceeded.
        reset_at: When the rate limit will reset (timestamp).
    """

    def __init__(self, limit: int, reset_at: float | None = None) -> None:
        """
        Initialize the exception.

        Args:
            limit: The rate limit that was exceeded.
            reset_at: When the rate limit will reset (timestamp).
        """
        self.limit = limit
        self.reset_at = reset_at
        message = f"Rate limit exceeded: {limit} requests"
        details = {"limit": limit, "reset_at": reset_at}
        super().__init__(message, details)
