"""
RAG-Advanced Sequential Chain Executor.

Execute strategies sequentially in a chain, passing context between steps.
Each step's results are appended to the ChainContext for downstream use.

Usage:
    from orchestration.chain_executor import ChainExecutor

    executor = ChainExecutor()
    
    # Define chain
    chain = [
        ChainStep(strategy="standard", name="initial_search"),
        ChainStep(strategy="reranking", name="refine_results"),
    ]
    
    result = await executor.execute_chain(chain, query="What is AI?")
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Callable, Coroutine

from pydantic import BaseModel, ConfigDict, Field

from orchestration.errors import (
    ChainConfigurationError,
    ChainExecutionError,
)
from orchestration.executor import StrategyExecutor
from orchestration.models import (
    ChainConfig,
    ChainContext,
    ChainStep,
    Document,
    ExecutionResult,
    StrategyConfig,
)


logger = logging.getLogger(__name__)


# =============================================================================
# Chain Result
# =============================================================================


@dataclass
class ChainStepResult:
    """
    Result of a single chain step.

    Attributes:
        step_name: Name of the step.
        strategy_name: Strategy that was executed.
        result: Execution result.
        duration_ms: Step duration in milliseconds.
        input_document_count: Number of input documents.
        output_document_count: Number of output documents.
    """

    step_name: str
    strategy_name: str
    result: ExecutionResult
    duration_ms: int = 0
    input_document_count: int = 0
    output_document_count: int = 0


@dataclass
class ChainResult:
    """
    Result of a complete chain execution.

    Attributes:
        query: The original query.
        steps: Results from each step.
        final_context: Final ChainContext after all steps.
        total_latency_ms: Total execution time.
        total_cost_usd: Total cost across all steps.
        success: Whether chain completed successfully.
        error: Error message if failed.
    """

    query: str
    steps: list[ChainStepResult] = field(default_factory=list)
    final_context: ChainContext | None = None
    total_latency_ms: int = 0
    total_cost_usd: float = 0.0
    success: bool = True
    error: str | None = None

    @property
    def step_count(self) -> int:
        """Get number of executed steps."""
        return len(self.steps)

    @property
    def final_documents(self) -> list[Document]:
        """Get documents from final step."""
        if self.steps:
            return self.steps[-1].result.documents
        return []

    def get_step(self, name: str) -> ChainStepResult | None:
        """Get a specific step result by name."""
        for step in self.steps:
            if step.step_name == name:
                return step
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "query": self.query,
            "success": self.success,
            "total_latency_ms": self.total_latency_ms,
            "total_cost_usd": self.total_cost_usd,
            "step_count": self.step_count,
            "steps": [
                {
                    "name": s.step_name,
                    "strategy": s.strategy_name,
                    "duration_ms": s.duration_ms,
                    "document_count": s.output_document_count,
                }
                for s in self.steps
            ],
            "error": self.error,
        }


# =============================================================================
# Chain Executor Configuration
# =============================================================================


class ChainExecutorConfig(BaseModel):
    """
    Configuration for chain executor.

    Attributes:
        continue_on_error: Continue to next step if one fails.
        default_timeout_seconds: Default timeout per step.
        enable_context_passing: Pass documents between steps.
    """

    model_config = ConfigDict(frozen=True)

    continue_on_error: bool = Field(default=False, description="Continue on step error")
    default_timeout_seconds: float = Field(default=60.0, ge=1.0, description="Timeout per step")
    enable_context_passing: bool = Field(default=True, description="Pass context between steps")


# =============================================================================
# Chain Executor
# =============================================================================


class ChainExecutor:
    """
    Execute strategies sequentially in a chain.

    Passes ChainContext between steps, accumulating results
    and costs as the chain progresses.

    Example:
        >>> executor = ChainExecutor()
        >>> chain = [
        ...     ChainStep(strategy="standard", name="search"),
        ...     ChainStep(strategy="reranking", name="rerank"),
        ... ]
        >>> result = await executor.execute_chain(chain, "test query")
    """

    def __init__(
        self,
        strategy_executor: StrategyExecutor | None = None,
        config: ChainExecutorConfig | None = None,
    ) -> None:
        """
        Initialize chain executor.

        Args:
            strategy_executor: Strategy executor to use.
            config: Chain executor configuration.
        """
        self._executor = strategy_executor or StrategyExecutor()
        self._config = config or ChainExecutorConfig()

    async def execute_chain(
        self,
        steps: list[ChainStep],
        query: str,
        initial_context: ChainContext | None = None,
        chain_config: ChainConfig | None = None,
    ) -> ChainResult:
        """
        Execute a chain of strategies sequentially.

        Args:
            steps: List of chain steps to execute.
            query: Initial query.
            initial_context: Optional initial context.
            chain_config: Optional chain configuration.

        Returns:
            ChainResult with all step results.

        Raises:
            ChainConfigurationError: If chain is invalid.
            ChainExecutionError: If chain fails and continue_on_error is False.
        """
        if not steps:
            raise ChainConfigurationError(
                "Chain must have at least one step",
            )

        # Initialize
        start_time = time.time()
        context = initial_context or ChainContext.create(query=query)
        result = ChainResult(query=query)

        # Execute each step
        for i, step in enumerate(steps):
            step_start = time.time()
            step_name = f"{step.strategy}_{i}"

            try:
                step_result = await self._execute_step(
                    step=step,
                    context=context,
                    step_index=i,
                    step_name=step_name,
                )

                # Update context with step result
                context = context.with_step_result(step_result.result)

                # Record step
                step_result.duration_ms = int((time.time() - step_start) * 1000)
                result.steps.append(step_result)
                result.total_cost_usd += step_result.result.cost_usd

                logger.info(
                    f"Chain step '{step_name}' completed: "
                    f"{step_result.output_document_count} documents, "
                    f"{step_result.duration_ms}ms"
                )

            except Exception as e:
                logger.warning(f"Chain step '{step_name}' failed: {e}")

                # Try fallback strategy if configured
                if step.fallback_strategy:
                    fallback_result = await self._try_fallback(
                        step=step,
                        context=context,
                        step_index=i,
                        original_error=e,
                    )
                    if fallback_result:
                        context = context.with_step_result(fallback_result.result)
                        fallback_result.duration_ms = int((time.time() - step_start) * 1000)
                        result.steps.append(fallback_result)
                        result.total_cost_usd += fallback_result.result.cost_usd
                        continue

                # No fallback or fallback failed
                should_continue = step.continue_on_error or self._config.continue_on_error

                if not should_continue:
                    result.success = False
                    result.error = f"Step '{step_name}' failed: {e}"
                    break
                else:
                    # Record error and continue
                    context = context.with_error(str(e))
                    result.steps.append(
                        ChainStepResult(
                            step_name=step_name,
                            strategy_name=step.strategy,
                            result=ExecutionResult(
                                documents=[],
                                query=query,
                                strategy_name=step.strategy,
                                latency_ms=int((time.time() - step_start) * 1000),
                                cost_usd=0.0,
                            ),
                            duration_ms=int((time.time() - step_start) * 1000),
                        )
                    )

        # Finalize result
        result.total_latency_ms = int((time.time() - start_time) * 1000)
        result.final_context = context

        return result

    async def _execute_step(
        self,
        step: ChainStep,
        context: ChainContext,
        step_index: int,
        step_name: str,
    ) -> ChainStepResult:
        """Execute a single chain step."""
        # Note: ChainContext stores summaries, not full documents
        # Full document passing between steps requires custom implementation
        input_doc_count = 0
        if context.intermediate_results:
            input_doc_count = context.intermediate_results[-1].get("document_count", 0)

        # Execute strategy (preserve step config; merge metadata)
        base = step.config if step.config else StrategyConfig()
        meta = dict(base.metadata) if base.metadata else {}
        meta.update({"chain_step": step_index, "step_name": step_name})
        strategy_config = base.model_copy(update={"metadata": meta})

        result = await self._executor.execute(
            strategy_name=step.strategy,
            query=context.query,
            config=strategy_config,
            timeout=self._config.default_timeout_seconds,
            metadata={"step": step_name},
        )

        return ChainStepResult(
            step_name=step_name,
            strategy_name=step.strategy,
            result=result,
            input_document_count=input_doc_count,
            output_document_count=len(result.documents),
        )

    async def _try_fallback(
        self,
        step: ChainStep,
        context: ChainContext,
        step_index: int,
        original_error: Exception,
    ) -> ChainStepResult | None:
        """
        Try executing fallback strategy.

        Args:
            step: The failed chain step.
            context: Current chain context.
            step_index: Index of the step.
            original_error: The error that caused the fallback.

        Returns:
            ChainStepResult if fallback succeeds, None otherwise.
        """
        if not step.fallback_strategy:
            return None

        fallback_name = f"{step.fallback_strategy}_{step_index}_fallback"
        logger.info(f"Attempting fallback strategy: {step.fallback_strategy}")

        try:
            # Create a temporary step for the fallback
            fallback_step = ChainStep(strategy=step.fallback_strategy)

            result = await self._execute_step(
                step=fallback_step,
                context=context,
                step_index=step_index,
                step_name=fallback_name,
            )

            logger.info(
                f"Fallback '{step.fallback_strategy}' succeeded: "
                f"{result.output_document_count} documents"
            )

            return result

        except Exception as fallback_error:
            logger.warning(
                f"Fallback strategy '{step.fallback_strategy}' also failed: "
                f"{fallback_error}"
            )
            return None

    def execute_chain_sync(
        self,
        steps: list[ChainStep],
        query: str,
    ) -> ChainResult:
        """
        Execute chain synchronously.

        Args:
            steps: List of chain steps.
            query: Initial query.

        Returns:
            ChainResult.
        """
        return asyncio.run(self.execute_chain(steps, query))


# =============================================================================
# Convenience Functions
# =============================================================================


async def execute_chain(
    steps: list[ChainStep],
    query: str,
    continue_on_error: bool = False,
) -> ChainResult:
    """
    Execute a chain of strategies (convenience function).

    Args:
        steps: List of chain steps.
        query: Initial query.
        continue_on_error: Whether to continue on step failure.

    Returns:
        ChainResult.
    """
    config = ChainExecutorConfig(continue_on_error=continue_on_error)
    executor = ChainExecutor(config=config)
    return await executor.execute_chain(steps, query)
