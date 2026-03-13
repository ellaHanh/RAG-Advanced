"""
RAGAS-based RAG generation evaluation.

Builds a HuggingFace Dataset from (question, contexts, answer, ground_truth) samples,
runs ragas.evaluate() with configurable metrics, and returns structured scores.
Used by the evaluation pipeline and by the POST /evaluate/generation API.

Key features:
- Input: list of dicts with question, contexts (list[str]), answer, ground_truth.
- Output: overall scores dict and optional per-sample scores.
- Default metrics: faithfulness, answer_relevancy, context_precision.

Usage:
    from evaluation.ragas_eval import evaluate_generation, RagasEvaluationResult

    samples = [
        {
            "question": "What is RAG?",
            "contexts": ["RAG combines retrieval and generation..."],
            "answer": "RAG is retrieval-augmented generation.",
            "ground_truth": "RAG augments LLMs with retrieved knowledge.",
        },
    ]
    result = evaluate_generation(samples)
    print(result.scores)
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Dedicated thread pool for running async embedder from sync context (BGE-M3).
# Used when RAGAS calls our adapter from inside an already-running event loop.
_embedder_executor: ThreadPoolExecutor | None = None


def _get_embedder_executor() -> ThreadPoolExecutor:
    global _embedder_executor
    if _embedder_executor is None:
        _embedder_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ragas_bge_embedder")
    return _embedder_executor


def _run_async_in_thread(coro: Any) -> Any:
    """Run a coroutine in a dedicated thread with its own event loop.
    Safe when the caller is already inside a running event loop (e.g. RAGAS executor).
    """
    def _run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()
    return _get_embedder_executor().submit(_run).result()


def _get_embedding_backend() -> str:
    """Return embedding backend: openai or bge-m3 (same as strategies.utils.embedder)."""
    return (os.getenv("EMBEDDING_BACKEND") or "openai").strip().lower()


class _EmbeddingUsageTracker:
    """Wraps an embeddings object and records request_count and texts_embedded for monitoring."""

    def __init__(self, inner: Any, model_name: str) -> None:
        self._inner = inner
        self._model_name = model_name
        self._request_count = 0
        self._texts_embedded = 0
        self._lock = threading.Lock()

    def embed_query(self, text: str) -> list[float]:
        out = self._inner.embed_query(text or " ")
        with self._lock:
            self._request_count += 1
            self._texts_embedded += 1
        return out

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        out = self._inner.embed_documents(texts)
        with self._lock:
            self._request_count += 1
            self._texts_embedded += len(texts)
        return out

    def get_usage(self) -> dict[str, Any]:
        with self._lock:
            return {
                "request_count": self._request_count,
                "texts_embedded": self._texts_embedded,
                "total_tokens": None,
                "model": self._model_name,
            }


def _get_ragas_embeddings() -> Any:
    """
    Build a LangChain-compatible embeddings object matching EMBEDDING_BACKEND,
    wrapped in _EmbeddingUsageTracker for usage monitoring.
    """
    backend = _get_embedding_backend()
    if backend == "bge-m3":
        from strategies.utils.embedder import embed_documents as aembed_documents
        from strategies.utils.embedder import embed_query as aembed_query

        class _BGE_M3LangChainAdapter:
            """Sync LangChain-style adapter for project BGE-M3 embedder."""

            def embed_query(self, text: str) -> list[float]:
                return _run_async_in_thread(aembed_query(text or " "))

            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                if not texts:
                    return []
                return _run_async_in_thread(aembed_documents(texts))

        inner = _BGE_M3LangChainAdapter()
        return _EmbeddingUsageTracker(inner, model_name="bge-m3")
    # openai
    from strategies.utils.embedder import get_embedding_dimensions

    from langchain_openai import OpenAIEmbeddings

    model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    dimensions = get_embedding_dimensions()
    inner = OpenAIEmbeddings(model=model, dimensions=dimensions)
    return _EmbeddingUsageTracker(inner, model_name=model)

def _get_ragas_llm() -> Any:
    """Return a LangChain LLM for RAGAS (faithfulness, context_precision), wrapped for token usage tracking."""
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        return None
    model = os.getenv("RAGAS_LLM_MODEL") or os.getenv("GENERATION_MODEL") or "gpt-4o-mini"
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    llm = ChatOpenAI(model=model, api_key=api_key)
    try:
        from ragas.llms import LangchainLLMWrapper, TokenUsageParser

        class _OpenAIUsageParser(TokenUsageParser):
            """Parse OpenAI/LangChain token usage from response for RAGAS cost tracking."""

            def parse(self, data: dict) -> dict:
                usage = data.get("usage") or {}
                if not usage and isinstance(data.get("response_metadata"), dict):
                    usage = (
                        data["response_metadata"].get("token_usage")
                        or data["response_metadata"].get("usage")
                        or {}
                    )
                prompt_tokens = usage.get("prompt_tokens") or usage.get("input_tokens", 0)
                completion_tokens = usage.get("completion_tokens") or usage.get("output_tokens", 0)
                total_tokens = usage.get("total_tokens", 0) or (prompt_tokens + completion_tokens)
                return {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                }

        return LangchainLLMWrapper(llm, token_usage_parser=_OpenAIUsageParser())
    except ImportError:
        return llm


# Type for a single RAGAS sample (matches ragas default column names)
RagasSample = dict[str, Any]  # question, contexts, answer, ground_truth


@dataclass
class RagasEvaluationResult:
    """
    Result of RAGAS generation evaluation.

    Attributes:
        scores: Dict of metric name -> float (overall score).
        per_sample: Optional list of per-row score dicts.
        llm_usage: Optional LLM token/cost usage from RAGAS (when token_usage_parser used).
        embedding_usage: Optional embedding request/token usage from RAGAS (when tracker used).
    """

    scores: dict[str, float] = field(default_factory=dict)
    per_sample: list[dict[str, Any]] | None = field(default=None)
    llm_usage: dict[str, Any] | None = field(default=None)
    embedding_usage: dict[str, Any] | None = field(default=None)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors; returns value in [-1, 1]."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a <= 0 or norm_b <= 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _compute_answer_relevancy_embedding_only(
    dataset: Any, embeddings: Any
) -> tuple[float, list[float]]:
    """
    Compute embedding-only answer relevancy per row: cosine_sim(embed(question), embed(answer)),
    rescaled to [0,1]. Returns (mean_score, list of per-row scores).
    """
    scores_list: list[float] = []
    for i in range(len(dataset)):
        q = (dataset["question"][i] or "").strip()
        a = (dataset["answer"][i] or "").strip()
        if not q or not a:
            scores_list.append(0.0)
            continue
        eq = embeddings.embed_query(q)
        ea = embeddings.embed_query(a)
        sim = _cosine_similarity(eq, ea)
        scores_list.append((sim + 1.0) / 2.0)
    mean_score = sum(scores_list) / len(scores_list) if scores_list else 0.0
    return mean_score, scores_list


def _get_default_metrics() -> list[Any]:
    """Return default RAGAS metrics: context_precision and faithfulness (no LLM-based answer_relevancy)."""
    try:
        from ragas.metrics import (
            context_precision,
            faithfulness,
        )
        return [context_precision, faithfulness]
    except ImportError as e:
        logger.warning("ragas.metrics import failed: %s", e)
        return []


def _samples_to_dataset(samples: list[RagasSample]) -> Any:
    """Build HuggingFace Dataset from list of sample dicts."""
    try:
        from datasets import Dataset
    except ImportError as err:
        raise ImportError(
            "datasets package is required for RAGAS evaluation. "
            "Install with: pip install datasets"
        ) from err
    if not samples:
        raise ValueError("samples must be non-empty")
    required = {"question", "contexts", "answer", "ground_truth"}
    for i, s in enumerate(samples):
        missing = required - set(s.keys())
        if missing:
            raise ValueError(
                f"Sample {i} missing keys: {missing}. "
                "Each sample must have question, contexts, answer, ground_truth."
            )
        if not isinstance(s["contexts"], list):
            raise ValueError(f"Sample {i}: 'contexts' must be a list of strings")
    data = {
        "question": [s["question"] for s in samples],
        "contexts": [s["contexts"] for s in samples],
        "answer": [s["answer"] for s in samples],
        "ground_truth": [s["ground_truth"] for s in samples],
    }
    return Dataset.from_dict(data)


def evaluate_generation(
    samples: list[RagasSample],
    metrics: list[Any] | None = None,
    *,
    show_progress: bool = True,
    raise_exceptions: bool = False,
) -> RagasEvaluationResult:
    """
    Run RAGAS evaluation on RAG generation samples.

    Args:
        samples: List of dicts with keys question, contexts (list[str]),
            answer (model-generated), ground_truth (gold answer).
        metrics: List of ragas Metric instances; if None, uses default
            (context_precision, answer_relevancy, faithfulness).
        show_progress: Whether to show progress bar.
        raise_exceptions: If True, raise on metric failure; else use nan for failed rows.

    Returns:
        RagasEvaluationResult with scores dict and optional per_sample.

    Raises:
        ImportError: If ragas or datasets not installed.
        ValueError: If samples empty or missing required keys.
    """
    try:
        from ragas import evaluate
    except ImportError as err:
        raise ImportError(
            "ragas package is required for RAG generation evaluation. "
            "Install with: pip install ragas"
        ) from err

    dataset = _samples_to_dataset(samples)
    embeddings = _get_ragas_embeddings()
    metrics_list = metrics if metrics is not None else _get_default_metrics()
    if not metrics_list:
        logger.warning("No RAGAS metrics available; returning empty scores")
        return RagasEvaluationResult(scores={})

    llm = _get_ragas_llm()
    token_usage_parser = None
    try:
        from ragas.cost import get_token_usage_for_openai
        token_usage_parser = get_token_usage_for_openai
    except ImportError:
        pass

    eval_kw: dict[str, Any] = {
        "dataset": dataset,
        "metrics": metrics_list,
        "embeddings": embeddings,
        "show_progress": show_progress,
        "raise_exceptions": raise_exceptions,
    }
    if llm is not None:
        eval_kw["llm"] = llm
    if token_usage_parser is not None:
        eval_kw["token_usage_parser"] = token_usage_parser

    result = evaluate(**eval_kw)

    scores: dict[str, float] = {}
    per_sample: list[dict[str, Any]] | None = None
    if result is not None:
        if hasattr(result, "to_pandas"):
            df = result.to_pandas()
            if df is not None and not df.empty:
                per_sample = df.to_dict("records")
                for col in df.columns:
                    if col in ("question", "contexts", "answer", "ground_truth"):
                        continue
                    try:
                        scores[col] = float(df[col].mean())
                    except (TypeError, ValueError):
                        pass
        elif isinstance(result, dict):
            scores = {k: float(v) for k, v in result.items() if isinstance(v, (int, float))}
        elif hasattr(result, "__iter__") and not isinstance(result, (str, bytes)):
            for item in result:
                if hasattr(item, "items"):
                    scores.update({k: float(v) for k, v in item.items() if isinstance(v, (int, float))})
                    break

    # Embedding-only answer_relevancy (no LLM)
    ar_mean, ar_list = _compute_answer_relevancy_embedding_only(dataset, embeddings)
    scores["answer_relevancy"] = float(ar_mean)
    if per_sample is not None and len(ar_list) == len(per_sample):
        for i, row in enumerate(per_sample):
            row["answer_relevancy"] = float(ar_list[i])
    elif per_sample is not None:
        for i in range(len(per_sample)):
            per_sample[i]["answer_relevancy"] = float(ar_list[i]) if i < len(ar_list) else 0.0

    llm_usage: dict[str, Any] | None = None
    if result is not None and hasattr(result, "total_tokens"):
        try:
            tu = result.total_tokens()
            if tu is not None:
                input_tokens = getattr(tu, "input_tokens", None) or getattr(tu, "prompt_tokens", 0)
                output_tokens = getattr(tu, "output_tokens", None) or getattr(tu, "completion_tokens", 0)
                total_tokens = getattr(tu, "total_tokens", 0) or (input_tokens + output_tokens)
                model_name = (getattr(tu, "model", "") or "").strip()
                llm_usage = {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens,
                    "model": model_name,
                }
                cost_usd: float | None = None
                try:
                    from orchestration.pricing import get_pricing_provider_sync
                    provider = get_pricing_provider_sync()
                    if provider and model_name:
                        cost_usd = provider.calculate_cost(
                            model_name, input_tokens=input_tokens, output_tokens=output_tokens
                        )
                    elif provider:
                        pricing = provider.get_model_pricing("gpt-4o-mini")
                        cost_usd = pricing.calculate_cost(input_tokens=input_tokens, output_tokens=output_tokens)
                except Exception:
                    pass
                if cost_usd is not None:
                    llm_usage["cost_usd"] = round(cost_usd, 6)
                else:
                    llm_usage["cost_usd"] = None
        except Exception:
            pass

    embedding_usage: dict[str, Any] | None = None
    if hasattr(embeddings, "get_usage"):
        try:
            embedding_usage = embeddings.get_usage()
        except Exception:
            pass

    return RagasEvaluationResult(
        scores=scores,
        per_sample=per_sample,
        llm_usage=llm_usage,
        embedding_usage=embedding_usage,
    )
