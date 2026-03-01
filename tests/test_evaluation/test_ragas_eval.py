"""
Unit tests for RAGAS generation evaluation (evaluation/ragas_eval.py).

Tests dataset building and result handling. Full RAGAS evaluate() is optional
(requires ragas/datasets and API); tests use importorskip when calling evaluate_generation.
"""

from __future__ import annotations

import pytest

from evaluation.ragas_eval import (
    RagasEvaluationResult,
    _EmbeddingUsageTracker,
    _compute_answer_relevancy_embedding_only,
    _cosine_similarity,
    _get_embedding_backend,
    _get_ragas_embeddings,
    _samples_to_dataset,
    evaluate_generation,
)


def test_ragas_eval_result_dataclass() -> None:
    """RagasEvaluationResult has scores and optional per_sample."""
    r = RagasEvaluationResult(scores={"faithfulness": 0.9}, per_sample=None)
    assert r.scores == {"faithfulness": 0.9}
    assert r.per_sample is None

    r2 = RagasEvaluationResult(scores={}, per_sample=[{"faithfulness": 0.8}])
    assert r2.per_sample == [{"faithfulness": 0.8}]


def test_ragas_eval_result_llm_and_embedding_usage() -> None:
    """RagasEvaluationResult accepts optional llm_usage and embedding_usage."""
    r = RagasEvaluationResult(
        scores={"faithfulness": 0.9},
        per_sample=None,
        llm_usage={
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
            "model": "gpt-4o-mini",
            "cost_usd": 0.001,
        },
        embedding_usage={
            "request_count": 10,
            "texts_embedded": 42,
            "total_tokens": None,
            "model": "openai",
        },
    )
    assert r.llm_usage is not None
    assert r.llm_usage["total_tokens"] == 150
    assert r.embedding_usage is not None
    assert r.embedding_usage["request_count"] == 10
    assert r.embedding_usage["texts_embedded"] == 42

    r_default = RagasEvaluationResult(scores={})
    assert r_default.llm_usage is None
    assert r_default.embedding_usage is None


def test_samples_to_dataset_valid() -> None:
    """_samples_to_dataset builds Dataset from valid samples."""
    pytest.importorskip("datasets")
    samples = [
        {
            "question": "What is RAG?",
            "contexts": ["RAG combines retrieval and generation."],
            "answer": "RAG is retrieval-augmented generation.",
            "ground_truth": "RAG augments LLMs with retrieved knowledge.",
        },
    ]
    ds = _samples_to_dataset(samples)
    assert ds is not None
    assert len(ds) == 1
    assert ds["question"][0] == "What is RAG?"
    assert ds["contexts"][0] == ["RAG combines retrieval and generation."]
    assert ds["answer"][0] == "RAG is retrieval-augmented generation."
    assert ds["ground_truth"][0] == "RAG augments LLMs with retrieved knowledge."


def test_samples_to_dataset_empty_raises() -> None:
    """_samples_to_dataset raises when samples is empty."""
    pytest.importorskip("datasets")
    with pytest.raises(ValueError, match="non-empty"):
        _samples_to_dataset([])


def test_samples_to_dataset_missing_keys_raises() -> None:
    """_samples_to_dataset raises when a sample is missing required keys."""
    pytest.importorskip("datasets")
    samples = [
        {
            "question": "Q?",
            "contexts": ["C"],
            "answer": "A",
            # missing "ground_truth"
        },
    ]
    with pytest.raises(ValueError, match="missing keys"):
        _samples_to_dataset(samples)


def test_get_embedding_backend_default() -> None:
    """_get_embedding_backend returns openai when EMBEDDING_BACKEND unset."""
    import os
    orig = os.environ.pop("EMBEDDING_BACKEND", None)
    try:
        backend = _get_embedding_backend()
        assert backend == "openai"
    finally:
        if orig is not None:
            os.environ["EMBEDDING_BACKEND"] = orig


def test_get_ragas_embeddings_has_embed_interface() -> None:
    """_get_ragas_embeddings returns an object with embed_query and embed_documents."""
    pytest.importorskip("langchain_openai")
    import os
    orig = os.environ.get("EMBEDDING_BACKEND")
    try:
        os.environ["EMBEDDING_BACKEND"] = "openai"
        emb = _get_ragas_embeddings()
        assert hasattr(emb, "embed_query"), "embeddings must have embed_query"
        assert hasattr(emb, "embed_documents"), "embeddings must have embed_documents"
        assert callable(getattr(emb, "embed_query"))
        assert callable(getattr(emb, "embed_documents"))
    finally:
        if orig is not None:
            os.environ["EMBEDDING_BACKEND"] = orig
        elif "EMBEDDING_BACKEND" in os.environ:
            os.environ.pop("EMBEDDING_BACKEND")


def test_embedding_usage_tracker_counts_requests_and_texts() -> None:
    """_EmbeddingUsageTracker.get_usage() returns request_count and texts_embedded after embed calls."""
    class MockInner:
        def embed_query(self, text: str) -> list[float]:
            return [1.0, 0.0]

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [[1.0, 0.0]] * len(texts) if texts else []

    inner = MockInner()
    emb = _EmbeddingUsageTracker(inner, model_name="test-model")
    u0 = emb.get_usage()
    assert u0["request_count"] == 0
    assert u0["texts_embedded"] == 0
    assert u0["model"] == "test-model"
    emb.embed_query("hello")
    u1 = emb.get_usage()
    assert u1["request_count"] == 1
    assert u1["texts_embedded"] == 1
    emb.embed_documents(["a", "b", "c"])
    u2 = emb.get_usage()
    assert u2["request_count"] == 2
    assert u2["texts_embedded"] == 4


def test_cosine_similarity() -> None:
    """_cosine_similarity returns value in [-1, 1]; parallel vectors 1, opposite -1."""
    assert _cosine_similarity([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]) == pytest.approx(1.0)
    assert _cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)
    assert _cosine_similarity([1.0, 1.0], [1.0, 1.0]) == pytest.approx(1.0)
    assert _cosine_similarity([], []) == 0.0
    assert _cosine_similarity([1.0], [2.0]) == pytest.approx(1.0)
    # orthogonal
    assert _cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_compute_answer_relevancy_embedding_only() -> None:
    """_compute_answer_relevancy_embedding_only returns mean and list in [0, 1]; uses only embeddings."""
    datasets = pytest.importorskip("datasets")
    Dataset = getattr(datasets, "Dataset", None)
    if Dataset is None:
        pytest.skip("datasets.Dataset not available")

    dim = 4
    # Mock embedder: same text -> same vector; we control vectors so similarity is predictable.
    class MockEmb:
        def __init__(self) -> None:
            self._cache: dict[str, list[float]] = {}

        def _vec(self, text: str) -> list[float]:
            if text not in self._cache:
                # deterministic: hash text to a vector
                h = hash(text) % (2 ** 16)
                self._cache[text] = [(h >> i) % 2 for i in range(dim)]
            return self._cache[text]

        def embed_query(self, text: str) -> list[float]:
            return self._vec(text or " ")

    mock = MockEmb()
    # Identical q and a -> similarity 1 -> rescaled (1+1)/2 = 1.0
    ds = Dataset.from_dict({
        "question": ["x", "y"],
        "contexts": [["c1"], ["c2"]],
        "answer": ["x", "y"],
        "ground_truth": ["gt1", "gt2"],
    })
    mean, per = _compute_answer_relevancy_embedding_only(ds, mock)
    assert len(per) == 2
    assert all(0 <= s <= 1 for s in per)
    assert mean == pytest.approx(1.0)
    assert per[0] == pytest.approx(1.0)
    assert per[1] == pytest.approx(1.0)


def test_evaluate_generation_import_error_without_ragas() -> None:
    """evaluate_generation raises or returns empty when ragas not available."""
    # If ragas is installed, this runs and may return real scores or empty
    # If not, we expect ImportError from evaluate_generation when it tries to import ragas
    try:
        result = evaluate_generation(
            [
                {
                    "question": "Q?",
                    "contexts": ["C"],
                    "answer": "A",
                    "ground_truth": "GT",
                },
            ],
            show_progress=False,
        )
        assert isinstance(result, RagasEvaluationResult)
        assert isinstance(result.scores, dict)
    except ImportError as e:
        assert "ragas" in str(e).lower() or "datasets" in str(e).lower()
