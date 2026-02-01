"""
Unit tests for LLM-Assisted Ground Truth Generation.

Tests cover:
- generate_ground_truth_for_query with mock OpenAI
- enrich_dataset_with_llm with mock candidate provider
- Output compatibility with DatasetQuery (relevant_doc_ids, relevance_scores 0/1/2)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from evaluation.datasets import Dataset, DatasetManager, DatasetQuery
from evaluation.ground_truth_llm import (
    CandidateDoc,
    enrich_dataset_with_llm,
    generate_ground_truth_for_query,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_candidates() -> list[tuple[str, str]]:
    """Sample (doc_id, snippet) candidates."""
    return [
        ("doc1", "RAG is retrieval-augmented generation. It combines retrieval and LLMs."),
        ("doc2", "The weather today is sunny."),
        ("doc3", "Embeddings are vector representations of text for similarity search."),
    ]


@pytest.fixture
def mock_openai_response():
    """Mock OpenAI chat completion returning valid JSON."""

    def _make_response(relevant_ids: list[str], relevance_scores: dict[str, int] | None = None):
        import json

        payload = {"relevant_ids": relevant_ids}
        if relevance_scores is not None:
            payload["relevance_scores"] = relevance_scores
        content = json.dumps(payload)

        mock_choice = MagicMock()
        mock_choice.message.content = content
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        return mock_response

    return _make_response


# =============================================================================
# Test: generate_ground_truth_for_query
# =============================================================================


class TestGenerateGroundTruthForQuery:
    """Tests for generate_ground_truth_for_query."""

    @pytest.mark.asyncio
    async def test_returns_relevant_ids_and_scores(
        self,
        sample_candidates: list[tuple[str, str]],
        mock_openai_response,
    ):
        """Test that LLM response is parsed into relevant_ids and relevance_scores."""
        response = mock_openai_response(
            relevant_ids=["doc1", "doc3"],
            relevance_scores={"doc1": 2, "doc3": 1},
        )
        with patch("evaluation.ground_truth_llm.AsyncOpenAI") as mock_client_class:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=response)
            mock_client_class.return_value = mock_client

            relevant_ids, scores = await generate_ground_truth_for_query(
                query="What is RAG?",
                candidates=sample_candidates,
                api_key="test-key",
            )

        assert relevant_ids == ["doc1", "doc3"]
        assert scores == {"doc1": 2, "doc3": 1}

    @pytest.mark.asyncio
    async def test_accepts_candidate_tuples(
        self,
        mock_openai_response,
    ):
        """Test candidates as list of (id, snippet) tuples."""
        response = mock_openai_response(relevant_ids=["a"])
        with patch("evaluation.ground_truth_llm.AsyncOpenAI") as mock_client_class:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=response)
            mock_client_class.return_value = mock_client

            relevant_ids, scores = await generate_ground_truth_for_query(
                query="test",
                candidates=[("a", "snippet a")],
                api_key="test-key",
            )
        assert "a" in relevant_ids

    @pytest.mark.asyncio
    async def test_accepts_candidate_doc_objects(
        self,
        mock_openai_response,
    ):
        """Test candidates as list of CandidateDoc."""
        response = mock_openai_response(relevant_ids=["doc1"])
        with patch("evaluation.ground_truth_llm.AsyncOpenAI") as mock_client_class:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=response)
            mock_client_class.return_value = mock_client

            relevant_ids, _ = await generate_ground_truth_for_query(
                query="test",
                candidates=[CandidateDoc(id="doc1", snippet="content")],
                api_key="test-key",
            )
        assert "doc1" in relevant_ids

    @pytest.mark.asyncio
    async def test_empty_candidates_returns_empty(
        self,
    ):
        """Test that empty candidates returns ([], {})."""
        relevant_ids, scores = await generate_ground_truth_for_query(
            query="test",
            candidates=[],
            api_key="test-key",
        )
        assert relevant_ids == []
        assert scores == {}

    @pytest.mark.asyncio
    async def test_missing_api_key_raises(
        self,
        sample_candidates,
    ):
        """Test that missing API key raises ValueError."""
        with patch.dict("os.environ", {}, clear=False):
            with pytest.raises(ValueError, match="OPENAI_API_KEY"):
                await generate_ground_truth_for_query(
                    query="test",
                    candidates=sample_candidates,
                )

    @pytest.mark.asyncio
    async def test_default_scores_one_for_relevant_ids(
        self,
        mock_openai_response,
    ):
        """Test that relevant_ids without scores get default score 1."""
        response = mock_openai_response(relevant_ids=["doc1", "doc2"])
        # No relevance_scores in response
        with patch("evaluation.ground_truth_llm.AsyncOpenAI") as mock_client_class:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=response)
            mock_client_class.return_value = mock_client

            _, scores = await generate_ground_truth_for_query(
                query="test",
                candidates=[("doc1", "a"), ("doc2", "b")],
                api_key="test-key",
            )
        assert scores.get("doc1") == 1
        assert scores.get("doc2") == 1


# =============================================================================
# Test: enrich_dataset_with_llm
# =============================================================================


class TestEnrichDatasetWithLlm:
    """Tests for enrich_dataset_with_llm."""

    @pytest.mark.asyncio
    async def test_enrich_adds_ground_truth_to_query(
        self,
        mock_openai_response,
    ):
        """Test that enrich_dataset_with_llm adds relevant_doc_ids to query."""
        dataset = Dataset(
            name="test",
            queries=[
                DatasetQuery(
                    query_id="q1",
                    query="What is RAG?",
                    relevant_doc_ids=[],
                ),
            ],
        )
        response = mock_openai_response(relevant_ids=["d1"], relevance_scores={"d1": 2})

        async def provider(query: str):
            return [("d1", "RAG is retrieval-augmented generation.")]

        with patch("evaluation.ground_truth_llm.AsyncOpenAI") as mock_client_class:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=response)
            mock_client_class.return_value = mock_client

            enriched = await enrich_dataset_with_llm(
                dataset,
                provider,
                only_missing=True,
                api_key="test-key",
            )

        assert len(enriched.queries) == 1
        assert enriched.queries[0].relevant_doc_ids == ["d1"]
        assert enriched.queries[0].relevance_scores == {"d1": 2}
        assert enriched.queries[0].metadata.get("llm_ground_truth") is True

    @pytest.mark.asyncio
    async def test_only_missing_skips_queries_with_ground_truth(
        self,
        mock_openai_response,
    ):
        """Test that only_missing=True skips queries that already have relevant_doc_ids."""
        dataset = Dataset(
            name="test",
            queries=[
                DatasetQuery(
                    query_id="q1",
                    query="What is RAG?",
                    relevant_doc_ids=["existing"],
                ),
            ],
        )

        async def provider(query: str):
            return [("d1", "snippet")]

        enriched = await enrich_dataset_with_llm(
            dataset,
            provider,
            only_missing=True,
            api_key="test-key",
        )
        # LLM should not be called; query unchanged
        assert enriched.queries[0].relevant_doc_ids == ["existing"]
        assert "llm_ground_truth" not in enriched.queries[0].metadata

    @pytest.mark.asyncio
    async def test_dataset_metadata_marks_llm_enriched(
        self,
    ):
        """Test that enriched dataset has llm_enriched in metadata."""
        async def empty_provider(query: str):
            return []

        dataset = Dataset(name="t", queries=[])
        enriched = await enrich_dataset_with_llm(
            dataset,
            empty_provider,
            api_key="test-key",
        )
        assert enriched.metadata.get("llm_enriched") is True
        assert enriched.name == "t_llm_enriched"


# =============================================================================
# Test: DatasetManager.enrich_with_llm
# =============================================================================


class TestDatasetManagerEnrichWithLlm:
    """Tests for DatasetManager.enrich_with_llm."""

    @pytest.mark.asyncio
    async def test_manager_enrich_delegates(
        self,
        mock_openai_response,
    ):
        """Test that DatasetManager.enrich_with_llm delegates to enrich_dataset_with_llm."""
        dataset = Dataset(
            name="test",
            queries=[
                DatasetQuery(query_id="q1", query="What is RAG?", relevant_doc_ids=[]),
            ],
        )
        response = mock_openai_response(relevant_ids=["d1"])

        async def provider(query: str):
            return [("d1", "snippet")]

        with patch("evaluation.ground_truth_llm.AsyncOpenAI") as mock_client_class:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=response)
            mock_client_class.return_value = mock_client

            manager = DatasetManager()
            enriched = await manager.enrich_with_llm(
                dataset,
                provider,
                api_key="test-key",
            )

        assert enriched.name == "test_llm_enriched"
        assert len(enriched.queries) == 1
        assert enriched.queries[0].relevant_doc_ids == ["d1"]
