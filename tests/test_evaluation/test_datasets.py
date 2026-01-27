"""
Unit tests for Dataset Management.

Tests cover:
- Dataset loading (JSON, JSONL)
- Schema validation
- Train/test splits
- Dataset operations (filter, sample, merge)
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from evaluation.datasets import (
    Dataset,
    DatasetConfig,
    DatasetManager,
    DatasetQuery,
    DatasetSplit,
    create_dataset_from_queries,
)
from orchestration.errors import DatasetError, InvalidInputError


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_dataset_data() -> dict:
    """Sample dataset in JSON format."""
    return {
        "name": "test_dataset",
        "description": "Test dataset for unit tests",
        "queries": [
            {
                "query_id": "q1",
                "query": "What is machine learning?",
                "relevant_doc_ids": ["doc1", "doc2"],
                "category": "ml",
            },
            {
                "query_id": "q2",
                "query": "How does RAG work?",
                "relevant_doc_ids": ["doc3"],
                "relevance_scores": {"doc3": 2},
                "category": "rag",
            },
            {
                "query_id": "q3",
                "query": "What are embeddings?",
                "relevant_doc_ids": ["doc4", "doc5"],
                "category": "ml",
            },
            {
                "query_id": "q4",
                "query": "Explain vector search.",
                "relevant_doc_ids": ["doc6"],
                "category": "rag",
            },
        ],
        "metadata": {"version": "1.0"},
    }


@pytest.fixture
def temp_json_file(sample_dataset_data: dict) -> Path:
    """Create a temporary JSON dataset file."""
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        delete=False,
    ) as f:
        json.dump(sample_dataset_data, f)
        return Path(f.name)


@pytest.fixture
def temp_jsonl_file(sample_dataset_data: dict) -> Path:
    """Create a temporary JSONL dataset file."""
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".jsonl",
        delete=False,
    ) as f:
        for query in sample_dataset_data["queries"]:
            f.write(json.dumps(query) + "\n")
        return Path(f.name)


# =============================================================================
# Test: DatasetQuery Model
# =============================================================================


class TestDatasetQuery:
    """Tests for DatasetQuery model."""

    def test_valid_query(self):
        """Test valid query creation."""
        query = DatasetQuery(
            query_id="q1",
            query="Test query",
            relevant_doc_ids=["doc1", "doc2"],
        )

        assert query.query_id == "q1"
        assert query.query == "Test query"
        assert len(query.relevant_doc_ids) == 2

    def test_query_with_relevance_scores(self):
        """Test query with graded relevance."""
        query = DatasetQuery(
            query_id="q1",
            query="Test query",
            relevant_doc_ids=["doc1"],
            relevance_scores={"doc1": 2, "doc2": 1},
        )

        assert query.relevance_scores["doc1"] == 2

    def test_invalid_relevance_score(self):
        """Test invalid relevance score validation."""
        with pytest.raises(ValueError, match="Invalid relevance score"):
            DatasetQuery(
                query_id="q1",
                query="Test",
                relevant_doc_ids=["doc1"],
                relevance_scores={"doc1": 5},  # Invalid
            )

    def test_to_dict(self):
        """Test serialization."""
        query = DatasetQuery(
            query_id="q1",
            query="Test",
            relevant_doc_ids=["doc1"],
            category="test",
        )

        d = query.to_dict()
        assert d["query_id"] == "q1"
        assert d["category"] == "test"


# =============================================================================
# Test: Dataset Model
# =============================================================================


class TestDataset:
    """Tests for Dataset model."""

    def test_size_property(self, sample_dataset_data: dict):
        """Test size property."""
        queries = [DatasetQuery(**q) for q in sample_dataset_data["queries"]]
        dataset = Dataset(queries=queries)

        assert dataset.size == 4
        assert len(dataset) == 4

    def test_categories_property(self, sample_dataset_data: dict):
        """Test categories property."""
        queries = [DatasetQuery(**q) for q in sample_dataset_data["queries"]]
        dataset = Dataset(queries=queries)

        assert dataset.categories == {"ml", "rag"}

    def test_iteration(self, sample_dataset_data: dict):
        """Test dataset iteration."""
        queries = [DatasetQuery(**q) for q in sample_dataset_data["queries"]]
        dataset = Dataset(queries=queries)

        query_ids = [q.query_id for q in dataset]
        assert query_ids == ["q1", "q2", "q3", "q4"]

    def test_indexing(self, sample_dataset_data: dict):
        """Test dataset indexing."""
        queries = [DatasetQuery(**q) for q in sample_dataset_data["queries"]]
        dataset = Dataset(queries=queries)

        assert dataset[0].query_id == "q1"
        assert dataset[2].query_id == "q3"

    def test_filter_by_category(self, sample_dataset_data: dict):
        """Test category filtering."""
        queries = [DatasetQuery(**q) for q in sample_dataset_data["queries"]]
        dataset = Dataset(name="test", queries=queries)

        ml_dataset = dataset.filter_by_category("ml")

        assert ml_dataset.size == 2
        assert all(q.category == "ml" for q in ml_dataset)

    def test_sample(self, sample_dataset_data: dict):
        """Test random sampling."""
        queries = [DatasetQuery(**q) for q in sample_dataset_data["queries"]]
        dataset = Dataset(queries=queries)

        sampled = dataset.sample(2, seed=42)

        assert sampled.size == 2

    def test_sample_larger_than_dataset(self, sample_dataset_data: dict):
        """Test sampling more than available."""
        queries = [DatasetQuery(**q) for q in sample_dataset_data["queries"]]
        dataset = Dataset(queries=queries)

        sampled = dataset.sample(100)

        assert sampled.size == dataset.size


# =============================================================================
# Test: DatasetManager - Loading
# =============================================================================


class TestDatasetManagerLoading:
    """Tests for dataset loading."""

    @pytest.mark.asyncio
    async def test_load_json(self, temp_json_file: Path):
        """Test loading JSON dataset."""
        manager = DatasetManager()
        dataset = await manager.load(temp_json_file)

        assert dataset.name == "test_dataset"
        assert dataset.size == 4
        assert dataset.metadata.get("version") == "1.0"

    @pytest.mark.asyncio
    async def test_load_jsonl(self, temp_jsonl_file: Path):
        """Test loading JSONL dataset."""
        manager = DatasetManager()
        dataset = await manager.load(temp_jsonl_file)

        assert dataset.size == 4

    def test_load_sync(self, temp_json_file: Path):
        """Test synchronous loading."""
        manager = DatasetManager()
        dataset = manager.load_sync(temp_json_file)

        assert dataset.size == 4

    @pytest.mark.asyncio
    async def test_load_nonexistent_file(self):
        """Test loading nonexistent file."""
        manager = DatasetManager()

        with pytest.raises(DatasetError, match="not found"):
            await manager.load("nonexistent.json")

    @pytest.mark.asyncio
    async def test_load_invalid_json(self):
        """Test loading invalid JSON."""
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            delete=False,
        ) as f:
            f.write("not valid json {")
            path = Path(f.name)

        manager = DatasetManager()

        with pytest.raises(DatasetError, match="Invalid JSON"):
            await manager.load(path)

    @pytest.mark.asyncio
    async def test_load_with_base_path(self, sample_dataset_data: dict):
        """Test loading with base path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "data.json"
            with open(file_path, "w") as f:
                json.dump(sample_dataset_data, f)

            manager = DatasetManager(base_path=tmpdir)
            dataset = await manager.load("data.json")

            assert dataset.size == 4


# =============================================================================
# Test: DatasetManager - Validation
# =============================================================================


class TestDatasetManagerValidation:
    """Tests for dataset validation."""

    @pytest.mark.asyncio
    async def test_validation_missing_query_id(self):
        """Test validation catches missing query_id."""
        data = {
            "queries": [
                {"query": "Test", "relevant_doc_ids": ["doc1"]},  # Missing query_id
            ]
        }

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            delete=False,
        ) as f:
            json.dump(data, f)
            path = Path(f.name)

        manager = DatasetManager()

        with pytest.raises(DatasetError, match="missing query_id"):
            await manager.load(path)

    @pytest.mark.asyncio
    async def test_validation_missing_ground_truth(self):
        """Test validation requires ground truth by default."""
        data = {
            "queries": [
                {"query_id": "q1", "query": "Test"},  # Missing relevant_doc_ids
            ]
        }

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            delete=False,
        ) as f:
            json.dump(data, f)
            path = Path(f.name)

        manager = DatasetManager()

        with pytest.raises(DatasetError, match="missing relevant_doc_ids"):
            await manager.load(path)

    @pytest.mark.asyncio
    async def test_validation_ground_truth_not_required(self):
        """Test loading without ground truth when not required."""
        data = {
            "queries": [
                {"query_id": "q1", "query": "Test"},
            ]
        }

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            delete=False,
        ) as f:
            json.dump(data, f)
            path = Path(f.name)

        manager = DatasetManager()
        config = DatasetConfig(require_ground_truth=False)
        dataset = await manager.load(path, config)

        assert dataset.size == 1

    @pytest.mark.asyncio
    async def test_validation_duplicate_query_ids(self):
        """Test validation catches duplicate query_ids."""
        data = {
            "queries": [
                {"query_id": "q1", "query": "Test1", "relevant_doc_ids": ["doc1"]},
                {"query_id": "q1", "query": "Test2", "relevant_doc_ids": ["doc2"]},
            ]
        }

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            delete=False,
        ) as f:
            json.dump(data, f)
            path = Path(f.name)

        manager = DatasetManager()

        with pytest.raises(DatasetError, match="Duplicate query_id"):
            await manager.load(path)


# =============================================================================
# Test: DatasetManager - Splits
# =============================================================================


class TestDatasetManagerSplits:
    """Tests for train/test splits."""

    def test_random_split(self, sample_dataset_data: dict):
        """Test basic random split."""
        queries = [DatasetQuery(**q) for q in sample_dataset_data["queries"]]
        dataset = Dataset(name="test", queries=queries)

        manager = DatasetManager()
        split = manager.split(dataset, train_ratio=0.5, seed=42)

        assert isinstance(split, DatasetSplit)
        assert split.train.size + split.test.size == dataset.size
        assert split.train_ratio == 0.5

    def test_stratified_split(self, sample_dataset_data: dict):
        """Test stratified split by category."""
        queries = [DatasetQuery(**q) for q in sample_dataset_data["queries"]]
        dataset = Dataset(name="test", queries=queries)

        manager = DatasetManager()
        split = manager.split(
            dataset,
            train_ratio=0.5,
            stratified=True,
            seed=42,
        )

        # Both splits should have both categories
        train_cats = {q.category for q in split.train}
        test_cats = {q.category for q in split.test}

        assert len(train_cats) == 2
        assert len(test_cats) >= 1  # May have 1 or 2 depending on split

    def test_split_reproducible(self, sample_dataset_data: dict):
        """Test that splits are reproducible with seed."""
        queries = [DatasetQuery(**q) for q in sample_dataset_data["queries"]]
        dataset = Dataset(queries=queries)

        manager = DatasetManager()
        split1 = manager.split(dataset, train_ratio=0.5, seed=42)
        split2 = manager.split(dataset, train_ratio=0.5, seed=42)

        ids1 = [q.query_id for q in split1.train]
        ids2 = [q.query_id for q in split2.train]

        assert ids1 == ids2

    def test_split_invalid_ratio(self, sample_dataset_data: dict):
        """Test split with invalid ratio."""
        queries = [DatasetQuery(**q) for q in sample_dataset_data["queries"]]
        dataset = Dataset(queries=queries)

        manager = DatasetManager()

        with pytest.raises(InvalidInputError):
            manager.split(dataset, train_ratio=0.0)

        with pytest.raises(InvalidInputError):
            manager.split(dataset, train_ratio=1.0)


# =============================================================================
# Test: DatasetManager - Operations
# =============================================================================


class TestDatasetManagerOperations:
    """Tests for dataset operations."""

    def test_merge_datasets(self, sample_dataset_data: dict):
        """Test merging multiple datasets."""
        queries1 = [DatasetQuery(**q) for q in sample_dataset_data["queries"][:2]]
        queries2 = [DatasetQuery(**q) for q in sample_dataset_data["queries"][2:]]

        dataset1 = Dataset(name="d1", queries=queries1)
        dataset2 = Dataset(name="d2", queries=queries2)

        manager = DatasetManager()
        merged = manager.merge(dataset1, dataset2)

        assert merged.size == 4
        assert "d1_d2" in merged.name

    def test_merge_with_duplicates(self, sample_dataset_data: dict):
        """Test merging handles duplicate query_ids."""
        queries = [DatasetQuery(**q) for q in sample_dataset_data["queries"]]

        dataset1 = Dataset(queries=queries)
        dataset2 = Dataset(queries=queries)  # Same queries

        manager = DatasetManager()
        merged = manager.merge(dataset1, dataset2)

        # Should deduplicate
        assert merged.size == 4

    @pytest.mark.asyncio
    async def test_save_json(self, sample_dataset_data: dict):
        """Test saving dataset as JSON."""
        queries = [DatasetQuery(**q) for q in sample_dataset_data["queries"]]
        dataset = Dataset(name="test", queries=queries)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "output.json"
            manager = DatasetManager()
            await manager.save(dataset, path)

            # Reload and verify
            loaded = await manager.load(path)
            assert loaded.size == dataset.size


# =============================================================================
# Test: Convenience Functions
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_create_dataset_from_queries(self):
        """Test creating dataset from query dicts."""
        queries = [
            {"query": "What is ML?", "relevant_doc_ids": ["doc1"]},
            {"query": "How does RAG work?", "relevant_doc_ids": ["doc2"]},
        ]

        dataset = create_dataset_from_queries(queries, name="custom")

        assert dataset.size == 2
        assert dataset.name == "custom"
        # Auto-generated query_ids
        assert dataset[0].query_id == "q0"
        assert dataset[1].query_id == "q1"

    def test_create_dataset_with_query_ids(self):
        """Test creating dataset with existing query_ids."""
        queries = [
            {"query_id": "custom_1", "query": "Test", "relevant_doc_ids": ["doc1"]},
        ]

        dataset = create_dataset_from_queries(queries)

        assert dataset[0].query_id == "custom_1"
