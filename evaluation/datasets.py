"""
RAG-Advanced Dataset Management.

Load, validate, and manage test datasets with ground truth for RAG evaluation.
Supports JSON/JSONL formats, train/test splits, and optional LLM-assisted generation.

Usage:
    from evaluation.datasets import DatasetManager, DatasetConfig

    manager = DatasetManager()
    dataset = await manager.load("datasets/sample/basic_queries.json")

    train, test = manager.split(dataset, train_ratio=0.8)
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import aiofiles
from pydantic import BaseModel, ConfigDict, Field, field_validator

from orchestration.errors import DatasetError, InvalidInputError


logger = logging.getLogger(__name__)


# =============================================================================
# Schema Definitions
# =============================================================================


class DatasetQuery(BaseModel):
    """
    A single query in the dataset with ground truth.

    Attributes:
        query_id: Unique identifier for the query.
        query: The query text.
        relevant_doc_ids: List of relevant document IDs.
        relevance_scores: Optional graded relevance (0, 1, or 2).
        category: Optional category for stratified splits.
        metadata: Optional additional metadata.
    """

    model_config = ConfigDict(frozen=True)

    query_id: str = Field(..., min_length=1, description="Query identifier")
    query: str = Field(..., min_length=1, description="Query text")
    relevant_doc_ids: list[str] = Field(
        default_factory=list,
        description="Relevant document IDs",
    )
    relevance_scores: dict[str, int] | None = Field(
        default=None,
        description="Graded relevance scores (0, 1, or 2)",
    )
    category: str | None = Field(default=None, description="Query category")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    @field_validator("relevance_scores")
    @classmethod
    def validate_relevance_scores(
        cls,
        v: dict[str, int] | None,
    ) -> dict[str, int] | None:
        """Validate relevance scores are 0, 1, or 2."""
        if v is not None:
            for doc_id, score in v.items():
                if score not in (0, 1, 2):
                    raise ValueError(f"Invalid relevance score {score} for {doc_id}")
        return v

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "query_id": self.query_id,
            "query": self.query,
            "relevant_doc_ids": self.relevant_doc_ids,
            "relevance_scores": self.relevance_scores,
            "category": self.category,
            "metadata": self.metadata,
        }


class Dataset(BaseModel):
    """
    A collection of queries with ground truth.

    Attributes:
        name: Dataset name.
        description: Dataset description.
        queries: List of dataset queries.
        metadata: Dataset-level metadata.
    """

    model_config = ConfigDict(frozen=False)  # Allow modification for filtering

    name: str = Field(default="unnamed", description="Dataset name")
    description: str = Field(default="", description="Dataset description")
    queries: list[DatasetQuery] = Field(default_factory=list, description="Dataset queries")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Dataset metadata")

    @property
    def size(self) -> int:
        """Number of queries in the dataset."""
        return len(self.queries)

    @property
    def categories(self) -> set[str]:
        """Set of unique categories in the dataset."""
        return {q.category for q in self.queries if q.category is not None}

    def __len__(self) -> int:
        """Return number of queries."""
        return len(self.queries)

    def __iter__(self) -> Iterator[DatasetQuery]:
        """Iterate over queries."""
        return iter(self.queries)

    def __getitem__(self, index: int) -> DatasetQuery:
        """Get query by index."""
        return self.queries[index]

    def filter_by_category(self, category: str) -> "Dataset":
        """Filter dataset by category."""
        filtered_queries = [q for q in self.queries if q.category == category]
        return Dataset(
            name=f"{self.name}_{category}",
            description=f"Filtered by category: {category}",
            queries=filtered_queries,
            metadata={**self.metadata, "filtered_category": category},
        )

    def sample(self, n: int, seed: int | None = None) -> "Dataset":
        """
        Random sample of n queries.

        Args:
            n: Number of queries to sample.
            seed: Optional random seed for reproducibility.

        Returns:
            New dataset with sampled queries.
        """
        if n >= len(self.queries):
            return self

        rng = random.Random(seed)
        sampled = rng.sample(self.queries, n)
        return Dataset(
            name=f"{self.name}_sample_{n}",
            description=f"Sampled {n} queries",
            queries=sampled,
            metadata={**self.metadata, "sample_size": n, "sample_seed": seed},
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "queries": [q.to_dict() for q in self.queries],
            "metadata": self.metadata,
        }


class DatasetConfig(BaseModel):
    """
    Configuration for dataset loading.

    Attributes:
        validate_schema: Whether to validate schema strictly.
        require_ground_truth: Whether ground truth is required.
        min_queries: Minimum number of queries required.
        allowed_categories: Optional list of allowed categories.
    """

    model_config = ConfigDict(frozen=True)

    validate_schema: bool = Field(default=True, description="Validate schema")
    require_ground_truth: bool = Field(default=True, description="Require ground truth")
    min_queries: int = Field(default=1, ge=1, description="Minimum queries")
    allowed_categories: list[str] | None = Field(default=None, description="Allowed categories")


# =============================================================================
# Split Results
# =============================================================================


@dataclass
class DatasetSplit:
    """
    Result of train/test split.

    Attributes:
        train: Training dataset.
        test: Test dataset.
        train_ratio: Ratio used for split.
        stratified: Whether split was stratified by category.
        seed: Random seed used.
    """

    train: Dataset
    test: Dataset
    train_ratio: float = 0.8
    stratified: bool = False
    seed: int | None = None


# =============================================================================
# Validation
# =============================================================================


def _validate_query_data(
    data: dict[str, Any],
    index: int,
    config: DatasetConfig,
) -> list[str]:
    """
    Validate a single query dictionary.

    Args:
        data: Query data dictionary.
        index: Query index for error messages.
        config: Dataset configuration.

    Returns:
        List of validation errors.
    """
    errors = []

    # Required fields
    if "query_id" not in data:
        errors.append(f"Query {index}: missing query_id")
    elif not isinstance(data["query_id"], str) or not data["query_id"]:
        errors.append(f"Query {index}: query_id must be non-empty string")

    if "query" not in data:
        errors.append(f"Query {index}: missing query field")
    elif not isinstance(data["query"], str) or not data["query"]:
        errors.append(f"Query {index}: query must be non-empty string")

    # Ground truth validation
    if config.require_ground_truth:
        if "relevant_doc_ids" not in data:
            errors.append(f"Query {index}: missing relevant_doc_ids (ground truth required)")
        elif not isinstance(data["relevant_doc_ids"], list):
            errors.append(f"Query {index}: relevant_doc_ids must be a list")
        elif len(data["relevant_doc_ids"]) == 0:
            errors.append(f"Query {index}: relevant_doc_ids cannot be empty")

    # Relevance scores validation
    if "relevance_scores" in data and data["relevance_scores"] is not None:
        scores = data["relevance_scores"]
        if not isinstance(scores, dict):
            errors.append(f"Query {index}: relevance_scores must be a dict")
        else:
            for doc_id, score in scores.items():
                if score not in (0, 1, 2):
                    errors.append(f"Query {index}: invalid relevance score {score}")

    # Category validation
    if config.allowed_categories and "category" in data:
        if data["category"] not in config.allowed_categories:
            errors.append(
                f"Query {index}: category '{data['category']}' not in allowed list"
            )

    return errors


def _validate_dataset(
    data: dict[str, Any] | list[dict[str, Any]],
    config: DatasetConfig,
) -> tuple[dict[str, Any], list[str]]:
    """
    Validate dataset data.

    Args:
        data: Raw dataset data.
        config: Dataset configuration.

    Returns:
        Tuple of (normalized data, validation errors).
    """
    errors = []

    # Handle list format (queries only)
    if isinstance(data, list):
        data = {"queries": data}

    # Ensure queries key exists
    if "queries" not in data:
        errors.append("Missing 'queries' key in dataset")
        return data, errors

    queries = data["queries"]
    if not isinstance(queries, list):
        errors.append("'queries' must be a list")
        return data, errors

    # Validate minimum queries
    if len(queries) < config.min_queries:
        errors.append(f"Dataset has {len(queries)} queries, minimum is {config.min_queries}")

    # Validate each query
    query_ids = set()
    for i, query in enumerate(queries):
        if not isinstance(query, dict):
            errors.append(f"Query {i}: must be a dictionary")
            continue

        # Check for duplicate query_ids
        qid = query.get("query_id")
        if qid in query_ids:
            errors.append(f"Duplicate query_id: {qid}")
        query_ids.add(qid)

        # Validate query fields
        query_errors = _validate_query_data(query, i, config)
        errors.extend(query_errors)

    return data, errors


# =============================================================================
# Dataset Manager
# =============================================================================


class DatasetManager:
    """
    Load, validate, and manage evaluation datasets.

    Supports JSON and JSONL formats with optional schema validation.

    Example:
        >>> manager = DatasetManager()
        >>> dataset = await manager.load("data/queries.json")
        >>> train, test = manager.split(dataset, train_ratio=0.8)
    """

    def __init__(self, base_path: Path | str | None = None) -> None:
        """
        Initialize the dataset manager.

        Args:
            base_path: Optional base path for relative file paths.
        """
        self._base_path = Path(base_path) if base_path else Path.cwd()

    async def load(
        self,
        path: Path | str,
        config: DatasetConfig | None = None,
    ) -> Dataset:
        """
        Load a dataset from file.

        Args:
            path: Path to dataset file (JSON or JSONL).
            config: Optional dataset configuration.

        Returns:
            Loaded and validated Dataset.

        Raises:
            DatasetError: If loading or validation fails.
        """
        config = config or DatasetConfig()
        file_path = self._resolve_path(path)

        if not file_path.exists():
            raise DatasetError(f"Dataset file not found: {file_path}")

        try:
            # Load file content
            async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                content = await f.read()

            # Parse based on extension
            if file_path.suffix == ".jsonl":
                data = self._parse_jsonl(content)
            else:
                data = json.loads(content)

            # Validate
            if config.validate_schema:
                data, errors = _validate_dataset(data, config)
                if errors:
                    raise DatasetError(
                        f"Dataset validation failed: {errors[0]}",
                        details={"errors": errors},
                    )

            # Create Dataset object
            return self._create_dataset(data, file_path.stem)

        except json.JSONDecodeError as e:
            raise DatasetError(f"Invalid JSON in {file_path}: {e}") from e
        except Exception as e:
            if isinstance(e, DatasetError):
                raise
            raise DatasetError(f"Failed to load dataset: {e}") from e

    def load_sync(
        self,
        path: Path | str,
        config: DatasetConfig | None = None,
    ) -> Dataset:
        """
        Synchronous version of load.

        Args:
            path: Path to dataset file.
            config: Optional dataset configuration.

        Returns:
            Loaded Dataset.
        """
        config = config or DatasetConfig()
        file_path = self._resolve_path(path)

        if not file_path.exists():
            raise DatasetError(f"Dataset file not found: {file_path}")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            if file_path.suffix == ".jsonl":
                data = self._parse_jsonl(content)
            else:
                data = json.loads(content)

            if config.validate_schema:
                data, errors = _validate_dataset(data, config)
                if errors:
                    raise DatasetError(
                        f"Dataset validation failed: {errors[0]}",
                        details={"errors": errors},
                    )

            return self._create_dataset(data, file_path.stem)

        except json.JSONDecodeError as e:
            raise DatasetError(f"Invalid JSON in {file_path}: {e}") from e
        except Exception as e:
            if isinstance(e, DatasetError):
                raise
            raise DatasetError(f"Failed to load dataset: {e}") from e

    async def save(
        self,
        dataset: Dataset,
        path: Path | str,
        format: str = "json",
    ) -> None:
        """
        Save a dataset to file.

        Args:
            dataset: Dataset to save.
            path: Output file path.
            format: Output format ('json' or 'jsonl').
        """
        file_path = self._resolve_path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "jsonl":
            content = "\n".join(
                json.dumps(q.to_dict(), ensure_ascii=False)
                for q in dataset.queries
            )
        else:
            content = json.dumps(dataset.to_dict(), indent=2, ensure_ascii=False)

        async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
            await f.write(content)

        logger.info(f"Saved dataset to {file_path}")

    def split(
        self,
        dataset: Dataset,
        train_ratio: float = 0.8,
        stratified: bool = False,
        seed: int | None = None,
    ) -> DatasetSplit:
        """
        Split dataset into train and test sets.

        Args:
            dataset: Dataset to split.
            train_ratio: Ratio of data for training (0 to 1).
            stratified: Whether to stratify by category.
            seed: Random seed for reproducibility.

        Returns:
            DatasetSplit with train and test datasets.

        Raises:
            InvalidInputError: If train_ratio is invalid.
        """
        if not 0 < train_ratio < 1:
            raise InvalidInputError("train_ratio", "Must be between 0 and 1")

        rng = random.Random(seed)

        if stratified and dataset.categories:
            train_queries, test_queries = self._stratified_split(
                dataset.queries,
                train_ratio,
                rng,
            )
        else:
            train_queries, test_queries = self._random_split(
                dataset.queries,
                train_ratio,
                rng,
            )

        train = Dataset(
            name=f"{dataset.name}_train",
            description="Training split",
            queries=train_queries,
            metadata={
                **dataset.metadata,
                "split": "train",
                "train_ratio": train_ratio,
            },
        )

        test = Dataset(
            name=f"{dataset.name}_test",
            description="Test split",
            queries=test_queries,
            metadata={
                **dataset.metadata,
                "split": "test",
                "train_ratio": train_ratio,
            },
        )

        return DatasetSplit(
            train=train,
            test=test,
            train_ratio=train_ratio,
            stratified=stratified,
            seed=seed,
        )

    def merge(self, *datasets: Dataset) -> Dataset:
        """
        Merge multiple datasets into one.

        Args:
            datasets: Datasets to merge.

        Returns:
            Merged dataset.
        """
        if not datasets:
            return Dataset()

        all_queries = []
        seen_ids: set[str] = set()

        for ds in datasets:
            for query in ds.queries:
                if query.query_id not in seen_ids:
                    all_queries.append(query)
                    seen_ids.add(query.query_id)

        return Dataset(
            name="_".join(ds.name for ds in datasets),
            description="Merged dataset",
            queries=all_queries,
            metadata={"merged_from": [ds.name for ds in datasets]},
        )

    def _resolve_path(self, path: Path | str) -> Path:
        """Resolve path relative to base path."""
        path = Path(path)
        if path.is_absolute():
            return path
        return self._base_path / path

    def _parse_jsonl(self, content: str) -> dict[str, Any]:
        """Parse JSONL content to dataset format."""
        queries = []
        for line in content.strip().split("\n"):
            if line.strip():
                queries.append(json.loads(line))
        return {"queries": queries}

    def _create_dataset(
        self,
        data: dict[str, Any],
        name: str,
    ) -> Dataset:
        """Create Dataset from parsed data."""
        queries = []
        for q in data.get("queries", []):
            queries.append(DatasetQuery(**q))

        return Dataset(
            name=data.get("name", name),
            description=data.get("description", ""),
            queries=queries,
            metadata=data.get("metadata", {}),
        )

    def _random_split(
        self,
        queries: list[DatasetQuery],
        train_ratio: float,
        rng: random.Random,
    ) -> tuple[list[DatasetQuery], list[DatasetQuery]]:
        """Random train/test split."""
        queries = list(queries)
        rng.shuffle(queries)
        split_idx = int(len(queries) * train_ratio)
        return queries[:split_idx], queries[split_idx:]

    def _stratified_split(
        self,
        queries: list[DatasetQuery],
        train_ratio: float,
        rng: random.Random,
    ) -> tuple[list[DatasetQuery], list[DatasetQuery]]:
        """Stratified train/test split by category."""
        # Group by category
        by_category: dict[str | None, list[DatasetQuery]] = {}
        for q in queries:
            cat = q.category
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(q)

        train_queries = []
        test_queries = []

        # Split each category
        for cat_queries in by_category.values():
            rng.shuffle(cat_queries)
            split_idx = max(1, int(len(cat_queries) * train_ratio))
            train_queries.extend(cat_queries[:split_idx])
            test_queries.extend(cat_queries[split_idx:])

        return train_queries, test_queries


# =============================================================================
# Convenience Functions
# =============================================================================


async def load_dataset(
    path: Path | str,
    config: DatasetConfig | None = None,
) -> Dataset:
    """
    Load a dataset from file.

    Args:
        path: Path to dataset file.
        config: Optional configuration.

    Returns:
        Loaded Dataset.
    """
    manager = DatasetManager()
    return await manager.load(path, config)


def create_dataset_from_queries(
    queries: list[dict[str, Any]],
    name: str = "custom",
) -> Dataset:
    """
    Create a dataset from a list of query dictionaries.

    Args:
        queries: List of query dictionaries.
        name: Dataset name.

    Returns:
        Dataset object.
    """
    dataset_queries = []
    for q in queries:
        if "query_id" not in q:
            q["query_id"] = f"q{len(dataset_queries)}"
        dataset_queries.append(DatasetQuery(**q))

    return Dataset(
        name=name,
        queries=dataset_queries,
    )
