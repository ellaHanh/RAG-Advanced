"""
Unit tests for xlsx evaluation loaders.

Tests gold and corpus loaders with configurable column mapping and list parsing.
Uses pandas to create temporary xlsx fixtures (no committed binary files).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from evaluation.loaders.xlsx_config import (
    BIOASQ_V1_CORPUS_CONFIG,
    BIOASQ_V1_GOLD_CONFIG,
    XlsxCorpusConfig,
    XlsxGoldConfig,
)
from pydantic import ValidationError
from evaluation.loaders.xlsx_loader import load_corpus_xlsx, load_gold_dataset_from_xlsx


# =============================================================================
# Gold loader
# =============================================================================


@pytest.fixture
def gold_xlsx_path() -> Path:
    """Create a temporary gold xlsx (BioASQ-style columns)."""
    df = pd.DataFrame(
        [
            {
                "id": "q1",
                "question": "What is hypertension?",
                "relevant_passage_ids": '["p1", "p2"]',
                "answer": "High blood pressure",
            },
            {
                "id": "q2",
                "question": "What causes diabetes?",
                "relevant_passage_ids": '["p3"]',
                "answer": "Insulin resistance",
            },
        ]
    )
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
        df.to_excel(f.name, index=False, engine="openpyxl")
        path = Path(f.name)
    yield path
    path.unlink(missing_ok=True)


def test_load_gold_dataset_from_xlsx_default_config(gold_xlsx_path: Path) -> None:
    """Gold loader with default BioASQ v1 config produces Dataset with parsed list."""
    dataset = load_gold_dataset_from_xlsx(gold_xlsx_path, config=BIOASQ_V1_GOLD_CONFIG)
    assert dataset.name == gold_xlsx_path.stem
    assert len(dataset.queries) == 2
    q1 = dataset.queries[0]
    assert q1.query_id == "q1"
    assert q1.query == "What is hypertension?"
    assert q1.relevant_doc_ids == ["p1", "p2"]
    q2 = dataset.queries[1]
    assert q2.relevant_doc_ids == ["p3"]


def test_load_gold_dataset_from_xlsx_custom_config() -> None:
    """Gold loader with custom column names works."""
    df = pd.DataFrame(
        [
            {"qid": "a", "text": "Query A", "doc_ids": "id1|id2"},
        ]
    )
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
        df.to_excel(f.name, index=False, engine="openpyxl")
        path = Path(f.name)
    try:
        config = XlsxGoldConfig(
            query_id_column="qid",
            query_column="text",
            relevant_doc_ids_column="doc_ids",
            list_format="pipe_separated",
        )
        dataset = load_gold_dataset_from_xlsx(path, config=config)
        assert len(dataset.queries) == 1
        assert dataset.queries[0].relevant_doc_ids == ["id1", "id2"]
    finally:
        path.unlink(missing_ok=True)


def test_load_gold_dataset_from_xlsx_missing_columns() -> None:
    """Gold loader raises when required column is missing."""
    df = pd.DataFrame([{"id": "q1", "question": "Q?"}])  # no relevant_passage_ids
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
        df.to_excel(f.name, index=False, engine="openpyxl")
        path = Path(f.name)
    try:
        with pytest.raises(ValueError, match="missing column"):
            load_gold_dataset_from_xlsx(path, config=BIOASQ_V1_GOLD_CONFIG)
    finally:
        path.unlink(missing_ok=True)


def test_load_gold_dataset_from_xlsx_file_not_found() -> None:
    """Gold loader raises FileNotFoundError for missing file."""
    with pytest.raises(FileNotFoundError, match="not found"):
        load_gold_dataset_from_xlsx("/nonexistent/gold.xlsx")


# =============================================================================
# Corpus loader
# =============================================================================


@pytest.fixture
def corpus_xlsx_path() -> Path:
    """Create a temporary corpus xlsx (doc_id, passage, title)."""
    df = pd.DataFrame(
        [
            {"doc_id": "p1", "passage": "Content one.", "title": "Title 1"},
            {"doc_id": "p2", "passage": "Content two.", "title": "Title 2"},
        ]
    )
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
        df.to_excel(f.name, index=False, engine="openpyxl")
        path = Path(f.name)
    yield path
    path.unlink(missing_ok=True)


def test_load_corpus_xlsx_default_config(corpus_xlsx_path: Path) -> None:
    """Corpus loader with default config returns list of id/text/title dicts."""
    docs = load_corpus_xlsx(corpus_xlsx_path, config=BIOASQ_V1_CORPUS_CONFIG)
    assert len(docs) == 2
    assert docs[0]["id"] == "p1"
    assert docs[0]["text"] == "Content one."
    assert docs[0]["title"] == "Title 1"
    assert docs[1]["id"] == "p2"
    assert docs[1]["text"] == "Content two."


def test_load_corpus_xlsx_custom_config() -> None:
    """Corpus loader with custom column names works."""
    df = pd.DataFrame(
        [{"pmid": "123", "abstract": "Some text.", "article_title": "A Title"}]
    )
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
        df.to_excel(f.name, index=False, engine="openpyxl")
        path = Path(f.name)
    try:
        config = XlsxCorpusConfig(
            doc_id_column="pmid",
            text_column="abstract",
            title_column="article_title",
        )
        docs = load_corpus_xlsx(path, config=config)
        assert len(docs) == 1
        assert docs[0]["id"] == "123"
        assert docs[0]["text"] == "Some text."
        assert docs[0]["title"] == "A Title"
    finally:
        path.unlink(missing_ok=True)


def test_load_corpus_xlsx_file_not_found() -> None:
    """Corpus loader raises FileNotFoundError for missing file."""
    with pytest.raises(FileNotFoundError, match="not found"):
        load_corpus_xlsx("/nonexistent/corpus.xlsx")


# =============================================================================
# xlsx_config model validation
# =============================================================================


def test_xlsx_gold_config_validation() -> None:
    """XlsxGoldConfig requires query_id, query, relevant_doc_ids; list_format enum; sheet_index >= 0."""
    cfg = XlsxGoldConfig(
        query_id_column="id",
        query_column="question",
        relevant_doc_ids_column="relevant_passage_ids",
        list_format="json",
    )
    assert cfg.sheet_index == 0
    assert cfg.list_format == "json"

    with pytest.raises(ValidationError):
        XlsxGoldConfig(
            query_id_column="id",
            query_column="question",
            relevant_doc_ids_column="ids",
            list_format="invalid",
        )
    with pytest.raises(ValidationError):
        XlsxGoldConfig(
            query_id_column="id",
            query_column="question",
            relevant_doc_ids_column="ids",
            sheet_index=-1,
        )


def test_xlsx_corpus_config_validation() -> None:
    """XlsxCorpusConfig requires doc_id, text; title optional; sheet_index >= 0."""
    cfg = XlsxCorpusConfig(
        doc_id_column="doc_id",
        text_column="passage",
        title_column="title",
    )
    assert cfg.sheet_index == 0
    assert cfg.title_column == "title"

    cfg2 = XlsxCorpusConfig(doc_id_column="id", text_column="text")
    assert cfg2.title_column is None

    with pytest.raises(ValidationError):
        XlsxCorpusConfig(
            doc_id_column="id",
            text_column="text",
            sheet_index=-1,
        )
