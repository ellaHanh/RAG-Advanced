"""
Load evaluation gold and corpus data from xlsx with configurable column mapping.

Supports config file (XlsxGoldConfig / XlsxCorpusConfig) or default BioASQ v1 mapping.
Gold: one row per query; relevant_doc_ids column is parsed (JSON, pipe, or comma).
Corpus: one row per document/passage; returns list of dicts with id, text, title.

Usage:
    from evaluation.loaders import load_gold_dataset_from_xlsx, load_corpus_xlsx
    from evaluation.loaders.xlsx_config import BIOASQ_V1_GOLD_CONFIG

    dataset = load_gold_dataset_from_xlsx("gold.xlsx", config=BIOASQ_V1_GOLD_CONFIG)
    docs = load_corpus_xlsx("corpus.xlsx", config=BIOASQ_V1_CORPUS_CONFIG)
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd

from evaluation.datasets import Dataset, DatasetQuery
from evaluation.loaders.xlsx_config import XlsxCorpusConfig, XlsxGoldConfig

logger = logging.getLogger(__name__)


def _parse_relevant_ids(
    value: Any,
    list_format: str,
) -> list[str]:
    """
    Parse relevant_doc_ids from cell value to list of strings.

    Args:
        value: Cell value (str, list, or NaN).
        list_format: One of "json", "pipe_separated", "comma_separated".

    Returns:
        List of doc/passage IDs. Empty if value is missing or invalid.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if x is not None and str(x).strip()]
    s = str(value).strip()
    if not s:
        return []
    if list_format == "json":
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
            return [str(parsed).strip()]
        except (json.JSONDecodeError, TypeError):
            # Fallback: treat as comma-separated
            return [x.strip() for x in re.split(r"[,;]", s) if x.strip()]
    if list_format == "pipe_separated":
        return [x.strip() for x in s.split("|") if x.strip()]
    if list_format == "comma_separated":
        return [x.strip() for x in re.split(r"[,;]", s) if x.strip()]
    return [x.strip() for x in re.split(r"[,;|]", s) if x.strip()]


def load_gold_dataset_from_xlsx(
    path: Path | str,
    config: XlsxGoldConfig | None = None,
    dataset_name: str | None = None,
) -> Dataset:
    """
    Load gold (query + ground truth) from xlsx into a Dataset.

    One row per query. relevant_doc_ids column is parsed according to config.list_format.

    Args:
        path: Path to the xlsx file.
        config: Column mapping; if None, uses BioASQ v1 default.
        dataset_name: Name for the Dataset; defaults to file stem.

    Returns:
        Dataset with DatasetQuery entries (query_id, query, relevant_doc_ids).

    Raises:
        FileNotFoundError: If path does not exist.
        ValueError: If required columns are missing.
    """
    from evaluation.loaders.xlsx_config import BIOASQ_V1_GOLD_CONFIG

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Gold xlsx not found: {path}")
    cfg = config or BIOASQ_V1_GOLD_CONFIG
    df = pd.read_excel(path, sheet_name=cfg.sheet_index, engine="openpyxl")
    for col in (cfg.query_id_column, cfg.query_column, cfg.relevant_doc_ids_column):
        if col not in df.columns:
            raise ValueError(
                f"Gold xlsx missing column '{col}'. Available: {list(df.columns)}"
            )
    queries: list[DatasetQuery] = []
    for _, row in df.iterrows():
        qid = str(row[cfg.query_id_column]).strip() if pd.notna(row[cfg.query_id_column]) else ""
        qtext = str(row[cfg.query_column]).strip() if pd.notna(row[cfg.query_column]) else ""
        if not qid or not qtext:
            logger.warning("Skipping row with missing query_id or query: %s", row.to_dict())
            continue
        rel_ids = _parse_relevant_ids(
            row.get(cfg.relevant_doc_ids_column),
            cfg.list_format,
        )
        queries.append(
            DatasetQuery(
                query_id=qid,
                query=qtext,
                relevant_doc_ids=rel_ids,
                metadata={"source": str(path.name)},
            )
        )
    name = dataset_name or path.stem
    return Dataset(
        name=name,
        description=f"Gold dataset loaded from {path.name}",
        queries=queries,
        metadata={"path": str(path), "config": cfg.model_dump()},
    )


def load_corpus_xlsx(
    path: Path | str,
    config: XlsxCorpusConfig | None = None,
) -> list[dict[str, Any]]:
    """
    Load corpus (documents/passages) from xlsx.

    Returns list of dicts with keys: id, text, title (title optional).
    Suitable for writing to temp files for ingestion or for in-memory use.

    Args:
        path: Path to the xlsx file.
        config: Column mapping; if None, uses BioASQ v1 default.

    Returns:
        List of {"id": str, "text": str, "title": str | None} (and any extra keys).

    Raises:
        FileNotFoundError: If path does not exist.
        ValueError: If required columns are missing.
    """
    from evaluation.loaders.xlsx_config import BIOASQ_V1_CORPUS_CONFIG

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Corpus xlsx not found: {path}")
    cfg = config or BIOASQ_V1_CORPUS_CONFIG
    df = pd.read_excel(path, sheet_name=cfg.sheet_index, engine="openpyxl")
    for col in (cfg.doc_id_column, cfg.text_column):
        if col not in df.columns:
            raise ValueError(
                f"Corpus xlsx missing column '{col}'. Available: {list(df.columns)}"
            )
    out: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        doc_id = str(row[cfg.doc_id_column]).strip() if pd.notna(row[cfg.doc_id_column]) else ""
        text = str(row[cfg.text_column]).strip() if pd.notna(row[cfg.text_column]) else ""
        if not doc_id:
            logger.warning("Skipping corpus row with missing doc_id")
            continue
        title: str | None = None
        if cfg.title_column and cfg.title_column in df.columns:
            val = row.get(cfg.title_column)
            title = str(val).strip() if pd.notna(val) else None
        out.append({"id": doc_id, "text": text, "title": title or ""})
    return out
