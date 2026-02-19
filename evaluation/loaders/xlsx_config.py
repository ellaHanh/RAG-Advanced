"""
Column mapping config for xlsx evaluation loaders.

Defines Pydantic models for corpus and gold xlsx column mapping so different
dataset sources (BioASQ, custom) can be used with the same loader via config
file (YAML/JSON) or CLI arguments.

Usage:
    from evaluation.loaders.xlsx_config import XlsxGoldConfig, BIOASQ_V1_GOLD_CONFIG
    dataset = load_gold_dataset_from_xlsx(path, config=BIOASQ_V1_GOLD_CONFIG)
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class XlsxGoldConfig(BaseModel):
    """
    Column mapping for gold (query + ground truth) xlsx.

    Maps xlsx column names to canonical fields: query_id, query, relevant_doc_ids.
    relevant_doc_ids column may contain a string list (e.g. JSON or delimiter-separated).
    """

    model_config = ConfigDict(frozen=True)

    sheet_index: int = Field(default=0, ge=0, description="Zero-based sheet index")
    query_id_column: str = Field(..., description="Xlsx column for query id")
    query_column: str = Field(..., description="Xlsx column for query text")
    relevant_doc_ids_column: str = Field(
        ...,
        description="Xlsx column containing list of relevant doc/passage IDs",
    )
    list_format: Literal["json", "pipe_separated", "comma_separated"] = Field(
        default="json",
        description="How relevant_doc_ids string is parsed into a list",
    )


class XlsxCorpusConfig(BaseModel):
    """
    Column mapping for corpus xlsx.

    Maps xlsx column names to canonical fields: doc_id, text (passage/content), title.
    """

    model_config = ConfigDict(frozen=True)

    sheet_index: int = Field(default=0, ge=0, description="Zero-based sheet index")
    doc_id_column: str = Field(..., description="Xlsx column for document/passage id")
    text_column: str = Field(..., description="Xlsx column for passage/content text")
    title_column: str | None = Field(
        default=None,
        description="Xlsx column for title (optional)",
    )


# Default configs for BioASQ v1 (first sheet, column names as provided)
BIOASQ_V1_GOLD_CONFIG = XlsxGoldConfig(
    sheet_index=0,
    query_id_column="id",
    query_column="question",
    relevant_doc_ids_column="relevant_passage_ids",
    list_format="json",
)

BIOASQ_V1_CORPUS_CONFIG = XlsxCorpusConfig(
    sheet_index=0,
    doc_id_column="doc_id",
    text_column="passage",
    title_column="title",
)
