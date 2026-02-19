"""
RAG-Advanced evaluation data loaders.

Load evaluation corpus and gold datasets from xlsx (and optionally other formats)
with configurable column mapping for different dataset sources.

Exports:
    - load_gold_dataset_from_xlsx: Load gold (query + relevant_doc_ids) from xlsx.
    - load_corpus_xlsx: Load corpus (doc_id, text, title) from xlsx.
    - XlsxGoldConfig, XlsxCorpusConfig: Column mapping config models.
    - BIOASQ_V1_GOLD_CONFIG, BIOASQ_V1_CORPUS_CONFIG: Default configs for BioASQ v1.
"""

from evaluation.loaders.xlsx_config import (
    BIOASQ_V1_CORPUS_CONFIG,
    BIOASQ_V1_GOLD_CONFIG,
    XlsxCorpusConfig,
    XlsxGoldConfig,
)
from evaluation.loaders.xlsx_loader import load_corpus_xlsx, load_gold_dataset_from_xlsx

__all__ = [
    "load_gold_dataset_from_xlsx",
    "load_corpus_xlsx",
    "XlsxGoldConfig",
    "XlsxCorpusConfig",
    "BIOASQ_V1_GOLD_CONFIG",
    "BIOASQ_V1_CORPUS_CONFIG",
]
