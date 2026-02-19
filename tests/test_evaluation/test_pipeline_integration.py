"""
Integration tests for the full evaluation pipeline (ingestion → benchmark → report).

Runs run_pipeline with minimal data (1 query, 2 corpus docs) to catch integration
failures (e.g. strategy not found, DB/schema/embedder mismatch) in under a few
minutes instead of a full-hour run.

Does NOT wipe existing Postgres data: uses clean_before_ingest=False so previously
ingested corpus is preserved. The test adds 2 small docs (pipe_doc_1, pipe_doc_2)
and runs one benchmark query. For a fully isolated run, use a separate database
via TEST_DATABASE_URL.

Usage:
    pytest tests/test_evaluation/test_pipeline_integration.py -m integration -v
    # Requires DATABASE_URL (or TEST_DATABASE_URL) and EMBEDDING_BACKEND matching DB schema.
"""

from __future__ import annotations

import asyncio
import importlib.util
import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_SCRIPT = REPO_ROOT / "scripts" / "run_evaluation_pipeline.py"


def _load_pipeline_module():
    """Load run_evaluation_pipeline script as a module."""
    spec = importlib.util.spec_from_file_location(
        "run_evaluation_pipeline",
        PIPELINE_SCRIPT,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load script: {PIPELINE_SCRIPT}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.integration
@pytest.mark.asyncio
async def test_pipeline_e2e_minimal() -> None:
    """
    Run full pipeline with 1 query and 2 corpus docs to verify integration.

    Ensures: strategy registration, ingest, benchmark execution, and report
    generation complete without "strategy not found" or similar errors.
    Requires PostgreSQL with pgvector; EMBEDDING_BACKEND must match DB schema.
    """
    database_url = os.getenv("TEST_DATABASE_URL") or os.getenv("DATABASE_URL")
    if not database_url:
        pytest.skip("TEST_DATABASE_URL or DATABASE_URL not set")

    # BioASQ v1 columns: id, question, relevant_passage_ids (JSON string)
    gold_df = pd.DataFrame(
        [
            {
                "id": "pipe_q1",
                "question": "What is the main topic of doc_1?",
                "relevant_passage_ids": '["pipe_doc_1"]',
            },
        ]
    )
    # doc_id, passage, title
    corpus_df = pd.DataFrame(
        [
            {"doc_id": "pipe_doc_1", "passage": "Main topic is integration testing.", "title": "Doc 1"},
            {"doc_id": "pipe_doc_2", "passage": "Another short passage.", "title": "Doc 2"},
        ]
    )

    with tempfile.TemporaryDirectory(prefix="pipeline_e2e_") as tmpdir:
        tmp = Path(tmpdir)
        gold_path = tmp / "gold.xlsx"
        corpus_path = tmp / "corpus.xlsx"
        gold_df.to_excel(gold_path, index=False, engine="openpyxl")
        corpus_df.to_excel(corpus_path, index=False, engine="openpyxl")

        mod = _load_pipeline_module()
        run_pipeline = mod.run_pipeline

        try:
            await run_pipeline(
                gold_path=gold_path,
                corpus_path=corpus_path,
                gold_config=mod.BIOASQ_V1_GOLD_CONFIG,
                corpus_config=mod.BIOASQ_V1_CORPUS_CONFIG,
                strategies=["standard"],
                out_dir=None,
                limit=5,
                iterations=1,
                limit_queries=1,
                limit_corpus=2,
                clean_before_ingest=False,
                save_json_dataset=False,
                save_markdown_report=False,
            )
        except Exception as e:  # noqa: BLE001
            err_msg = str(e).lower()
            if "dimensions" in err_msg and "expected" in err_msg:
                pytest.skip(
                    "Embedding dimension mismatch: EMBEDDING_BACKEND does not match DB schema. "
                    "Default Docker uses 1536-dim (set EMBEDDING_BACKEND=openai or unset). "
                    "For BGE-M3 (1024-dim) use docker-compose.bge-m3.yml with a fresh volume."
                )
            if "strategy" in err_msg and "not found" in err_msg:
                raise AssertionError(
                    "Pipeline failed with strategy-not-found; ensure register_all_strategies "
                    "is called before benchmarks."
                ) from e
            raise
