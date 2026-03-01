"""
Unit tests for pipeline script helpers (_parse_map_arg, _load_pipeline_config).

Helpers live in scripts/run_evaluation_pipeline.py; we load that module via
importlib to avoid making scripts a package.
"""

from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path

import pytest

# Load the pipeline script as a module to get _parse_map_arg and _load_pipeline_config
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


# =============================================================================
# _parse_map_arg
# =============================================================================


def test_parse_map_arg_basic() -> None:
    """Parse key=val,key2=val2 into dict."""
    mod = _load_pipeline_module()
    result = mod._parse_map_arg("query_id=id,query=question,relevant_doc_ids=relevant_passage_ids")
    assert result == {
        "query_id": "id",
        "query": "question",
        "relevant_doc_ids": "relevant_passage_ids",
    }


def test_parse_map_arg_strips_whitespace() -> None:
    """Keys and values are stripped."""
    mod = _load_pipeline_module()
    result = mod._parse_map_arg("  a = b  ,  c = d  ")
    assert result == {"a": "b", "c": "d"}


def test_parse_map_arg_skips_malformed() -> None:
    """Parts without '=' are skipped."""
    mod = _load_pipeline_module()
    result = mod._parse_map_arg("valid=ok,noequals,bad=")
    assert result == {"valid": "ok", "bad": ""}


def test_parse_map_arg_empty_string() -> None:
    """Empty string returns empty dict."""
    mod = _load_pipeline_module()
    assert mod._parse_map_arg("") == {}


# =============================================================================
# _load_pipeline_config
# =============================================================================


def test_load_pipeline_config_json() -> None:
    """_load_pipeline_config loads JSON and returns gold and corpus configs."""
    mod = _load_pipeline_module()
    config_data = {
        "gold": {
            "sheet_index": 0,
            "query_id_column": "id",
            "query_column": "question",
            "relevant_doc_ids_column": "relevant_passage_ids",
            "list_format": "json",
        },
        "corpus": {
            "sheet_index": 0,
            "doc_id_column": "doc_id",
            "text_column": "passage",
            "title_column": "title",
        },
    }
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        delete=False,
    ) as f:
        json.dump(config_data, f)
        path = Path(f.name)
    try:
        gold_cfg, corpus_cfg, _ = mod._load_pipeline_config(path)
        assert gold_cfg is not None
        assert gold_cfg.query_id_column == "id"
        assert gold_cfg.query_column == "question"
        assert gold_cfg.relevant_doc_ids_column == "relevant_passage_ids"
        assert gold_cfg.sheet_index == 0
        assert corpus_cfg is not None
        assert corpus_cfg.doc_id_column == "doc_id"
        assert corpus_cfg.text_column == "passage"
        assert corpus_cfg.title_column == "title"
    finally:
        path.unlink(missing_ok=True)


def test_load_pipeline_config_json_bioasq_v1_file() -> None:
    """_load_pipeline_config loads evaluation/config/bioasq_v1.json correctly."""
    config_path = REPO_ROOT / "evaluation" / "config" / "bioasq_v1.json"
    if not config_path.exists():
        pytest.skip("bioasq_v1.json not found")
    mod = _load_pipeline_module()
    gold_cfg, corpus_cfg, _ = mod._load_pipeline_config(config_path)
    assert gold_cfg is not None
    assert gold_cfg.query_id_column == "id"
    assert gold_cfg.relevant_doc_ids_column == "relevant_passage_ids"
    assert corpus_cfg is not None
    assert corpus_cfg.text_column == "passage"
    assert corpus_cfg.doc_id_column == "doc_id"


def test_load_pipeline_config_empty_json() -> None:
    """Empty or minimal JSON returns None, None or partial configs."""
    mod = _load_pipeline_module()
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        delete=False,
    ) as f:
        f.write("{}")
        path = Path(f.name)
    try:
        gold_cfg, corpus_cfg, ingestion_overrides = mod._load_pipeline_config(path)
        assert gold_cfg is None
        assert corpus_cfg is None
        assert ingestion_overrides is None
    finally:
        path.unlink(missing_ok=True)


# =============================================================================
# Semantic-chunking option (--semantic-chunking -> IngestionConfig)
# =============================================================================


def test_semantic_chunking_option_builds_config_with_use_semantic_true() -> None:
    """When --semantic-chunking is passed, pipeline builds IngestionConfig with use_semantic_chunking=True."""
    from strategies.ingestion.models import IngestionConfig

    mod = _load_pipeline_module()
    base = mod.PIPELINE_DEFAULT_INGESTION.model_dump()
    assert base["use_semantic_chunking"] is False
    base["use_semantic_chunking"] = True
    cfg = IngestionConfig(**base)
    assert cfg.use_semantic_chunking is True
