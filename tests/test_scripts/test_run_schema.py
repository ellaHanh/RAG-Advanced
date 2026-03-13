"""
Unit tests for run_schema script: schema path selection by EMBEDDING_BACKEND.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_SCHEMA_SCRIPT = REPO_ROOT / "scripts" / "run_schema.py"


def _load_run_schema_module(module_name: str):
    """Load run_schema.py as a module with the given name (env must be set before)."""
    spec = importlib.util.spec_from_file_location(module_name, RUN_SCHEMA_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load script: {RUN_SCHEMA_SCRIPT}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_schema_base_bge_m3(monkeypatch: pytest.MonkeyPatch) -> None:
    """When EMBEDDING_BACKEND=bge-m3, SCHEMA_BASE is schema_1024.sql."""
    monkeypatch.setenv("EMBEDDING_BACKEND", "bge-m3")
    monkeypatch.chdir(REPO_ROOT)
    cwd_before = os.getcwd()
    try:
        mod = _load_run_schema_module("run_schema_bge")
        assert mod.SCHEMA_BASE.name == "schema_1024.sql"
        assert "schema_1024" in str(mod.SCHEMA_BASE)
    finally:
        os.chdir(cwd_before)


def test_schema_base_openai_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """When EMBEDDING_BACKEND is unset or openai, SCHEMA_BASE is schema.sql."""
    monkeypatch.setenv("EMBEDDING_BACKEND", "openai")  # override .env so module sees openai
    monkeypatch.chdir(REPO_ROOT)
    cwd_before = os.getcwd()
    try:
        mod = _load_run_schema_module("run_schema_default")
        assert mod.SCHEMA_BASE.name == "schema.sql"
        assert "schema_1024" not in str(mod.SCHEMA_BASE)
    finally:
        os.chdir(cwd_before)

    try:
        mod2 = _load_run_schema_module("run_schema_openai")
        assert mod2.SCHEMA_BASE.name == "schema.sql"
    finally:
        os.chdir(cwd_before)
