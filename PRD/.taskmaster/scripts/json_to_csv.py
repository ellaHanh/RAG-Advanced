#!/usr/bin/env python3
"""
Convert Task Master JSON files to CSV for easier reading.

Output: same base name as input, .csv extension, in the same directory as the input.

Default input: .taskmaster/tasks/tasks.json (resolved relative to this script).
Explicit inputs: pass one or more absolute (or relative) paths.

Usage:
  python json_to_csv.py
  python json_to_csv.py /path/to/tasks.json
  python json_to_csv.py /path/to/tasks.json /path/to/task-complexity-report.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


# Default input: tasks.json in the taskmaster tasks directory (sibling of scripts/)
_SCRIPT_DIR = Path(__file__).resolve().parent
_TASKMASTER_ROOT = _SCRIPT_DIR.parent
DEFAULT_INPUT = _TASKMASTER_ROOT / "tasks" / "tasks.json"


def _list_or_str(value: object) -> str:
    """Serialize list as comma-separated string; others as str."""
    if isinstance(value, list):
        return ",".join(str(x) for x in value)
    return str(value) if value is not None else ""


def _tasks_to_rows(data: dict) -> list[dict[str, str]]:
    """Flatten master.tasks (and nested subtasks) into CSV-friendly rows."""
    tasks = data.get("master", {}).get("tasks") or []
    rows: list[dict[str, str]] = []
    for t in tasks:
        row: dict[str, str] = {
            "kind": "task",
            "id": str(t.get("id", "")),
            "parent_id": "",
            "title": str(t.get("title", "")),
            "description": str(t.get("description", "")),
            "details": str(t.get("details", "")),
            "testStrategy": str(t.get("testStrategy", "")),
            "priority": str(t.get("priority", "")),
            "dependencies": _list_or_str(t.get("dependencies")),
            "status": str(t.get("status", "")),
            "complexity": str(t.get("complexity", "")),
            "recommendedSubtasks": str(t.get("recommendedSubtasks", "")),
            "expansionPrompt": str(t.get("expansionPrompt", "")),
            "updatedAt": str(t.get("updatedAt", "")),
        }
        rows.append(row)
        for st in t.get("subtasks") or []:
            sub: dict[str, str] = {
                "kind": "subtask",
                "id": str(st.get("id", "")),
                "parent_id": str(t.get("id", "")),
                "title": str(st.get("title", "")),
                "description": str(st.get("description", "")),
                "details": str(st.get("details", "")),
                "testStrategy": str(st.get("testStrategy", "")),
                "priority": "",
                "dependencies": _list_or_str(st.get("dependencies")),
                "status": str(st.get("status", "")),
                "complexity": "",
                "recommendedSubtasks": "",
                "expansionPrompt": "",
                "updatedAt": "",
            }
            rows.append(sub)
    return rows


def _complexity_to_rows(data: dict) -> list[dict[str, str]]:
    """Flatten complexityAnalysis (and optional meta) into CSV-friendly rows."""
    meta = data.get("meta") or {}
    analyses = data.get("complexityAnalysis") or []
    rows: list[dict[str, str]] = []
    for a in analyses:
        row: dict[str, str] = {
            "taskId": str(a.get("taskId", "")),
            "taskTitle": str(a.get("taskTitle", "")),
            "complexityScore": str(a.get("complexityScore", "")),
            "recommendedSubtasks": str(a.get("recommendedSubtasks", "")),
            "expansionPrompt": str(a.get("expansionPrompt", "")),
            "reasoning": str(a.get("reasoning", "")),
            "reportGeneratedAt": str(meta.get("generatedAt", "")),
            "thresholdScore": str(meta.get("thresholdScore", "")),
            "tasksAnalyzed": str(meta.get("tasksAnalyzed", "")),
        }
        rows.append(row)
    return rows


def _detect_and_convert(data: dict, path: Path) -> tuple[list[dict[str, str]], list[str]]:
    """Return (list of row dicts, ordered field names) for CSV."""
    if "master" in data and "tasks" in data.get("master"):
        rows = _tasks_to_rows(data)
        names = [
            "kind", "id", "parent_id", "title", "description", "details",
            "testStrategy", "priority", "dependencies", "status", "complexity",
            "recommendedSubtasks", "expansionPrompt", "updatedAt",
        ]
        return rows, names
    if "complexityAnalysis" in data:
        rows = _complexity_to_rows(data)
        names = [
            "taskId", "taskTitle", "complexityScore", "recommendedSubtasks",
            "expansionPrompt", "reasoning", "reportGeneratedAt", "thresholdScore",
            "tasksAnalyzed",
        ]
        return rows, names
    raise ValueError(f"Unknown JSON structure in {path.name}; expected master.tasks or complexityAnalysis.")


def convert_one(in_path: Path) -> Path:
    """Convert one JSON file to CSV. Returns path to written CSV."""
    in_path = in_path.resolve()
    if not in_path.exists():
        raise FileNotFoundError(in_path)
    if in_path.suffix.lower() != ".json":
        raise ValueError(f"Expected .json file, got {in_path.suffix}")

    text = in_path.read_text(encoding="utf-8")
    data = json.loads(text)
    rows, fieldnames = _detect_and_convert(data, in_path)

    out_path = in_path.with_suffix(".csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert Task Master JSON to CSV (same name, same dir).",
        epilog="Default input: .taskmaster/tasks/tasks.json",
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help="Input JSON file path(s). If omitted, use default tasks.json.",
    )
    args = parser.parse_args()

    if args.inputs:
        paths = [p.resolve() for p in args.inputs]
    else:
        if not DEFAULT_INPUT.exists():
            print(f"Default input not found: {DEFAULT_INPUT}", file=sys.stderr)
            print("Pass explicit path(s), e.g. python json_to_csv.py /path/to/tasks.json", file=sys.stderr)
            return 1
        paths = [DEFAULT_INPUT]

    for p in paths:
        try:
            out = convert_one(p)
            print(out)
        except Exception as e:
            print(f"Error converting {p}: {e}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
