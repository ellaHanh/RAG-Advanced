#!/usr/bin/env python3
"""
Run RAG-Advanced database schema without requiring psql.

Reads DATABASE_URL from environment (.env loaded if present), then executes
strategies/utils/schema.sql and evaluation/schema_extension.sql in order.

Usage (from repo root):
  python scripts/run_schema.py

Requires: asyncpg, python-dotenv (from project deps).
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

# Repo root: parent of scripts/
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
os.chdir(REPO_ROOT)

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass

DATABASE_URL = os.getenv("DATABASE_URL")
SCHEMA_BASE = REPO_ROOT / "strategies" / "utils" / "schema.sql"
SCHEMA_EXT = REPO_ROOT / "evaluation" / "schema_extension.sql"


def split_sql(content: str) -> list[str]:
    """
    Split SQL into single statements, respecting $$...$$ dollar-quoted strings.

    Semicolons inside $$...$$ are not treated as statement separators.
    """
    statements: list[str] = []
    current: list[str] = []
    i = 0
    inside_dollar = False
    n = len(content)

    while i < n:
        if not inside_dollar:
            if content[i:i + 2] == "$$":
                current.append("$$")
                i += 2
                inside_dollar = True
                continue
            if content[i] == ";":
                stmt = "".join(current).strip()
                if stmt and not re.match(r"^\s*--", stmt):
                    statements.append(stmt)
                current = []
                i += 1
                continue
            current.append(content[i])
            i += 1
        else:
            if content[i:i + 2] == "$$":
                current.append("$$")
                i += 2
                inside_dollar = False
                continue
            current.append(content[i])
            i += 1

    stmt = "".join(current).strip()
    if stmt and not re.match(r"^\s*--", stmt):
        statements.append(stmt)
    return statements


async def run_schema() -> None:
    """Execute schema files against DATABASE_URL."""
    if not DATABASE_URL:
        print("DATABASE_URL is not set. Set it in .env or the environment.", file=sys.stderr)
        sys.exit(1)
    if not SCHEMA_BASE.exists():
        print(f"Schema file not found: {SCHEMA_BASE}", file=sys.stderr)
        sys.exit(1)
    if not SCHEMA_EXT.exists():
        print(f"Schema extension not found: {SCHEMA_EXT}", file=sys.stderr)
        sys.exit(1)

    import asyncpg

    conn = await asyncpg.connect(DATABASE_URL)

    try:
        for label, path in [("base", SCHEMA_BASE), ("evaluation", SCHEMA_EXT)]:
            content = path.read_text(encoding="utf-8")
            statements = split_sql(content)
            for stmt in statements:
                stmt_clean = stmt.strip()
                if not stmt_clean or stmt_clean.startswith("--"):
                    continue
                try:
                    await conn.execute(stmt_clean)
                except Exception as e:
                    print(f"Error executing statement ({label}): {e}", file=sys.stderr)
                    print(f"Statement (first 200 chars): {stmt_clean[:200]}...", file=sys.stderr)
                    raise
            print(f"Applied {path.name} ({label})")
    finally:
        await conn.close()

    print("Schema applied successfully.")


def main() -> int:
    """Entry point."""
    try:
        import asyncio
        asyncio.run(run_schema())
        return 0
    except Exception as e:
        print(f"Schema run failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
