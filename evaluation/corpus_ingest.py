"""
Ingest evaluation corpus (list of docs with id, text, title) into the app DB.

Writes corpus to a temporary directory as one .txt file per document (so existing
ingestion pipeline can read and chunk them), runs run_ingestion, then builds
a mapping from corpus doc_id to chunk ID(s) for aligning gold relevant_passage_ids
with retrieval results (which return chunk IDs).

Usage:
    from evaluation.corpus_ingest import ingest_corpus_and_get_chunk_map

    corpus = load_corpus_xlsx("corpus.xlsx")
    doc_id_to_chunk_ids = await ingest_corpus_and_get_chunk_map(corpus, clean_before=True)
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import asyncpg

from strategies.ingestion.ingest import run_ingestion
from strategies.ingestion.models import IngestionConfig

logger = logging.getLogger(__name__)


def doc_id_to_stem(doc_id: str) -> str:
    """
    Sanitize doc_id to the same stem used as filename in _write_corpus_to_dir.

    Used when building doc_id -> chunk_ids from existing DB (e.g. --skip-ingest):
    DB stores source like "stem.txt"; gold has original doc_id; this maps doc_id -> stem.

    Args:
        doc_id: Original document ID (e.g. from gold relevant_passage_ids).

    Returns:
        Stem string (e.g. "doc_1", "PubMed_123").
    """
    s = str(doc_id).strip()
    stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in s)
    return stem or "doc"


def _write_corpus_to_dir(
    corpus: list[dict[str, Any]],
    dir_path: Path,
    *,
    id_key: str = "id",
    text_key: str = "text",
    title_key: str = "title",
) -> list[tuple[str, str]]:
    """
    Write corpus to directory as one .txt file per document.

    Filename: sanitized doc_id + .txt. Content: plain text (passage). Title is
    taken from filename by extract_title when no "# " heading in content.

    Args:
        corpus: List of dicts with id, text, optional title.
        dir_path: Directory to write files into (created if needed).
        id_key: Key for document id (used in filename).
        text_key: Key for passage text.
        title_key: Key for title (optional; if present, prepend "# title" to content).

    Returns:
        List of (original_doc_id, file_stem) so caller can map back to chunk IDs.
    """
    dir_path.mkdir(parents=True, exist_ok=True)
    written: list[tuple[str, str]] = []
    for doc in corpus:
        doc_id = str(doc.get(id_key, "")).strip()
        if not doc_id:
            continue
        text = str(doc.get(text_key, "") or "").strip()
        title = str(doc.get(title_key, "") or "").strip()
        if title:
            content = f"# {title}\n\n{text}" if text else f"# {title}"
        else:
            content = text or "(empty)"
        stem = doc_id_to_stem(doc_id)
        path = dir_path / f"{stem}.txt"
        path.write_text(content, encoding="utf-8")
        written.append((doc_id, stem))
    return written


async def get_doc_id_to_chunk_ids(
    pool: asyncpg.Pool,
    *,
    source_suffix: str = ".txt",
) -> dict[str, list[str]]:
    """
    Query DB for document source -> chunk id(s). Sources are filenames like doc_id.txt.

    Returns mapping from doc_id (source without suffix) to list of chunk UUID strings.
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT c.id::text AS chunk_id, d.source
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            """
        )
    out: dict[str, list[str]] = {}
    for row in rows:
        source = row["source"] or ""
        if source_suffix and source.endswith(source_suffix):
            doc_id = source[: -len(source_suffix)]
        else:
            doc_id = source
        chunk_id = row["chunk_id"]
        if doc_id not in out:
            out[doc_id] = []
        out[doc_id].append(chunk_id)
    return out


async def ingest_corpus_and_get_chunk_map(
    corpus: list[dict[str, Any]],
    config: IngestionConfig | None = None,
    clean_before: bool = True,
    progress_cb: Any = None,
) -> dict[str, list[str]]:
    """
    Write corpus to temp dir, run ingestion, return doc_id -> chunk_id(s) mapping.

    Corpus items must have "id" and "text" (and optionally "title"). Each is
    written as {id}.txt; after ingestion we query chunks joined with documents
    to map source (filename) to chunk UUID. Gold relevant_passage_ids (corpus
    doc_ids) can then be translated to chunk IDs for benchmark metrics.

    Args:
        corpus: List of {"id", "text", "title"?} from load_corpus_xlsx.
        config: Ingestion config; default chunk size/overlap.
        clean_before: If True, delete existing documents/chunks before ingest.
        progress_cb: Optional callback(current, total) during ingestion.

    Returns:
        Mapping from corpus doc_id to list of chunk UUID strings.

    Raises:
        ValueError: If DATABASE_URL is not set.
    """
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL must be set")
    config = config or IngestionConfig(
        chunk_size=2000,
        chunk_overlap=0,
        use_semantic_chunking=False,
    )
    with tempfile.TemporaryDirectory(prefix="rag_eval_corpus_") as tmpdir:
        dir_path = Path(tmpdir)
        original_to_stem = dict(_write_corpus_to_dir(corpus, dir_path))
        results = await run_ingestion(
            str(dir_path),
            config,
            clean_before=clean_before,
            progress_cb=progress_cb,
        )
        if not results:
            raise ValueError(
                "No documents were ingested; run_ingestion found no files in the corpus directory."
            )
        failed = [r for r in results if r.errors or r.chunks_created == 0]
        if failed:
            first_err = failed[0].errors[0] if failed[0].errors else "no chunks produced"
            raise ValueError(
                f"Ingestion failed for {len(failed)} doc(s). First error: {first_err}"
            )
        pool = await asyncpg.create_pool(
            database_url,
            min_size=1,
            max_size=2,
            command_timeout=120,
        )
        try:
            stem_to_chunk_ids = await get_doc_id_to_chunk_ids(pool, source_suffix=".txt")
            out = {
                orig: stem_to_chunk_ids.get(stem, [])
                for orig, stem in original_to_stem.items()
            }
            if not any(out.values()) and original_to_stem:
                raise ValueError(
                    "Ingestion reported success but no doc_id->chunk_ids mapping; "
                    "possible source path mismatch (DB source keys: "
                    f"{list(stem_to_chunk_ids.keys())!r})."
                )
            return out
        finally:
            await pool.close()
