"""
Ingestion pipeline: read documents, chunk, embed, and store in PostgreSQL.

Usage (from repo root):
  python -m strategies.ingestion.ingest --documents /path/to/documents
  python -m strategies.ingestion.ingest --documents ./documents --no-clean
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Callable

import asyncpg
from dotenv import load_dotenv

from strategies.ingestion.chunker import chunk_document
from strategies.ingestion.document_reader import extract_title, read_document
from strategies.ingestion.embedder import embed_chunks
from strategies.ingestion.models import DocumentChunk, IngestionConfig, IngestionResult

logger = logging.getLogger(__name__)

# Repo root for default .env
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(REPO_ROOT / ".env")

# Supported file patterns (same as all-rag-strategies)
DOCUMENT_GLOBS = [
    "*.md",
    "*.markdown",
    "*.txt",
    "*.pdf",
    "*.docx",
    "*.doc",
    "*.pptx",
    "*.ppt",
    "*.xlsx",
    "*.xls",
    "*.html",
    "*.htm",
    "*.mp3",
    "*.wav",
    "*.m4a",
    "*.flac",
]


def find_document_files(documents_dir: str) -> list[str]:
    """Return sorted list of supported document paths under documents_dir."""
    base = Path(documents_dir).resolve()
    if not base.exists():
        logger.error("Documents folder not found: %s", base)
        return []
    paths: list[Path] = []
    for pattern in DOCUMENT_GLOBS:
        paths.extend(base.rglob(pattern))
    return sorted(str(p) for p in set(paths))


async def save_document_and_chunks(
    conn: asyncpg.Connection,
    title: str,
    source: str,
    content: str,
    metadata: dict,
    chunks: list[DocumentChunk],
) -> str:
    """
    Insert one document and its chunks; return document UUID string.
    Chunks must have embedding set (list of 1536 floats).
    """
    doc_meta = json.dumps(metadata)
    row = await conn.fetchrow(
        """
        INSERT INTO documents (title, source, content, metadata)
        VALUES ($1, $2, $3, $4)
        RETURNING id::text
        """,
        title,
        source,
        content,
        doc_meta,
    )
    doc_id = row["id"]
    for c in chunks:
        if c.embedding is None:
            raise ValueError(f"Chunk {c.index} has no embedding")
        emb_str = "[" + ",".join(str(x) for x in c.embedding) + "]"
        await conn.execute(
            """
            INSERT INTO chunks (document_id, content, embedding, chunk_index, metadata)
            VALUES ($1::uuid, $2, $3::vector, $4, $5)
            """,
            doc_id,
            c.content,
            emb_str,
            c.index,
            json.dumps(c.metadata),
        )
    return doc_id


async def ingest_one(
    pool: asyncpg.Pool,
    file_path: str,
    documents_dir: str,
    config: IngestionConfig,
) -> IngestionResult:
    """Read, chunk, embed, and save one document."""
    start = datetime.now()
    try:
        content, docling_doc = read_document(file_path)
        title = extract_title(content, file_path)
        source = os.path.relpath(file_path, documents_dir)
        doc_metadata: dict = {
            "file_path": file_path,
            "ingestion_date": datetime.now().isoformat(),
            "line_count": len(content.splitlines()),
        }
        chunks = chunk_document(
            content,
            title,
            source,
            config,
            metadata=doc_metadata,
            docling_doc=docling_doc,
        )
        if not chunks:
            return IngestionResult(
                document_id="",
                title=title,
                chunks_created=0,
                processing_time_ms=(datetime.now() - start).total_seconds() * 1000,
                errors=["No chunks produced"],
            )
        await embed_chunks(chunks)
        async with pool.acquire() as conn:
            doc_id = await save_document_and_chunks(
                conn,
                title,
                source,
                content,
                doc_metadata,
                chunks,
            )
        elapsed_ms = (datetime.now() - start).total_seconds() * 1000
        return IngestionResult(
            document_id=doc_id,
            title=title,
            chunks_created=len(chunks),
            processing_time_ms=elapsed_ms,
            errors=[],
        )
    except Exception as e:
        logger.exception("Ingest failed for %s", file_path)
        return IngestionResult(
            document_id="",
            title=os.path.basename(file_path),
            chunks_created=0,
            processing_time_ms=(datetime.now() - start).total_seconds() * 1000,
            errors=[str(e)],
        )


async def run_ingestion(
    documents_dir: str,
    config: IngestionConfig,
    clean_before: bool = True,
    progress_cb: Callable[[int, int], None] | None = None,
) -> list[IngestionResult]:
    """
    Ingest all supported documents under documents_dir.

    Args:
        documents_dir: Path to folder containing documents.
        config: Chunk size, overlap, semantic chunking.
        clean_before: If True, delete existing documents and chunks first.
        progress_cb: Optional callback(current_index, total_count).

    Returns:
        List of IngestionResult per document.
    """
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL must be set")
    pool = await asyncpg.create_pool(
        database_url,
        min_size=1,
        max_size=5,
        command_timeout=120,
    )
    try:
        if clean_before:
            async with pool.acquire() as conn:
                await conn.execute("DELETE FROM chunks")
                await conn.execute("DELETE FROM documents")
            logger.info("Cleaned existing documents and chunks")
        files = find_document_files(documents_dir)
        if not files:
            logger.warning("No document files found in %s", documents_dir)
            return []
        results: list[IngestionResult] = []
        for i, path in enumerate(files):
            res = await ingest_one(pool, path, documents_dir, config)
            results.append(res)
            if progress_cb:
                progress_cb(i + 1, len(files))
        return results
    finally:
        await pool.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest documents into RAG-Advanced PostgreSQL (documents + chunks)."
    )
    parser.add_argument(
        "--documents",
        "-d",
        default=None,
        help="Path to documents folder (default: ./documents under repo root)",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Do not delete existing documents/chunks before ingestion",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Chunk size in characters (default: 1000)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Chunk overlap (default: 200)",
    )
    parser.add_argument(
        "--no-semantic",
        action="store_true",
        help="Disable Docling HybridChunker (use simple paragraph/size chunker only)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Max tokens per chunk for Docling HybridChunker (default: 512, same as all-rag-strategies)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    documents_dir = args.documents
    if not documents_dir:
        # Default: RAG-Advanced repo documents folder (self-contained)
        documents_dir = str(REPO_ROOT / "documents")
    documents_dir = str(Path(documents_dir).resolve())
    if not Path(documents_dir).exists():
        logger.error("Documents directory does not exist: %s", documents_dir)
        raise SystemExit(1)

    config = IngestionConfig(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        use_semantic_chunking=not args.no_semantic,
        max_tokens=args.max_tokens,
    )

    def progress(current: int, total: int) -> None:
        print(f"Progress: {current}/{total} documents")

    results = asyncio.run(
        run_ingestion(
            documents_dir,
            config,
            clean_before=not args.no_clean,
            progress_cb=progress,
        )
    )

    total_chunks = sum(r.chunks_created for r in results)
    total_errors = sum(len(r.errors) for r in results)
    print("\n" + "=" * 50)
    print("INGESTION SUMMARY")
    print("=" * 50)
    print(f"Documents processed: {len(results)}")
    print(f"Total chunks: {total_chunks}")
    print(f"Total errors: {total_errors}")
    for r in results:
        status = "OK" if not r.errors else "FAIL"
        print(f"  [{status}] {r.title}: {r.chunks_created} chunks")
        for err in r.errors:
            print(f"       {err}")


if __name__ == "__main__":
    main()
