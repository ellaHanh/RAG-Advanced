"""
Document Ingestion Pipeline.

Process and embed documents for RAG retrieval.

Components:
    - ingest: Main pipeline and CLI (python -m strategies.ingestion.ingest)
    - chunker: Simple or Docling HybridChunker
    - document_reader: Read .md, .txt, PDF, DOCX, audio (Docling)
    - embedder: Batch embeddings via strategies.utils.embedder
    - models: IngestionConfig, DocumentChunk, IngestionResult
"""

from strategies.ingestion.models import DocumentChunk, IngestionConfig, IngestionResult

__all__ = [
    "run_ingestion",
    "IngestionConfig",
    "DocumentChunk",
    "IngestionResult",
]


def __getattr__(name: str):  # noqa: D107
    if name == "run_ingestion":
        from strategies.ingestion.ingest import run_ingestion
        return run_ingestion
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
