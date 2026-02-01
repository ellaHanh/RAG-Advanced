"""
Pydantic models for the ingestion pipeline.

Used for configuration, chunk representation, and ingestion results.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class IngestionConfig(BaseModel):
    """Configuration for document ingestion (aligned with all-rag-strategies ChunkingConfig)."""

    chunk_size: int = Field(default=1000, ge=100, le=5000, description="Target chunk size in characters")
    chunk_overlap: int = Field(default=200, ge=0, le=1000, description="Overlap between chunks")
    use_semantic_chunking: bool = Field(
        default=True,
        description="Use Docling HybridChunker when DoclingDocument is available",
    )
    max_tokens: int = Field(
        default=512,
        ge=128,
        le=2048,
        description="Max tokens per chunk for Docling HybridChunker (same as all-rag-strategies)",
    )

    @field_validator("chunk_overlap")
    @classmethod
    def overlap_less_than_chunk_size(cls, v: int, info: Any) -> int:
        """Ensure overlap is less than chunk size."""
        chunk_size = info.data.get("chunk_size", 1000)
        if v >= chunk_size:
            raise ValueError(f"chunk_overlap ({v}) must be < chunk_size ({chunk_size})")
        return v


class DocumentChunk(BaseModel):
    """A single chunk of a document, with optional embedding."""

    content: str = Field(..., description="Chunk text content")
    index: int = Field(..., ge=0, description="Chunk index within the document")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Chunk metadata")
    embedding: list[float] | None = Field(default=None, description="Embedding vector (1536 dims)")

    model_config = {"extra": "forbid"}


class IngestionResult(BaseModel):
    """Result of ingesting one document."""

    document_id: str = Field(..., description="UUID of the inserted document")
    title: str = Field(..., description="Document title")
    chunks_created: int = Field(..., ge=0, description="Number of chunks created")
    processing_time_ms: float = Field(..., ge=0, description="Processing time in milliseconds")
    errors: list[str] = Field(default_factory=list, description="Errors encountered")
