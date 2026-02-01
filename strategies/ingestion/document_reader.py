"""
Document reading for ingestion.

Aligned with all-rag-strategies: Docling for PDF/DOCX/HTML (DocumentConverter),
Docling ASR (Whisper) for audio; plain text read directly. Returns (content, DoclingDocument)
for Docling formats so HybridChunker can be used.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Extensions that Docling can convert to document (for HybridChunker we get DoclingDocument).
DOCLING_DOCUMENT_EXT = {".pdf", ".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls", ".html", ".htm"}
# Plain text: read directly.
TEXT_EXT = {".md", ".markdown", ".txt"}
# Audio: transcribe via Docling ASR (optional).
AUDIO_EXT = {".mp3", ".wav", ".m4a", ".flac"}


def read_document(file_path: str) -> tuple[str, Any]:
    """
    Read a document file into text and optional DoclingDocument.

    Args:
        file_path: Absolute or relative path to the file.

    Returns:
        (content, docling_doc). content is markdown or plain text; docling_doc is
        a DoclingDocument for PDF/DOCX etc., or None for plain text/audio.
    """
    path = Path(file_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    ext = path.suffix.lower()

    if ext in AUDIO_EXT:
        content = _transcribe_audio(str(path))
        return (content, None)

    if ext in DOCLING_DOCUMENT_EXT:
        return _read_with_docling(str(path), ext)

    if ext in TEXT_EXT or ext == "":
        return _read_text(str(path))

    logger.warning("Unsupported extension %s, reading as text", ext)
    return _read_text(str(path))


def _read_text(file_path: str) -> tuple[str, None]:
    """Read plain text file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return (f.read(), None)
    except UnicodeDecodeError:
        with open(file_path, "r", encoding="latin-1") as f:
            return (f.read(), None)


def _read_with_docling(file_path: str, ext: str) -> tuple[str, Any]:
    """Convert PDF/DOCX etc. to markdown using Docling; return (markdown, DoclingDocument)."""
    try:
        from docling.document_converter import DocumentConverter

        logger.info("Converting %s file using Docling: %s", ext, os.path.basename(file_path))
        converter = DocumentConverter()
        result = converter.convert(file_path)
        markdown = result.document.export_to_markdown()
        logger.info("Successfully converted %s to markdown", os.path.basename(file_path))
        return (markdown, result.document)
    except Exception as e:
        logger.error("Failed to convert %s with Docling: %s", file_path, e)
        logger.warning("Falling back to raw text extraction for %s", file_path)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return (f.read(), None)
        except Exception:
            try:
                with open(file_path, "r", encoding="latin-1") as f:
                    return (f.read(), None)
            except Exception:
                return (f"[Error: Could not read file {os.path.basename(file_path)}]", None)


def _transcribe_audio(file_path: str) -> str:
    """Transcribe audio file using Docling ASR (Whisper Turbo, same as all-rag-strategies)."""
    try:
        from pathlib import Path as P

        from docling.document_converter import DocumentConverter, AudioFormatOption
        from docling.datamodel.pipeline_options import AsrPipelineOptions
        from docling.datamodel import asr_model_specs
        from docling.datamodel.base_models import InputFormat
        from docling.pipeline.asr_pipeline import AsrPipeline

        audio_path = P(file_path).resolve()
        if not audio_path.exists():
            return f"[Error: Audio file not found {audio_path.name}]"
        logger.info("Transcribing audio file using Whisper Turbo: %s", audio_path.name)
        opts = AsrPipelineOptions()
        opts.asr_options = asr_model_specs.WHISPER_TURBO
        converter = DocumentConverter(
            format_options={
                InputFormat.AUDIO: AudioFormatOption(
                    pipeline_cls=AsrPipeline,
                    pipeline_options=opts,
                )
            }
        )
        result = converter.convert(audio_path)
        logger.info("Successfully transcribed %s", os.path.basename(file_path))
        return result.document.export_to_markdown()
    except Exception as e:
        logger.error("Failed to transcribe %s with Whisper ASR: %s", file_path, e)
        return f"[Error: Could not transcribe {os.path.basename(file_path)}]"


def extract_title(content: str, file_path: str) -> str:
    """Extract title from first markdown heading or filename."""
    for line in content.split("\n")[:15]:
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip()
    return os.path.splitext(os.path.basename(file_path))[0]
