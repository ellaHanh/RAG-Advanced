# Ingestion Pipeline

The ingestion pipeline reads documents from a folder, chunks them, generates embeddings (OpenAI), and stores documents and chunks in PostgreSQL. It is aligned with [all-rag-strategies](https://github.com/coleam00/ottomator-agents/tree/main/all-rag-strategies): same file types, Docling for conversion and chunking (DocumentConverter, HybridChunker, Whisper ASR).

## Requirements

- **Environment**: `DATABASE_URL` and `OPENAI_API_KEY` in `.env` or the environment.
- **Database**: Schema applied (e.g. `python scripts/run_schema.py` or Docker Compose).
- **Dependencies**: Project installed (`pip install -e .`), including `docling` for PDF/DOCX/audio.

## Usage

From the RAG-Advanced repo root (with venv activated):

```bash
# Ingest from default path: ./documents (under repo root)
python -m strategies.ingestion.ingest

# Ingest from a specific folder
python -m strategies.ingestion.ingest --documents /path/to/your/documents

# Append to existing data (do not delete documents/chunks first)
python -m strategies.ingestion.ingest --documents ./my-docs --no-clean

# Tune chunking (character-based for simple chunker; max_tokens for Docling HybridChunker)
python -m strategies.ingestion.ingest --chunk-size 800 --chunk-overlap 150 --max-tokens 512

# Disable Docling HybridChunker (use simple paragraph/size chunker only)
python -m strategies.ingestion.ingest --no-semantic --verbose
```

## Supported file types

| Type        | Extensions              | How it’s handled                          |
|------------|-------------------------|-------------------------------------------|
| Text       | `.md`, `.markdown`, `.txt` | Read as UTF-8 (or latin-1 fallback).      |
| Documents  | `.pdf`, `.docx`, `.doc`, `.pptx`, `.ppt`, `.xlsx`, `.xls`, `.html`, `.htm` | Converted to markdown via Docling.        |
| Audio      | `.mp3`, `.wav`, `.m4a`, `.flac` | Transcribed via Docling ASR (Whisper).    |

## Pipeline steps

1. **Discover** – Recursively find supported files under the given documents directory.
2. **Read** – Plain text is read directly; PDF/DOCX/audio are processed with Docling (convert or transcribe).
3. **Chunk** – Simple paragraph/size-based chunking, or Docling HybridChunker when a Docling document is available and `--no-semantic` is not set.
4. **Embed** – Batch embedding with OpenAI `text-embedding-3-small` (1536 dims) via `strategies.utils.embedder.embed_documents`.
5. **Store** – One row in `documents` (title, source, content, metadata) and multiple rows in `chunks` (content, embedding, chunk_index, metadata). Schema: `strategies/utils/schema.sql`.

## Default documents path

If `--documents` is omitted, the pipeline uses **`./documents`** under the RAG-Advanced repo root. This keeps the project self-contained. The repo includes a `documents/` folder with example files (copied from all-rag-strategies) so ingestion works out of the box.

## Alignment with all-rag-strategies

| Area | RAG-Advanced | all-rag-strategies |
|------|-------------|--------------------|
| **Document conversion** | Docling `DocumentConverter` → markdown + `DoclingDocument` | Same |
| **Audio** | Docling ASR (Whisper Turbo) | Same |
| **Chunking** | Docling `HybridChunker` when `DoclingDocument` is available; tokenizer `sentence-transformers/all-MiniLM-L6-v2`, `max_tokens` configurable (default 512), `merge_peers=True` | Same model and options |
| **Fallback** | Simple paragraph/size chunker when no Docling doc or `--no-semantic` | SimpleChunker or simple fallback in HybridChunker |
| **Config** | `IngestionConfig`: chunk_size, chunk_overlap, use_semantic_chunking, max_tokens | `ChunkingConfig`: chunk_size, chunk_overlap, max_tokens, use_semantic_splitting |

**Intentional differences**

- **Config model**: RAG-Advanced uses Pydantic `IngestionConfig`; all-rag-strategies uses dataclass `ChunkingConfig`. Same semantics (chunk_size, overlap, max_tokens, semantic flag).
- **Chunker API**: RAG-Advanced uses a single function `chunk_document(content, ..., docling_doc=...)` that picks HybridChunker or simple internally; all-rag-strategies uses a class `DoclingHybridChunker` with async `chunk_document()`. We use sync chunking and cache the HybridChunker per `max_tokens` so tokenizer is loaded once per run.
- **Contextual enrichment**: all-rag-strategies supports optional Anthropic contextual retrieval (one LLM call per chunk). RAG-Advanced does not implement this yet; add it if needed for parity.

## Programmatic use

```python
from strategies.ingestion import run_ingestion, IngestionConfig

config = IngestionConfig(
    chunk_size=1000,
    chunk_overlap=200,
    use_semantic_chunking=True,
    max_tokens=512,
)
results = await run_ingestion(
    "/path/to/documents",
    config,
    clean_before=True,
    progress_cb=lambda i, total: print(f"{i}/{total}"),
)
for r in results:
    print(r.title, r.chunks_created, r.document_id)
```

## Components

| Module                  | Role                                                                 |
|-------------------------|----------------------------------------------------------------------|
| `strategies/ingestion/ingest.py`   | CLI and `run_ingestion()`; find files, read, chunk, embed, save.     |
| `strategies/ingestion/document_reader.py` | Read/convert files; extract title.                          |
| `strategies/ingestion/chunker.py`  | Simple or Docling HybridChunker.                                     |
| `strategies/ingestion/embedder.py` | Batch embed chunks via `strategies.utils.embedder.embed_documents`.   |
| `strategies/ingestion/models.py`   | `IngestionConfig`, `DocumentChunk`, `IngestionResult`.               |

## Tests

Unit tests (no DB or OpenAI) live in `tests/test_strategies/test_ingestion.py`:

```bash
pytest tests/test_strategies/test_ingestion.py -v
```

Run with the full project environment (e.g. `pip install -e .`) so all imports resolve.
