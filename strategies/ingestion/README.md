# Ingestion Pipeline

<!-- TOC: Quickstart, Features, Usage, Dependencies, Related -->

## Quickstart

```bash
# From repo root, venv activated; default path ./documents
python -m strategies.ingestion.ingest

# Custom path
python -m strategies.ingestion.ingest --documents /path/to/documents
```

Requires `DATABASE_URL` and `OPENAI_API_KEY`; schema applied (e.g. `python scripts/run_schema.py` or Docker).

## Features

- **Discover** — Recursively find supported files (MD, PDF, DOCX, audio).
- **Read** — Plain text or Docling (convert/transcribe).
- **Chunk** — Docling HybridChunker (token-based) or simple sliding-window; `--chunk-size`, `--chunk-overlap`, `--max-tokens`, `--no-semantic`.
- **Embed** — OpenAI `text-embedding-3-small` (1536 dims) via `strategies.utils.embedder.embed_documents`.
- **Store** — `documents` and `chunks` in PostgreSQL (`strategies/utils/schema.sql`).

| Type     | Extensions | Handling                    |
|----------|------------|-----------------------------|
| Text     | `.md`, `.txt` | UTF-8 read                  |
| Documents| `.pdf`, `.docx`, etc. | Docling → markdown |
| Audio    | `.mp3`, `.wav`, `.m4a`, `.flac` | Docling ASR (Whisper) |

## Usage

```bash
# Append (no clean)
python -m strategies.ingestion.ingest --documents ./my-docs --no-clean

# Tune chunking
python -m strategies.ingestion.ingest --chunk-size 800 --chunk-overlap 150 --max-tokens 512

# Simple chunker only
python -m strategies.ingestion.ingest --no-semantic --verbose
```

Programmatic:

```python
from strategies.ingestion import run_ingestion, IngestionConfig

config = IngestionConfig(chunk_size=1000, chunk_overlap=200, use_semantic_chunking=True, max_tokens=512)
results = await run_ingestion("/path/to/documents", config, clean_before=True)
```

## Dependencies

- Project: `pip install -e .` (includes `docling` for PDF/DOCX/audio).
- Env: `DATABASE_URL`, `OPENAI_API_KEY`.

## Related

- [Root README](../../README.md)
- [documents/](../../documents/README.md) — Default ingestion folder
- [evaluation/](../../evaluation/README.md) — Metrics and benchmarks
