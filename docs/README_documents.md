# Documents Folder

The `documents/` folder is the **default source** for the RAG-Advanced ingestion pipeline. It holds the files that are chunked, embedded, and stored in PostgreSQL for retrieval.

---

## Purpose

- **Default path** when you run `python -m strategies.ingestion.ingest` without `--documents`; the pipeline discovers all supported files under `documents/`.
- **Example content** is included so ingestion works out of the box; files are copied from [all-rag-strategies/implementation/documents](https://github.com/coleam00/ottomator-agents/tree/main/all-rag-strategies/implementation/documents) so RAG-Advanced can run independently.

---

## Contents

Typical contents include:

- **Markdown** (e.g. `company-overview.md`, `mission-and-goals.md`, `team-handbook.md`, `implementation-playbook.md`) — read as plain text.
- **PDF, DOCX, etc.** — converted to markdown via Docling before chunking.
- **Audio** (MP3, WAV, etc.) — transcribed via Docling ASR (Whisper) before chunking.

You can add, remove, or replace files; the pipeline will process whatever is under the path you pass (default: `./documents`).

---

## Usage

```bash
# Ingest from default folder (documents/)
python -m strategies.ingestion.ingest

# Ingest from another folder
python -m strategies.ingestion.ingest --documents /path/to/your/docs
```

Supported types and pipeline steps are described in [README_ingestion.md](README_ingestion.md).
