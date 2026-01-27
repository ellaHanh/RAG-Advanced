"""
Document Ingestion Pipeline.

Process and embed documents for RAG retrieval.

Components:
    - ingest: Main ingestion entry point
    - chunker: Docling HybridChunker for semantic chunking
    - embedder: OpenAI text-embedding-3-small wrapper
    - contextual_enrichment: Anthropic contextual retrieval
"""
