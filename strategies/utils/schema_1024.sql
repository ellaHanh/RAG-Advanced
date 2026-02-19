-- =============================================================================
-- RAG-Advanced Schema for 1024-dim Embeddings (BGE-M3 / local CPU)
-- PostgreSQL with pgvector extension
-- =============================================================================
-- Use this schema when EMBEDDING_BACKEND=bge-m3 (BAAI/bge-m3, 1024 dims).
-- For OpenAI (1536 dims) use schema.sql instead.
-- =============================================================================

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- -----------------------------------------------------------------------------
-- Documents Table
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    title TEXT NOT NULL,
    source TEXT NOT NULL,
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_documents_source ON documents(source);
CREATE INDEX IF NOT EXISTS idx_documents_created_at ON documents(created_at DESC);

-- -----------------------------------------------------------------------------
-- Chunks Table (vector(1024) for BGE-M3)
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS chunks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    embedding vector(1024),
    chunk_index INTEGER NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
-- IVFFlat lists should be ~sqrt(rows).  For small datasets (<1000 rows),
-- lists=10 is safe.  Increase for production (e.g. lists=100 for ~10k rows,
-- lists=1000 for ~1M rows).  Too-high lists on few rows causes empty Voronoi
-- cells, making default probes=1 miss data.
CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON chunks USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 10);

-- -----------------------------------------------------------------------------
-- Vector Search Function (1024-dim)
-- -----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION match_chunks(
    query_embedding vector(1024),
    match_count INTEGER DEFAULT 5
)
RETURNS TABLE (
    id UUID,
    document_id UUID,
    content TEXT,
    metadata JSONB,
    title TEXT,
    source TEXT,
    similarity FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        c.id,
        c.document_id,
        c.content,
        c.metadata,
        d.title,
        d.source,
        1 - (c.embedding <=> query_embedding) AS similarity
    FROM chunks c
    JOIN documents d ON c.document_id = d.id
    ORDER BY c.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- -----------------------------------------------------------------------------
-- Full Document Retrieval Function
-- -----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION get_full_document(doc_id UUID)
RETURNS TABLE (
    id UUID,
    title TEXT,
    source TEXT,
    content TEXT,
    metadata JSONB
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        d.id,
        d.title,
        d.source,
        d.content,
        d.metadata
    FROM documents d
    WHERE d.id = doc_id;
END;
$$;

-- -----------------------------------------------------------------------------
-- Update Timestamp Trigger
-- -----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE TRIGGER update_documents_updated_at
    BEFORE UPDATE ON documents
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- -----------------------------------------------------------------------------
-- Comments
-- -----------------------------------------------------------------------------
COMMENT ON TABLE documents IS 'Original documents before chunking';
COMMENT ON TABLE chunks IS 'Document chunks with 1024-dim embeddings (BGE-M3)';
COMMENT ON FUNCTION match_chunks IS 'Vector similarity search for 1024-dim embeddings';
