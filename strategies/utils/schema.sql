-- =============================================================================
-- RAG-Advanced Base Database Schema
-- PostgreSQL with pgvector extension
-- =============================================================================

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- -----------------------------------------------------------------------------
-- Documents Table
-- Stores original documents before chunking
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

-- Index for source lookups
CREATE INDEX IF NOT EXISTS idx_documents_source ON documents(source);
CREATE INDEX IF NOT EXISTS idx_documents_created_at ON documents(created_at DESC);

-- -----------------------------------------------------------------------------
-- Chunks Table
-- Stores document chunks with embeddings for vector search
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS chunks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    embedding vector(1536),  -- text-embedding-3-small dimensions
    chunk_index INTEGER NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for efficient retrieval
CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON chunks USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- -----------------------------------------------------------------------------
-- Vector Search Function
-- Returns most similar chunks to a query embedding
-- -----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION match_chunks(
    query_embedding vector(1536),
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
-- Returns full document by ID
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
-- Automatically updates updated_at on document changes
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
-- Comments for Documentation
-- -----------------------------------------------------------------------------
COMMENT ON TABLE documents IS 'Original documents before chunking';
COMMENT ON TABLE chunks IS 'Document chunks with vector embeddings';
COMMENT ON FUNCTION match_chunks IS 'Vector similarity search for chunks';
COMMENT ON FUNCTION get_full_document IS 'Retrieve full document by ID';
