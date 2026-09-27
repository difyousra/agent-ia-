-- PostgreSQL + pgvector schema for vendor contract RAG
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS contract_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    contract_id TEXT NOT NULL,
    document_name TEXT NOT NULL,
    clause_number TEXT,
    clause_type TEXT,
    section TEXT,
    page INTEGER,
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    embedding VECTOR(1024),
    search_vector TSVECTOR,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT contract_chunks_hash_unique UNIQUE (contract_id, content_hash)
);

CREATE INDEX IF NOT EXISTS contract_embedding_idx
    ON contract_chunks
    USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS contract_search_idx
    ON contract_chunks
    USING GIN (search_vector);

CREATE INDEX IF NOT EXISTS contract_chunks_contract_id_idx
    ON contract_chunks (contract_id);

CREATE INDEX IF NOT EXISTS contract_chunks_document_name_idx
    ON contract_chunks (document_name);

CREATE OR REPLACE FUNCTION contract_chunks_search_vector_update()
RETURNS TRIGGER AS $$
BEGIN
    NEW.search_vector :=
        setweight(to_tsvector('english', coalesce(NEW.clause_number, '')), 'A') ||
        setweight(to_tsvector('english', coalesce(NEW.clause_type, '')), 'A') ||
        setweight(to_tsvector('english', coalesce(NEW.section, '')), 'B') ||
        setweight(to_tsvector('english', coalesce(NEW.document_name, '')), 'B') ||
        setweight(to_tsvector('english', coalesce(NEW.content, '')), 'C');
    NEW.updated_at := NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_contract_chunks_search_vector ON contract_chunks;
CREATE TRIGGER trg_contract_chunks_search_vector
    BEFORE INSERT OR UPDATE OF clause_number, clause_type, section, document_name, content
    ON contract_chunks
    FOR EACH ROW
    EXECUTE PROCEDURE contract_chunks_search_vector_update();
