-- FinSight: PostgreSQL Extensions Initialization
-- This runs automatically on first docker-compose up

-- TimescaleDB for time-series optimization
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- pgvector for embedding storage and similarity search
CREATE EXTENSION IF NOT EXISTS vector;

-- pg_trgm for text search (BM25-like keyword matching)
CREATE EXTENSION IF NOT EXISTS pg_trgm;
