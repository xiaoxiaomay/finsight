-- FinSight Database Schema
-- Runs automatically on first docker-compose up (after extensions)

-- ============================================================
-- 1. Market Data (TimescaleDB hypertable for OHLCV)
-- ============================================================
CREATE TABLE IF NOT EXISTS market_data (
    symbol      VARCHAR(20)  NOT NULL,
    date        DATE         NOT NULL,
    open        NUMERIC(12,4),
    high        NUMERIC(12,4),
    low         NUMERIC(12,4),
    close       NUMERIC(12,4),
    adj_close   NUMERIC(12,4),
    volume      BIGINT,
    source      VARCHAR(50)  DEFAULT 'yfinance',
    ingested_at TIMESTAMPTZ  DEFAULT NOW(),
    PRIMARY KEY (symbol, date)
);

SELECT create_hypertable('market_data', 'date', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_market_data_symbol ON market_data (symbol, date DESC);

-- ============================================================
-- 2. Fundamentals (Point-in-Time to prevent look-ahead bias)
-- ============================================================
CREATE TABLE IF NOT EXISTS fundamentals (
    symbol                VARCHAR(20)  NOT NULL,
    report_date           DATE         NOT NULL,  -- Period end date
    filing_date           DATE         NOT NULL,  -- When actually available (point-in-time)
    period_type           VARCHAR(10)  NOT NULL,  -- 'annual' or 'quarterly'
    revenue               NUMERIC(15,2),
    cost_of_revenue       NUMERIC(15,2),
    gross_profit          NUMERIC(15,2),
    operating_income      NUMERIC(15,2),
    net_income            NUMERIC(15,2),
    total_assets          NUMERIC(15,2),
    total_liabilities     NUMERIC(15,2),
    total_equity          NUMERIC(15,2),
    operating_cash_flow   NUMERIC(15,2),
    capital_expenditure   NUMERIC(15,2),
    free_cash_flow        NUMERIC(15,2),
    eps                   NUMERIC(10,4),
    book_value_per_share  NUMERIC(10,4),
    shares_outstanding    BIGINT,
    dividend_per_share    NUMERIC(10,4),
    source                VARCHAR(50)  DEFAULT 'fmp',
    ingested_at           TIMESTAMPTZ  DEFAULT NOW(),
    PRIMARY KEY (symbol, report_date, period_type)
);

CREATE INDEX IF NOT EXISTS idx_fundamentals_filing ON fundamentals (symbol, filing_date DESC);

-- ============================================================
-- 3. Macro Indicators (FRED data)
-- ============================================================
CREATE TABLE IF NOT EXISTS macro_indicators (
    indicator_id VARCHAR(50)  NOT NULL,  -- e.g., 'GDP', 'CPI', 'FEDFUNDS'
    date         DATE         NOT NULL,
    value        NUMERIC(15,4),
    source       VARCHAR(50)  DEFAULT 'FRED',
    ingested_at  TIMESTAMPTZ  DEFAULT NOW(),
    PRIMARY KEY (indicator_id, date)
);

-- ============================================================
-- 4. Factor Signals (precomputed cross-sectional signals)
-- ============================================================
CREATE TABLE IF NOT EXISTS factor_signals (
    symbol      VARCHAR(20)  NOT NULL,
    date        DATE         NOT NULL,
    factor_name VARCHAR(50)  NOT NULL,
    raw_value   NUMERIC(15,6),
    z_score     NUMERIC(8,4),
    percentile  NUMERIC(5,2),
    PRIMARY KEY (symbol, date, factor_name)
);

SELECT create_hypertable('factor_signals', 'date', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_factor_signals_lookup
    ON factor_signals (factor_name, date DESC, symbol);

-- ============================================================
-- 5. Portfolio Positions
-- ============================================================
CREATE TABLE IF NOT EXISTS portfolio_positions (
    id           SERIAL       PRIMARY KEY,
    date         DATE         NOT NULL,
    symbol       VARCHAR(20)  NOT NULL,
    shares       NUMERIC(12,4),
    avg_cost     NUMERIC(12,4),
    market_value NUMERIC(15,2),
    weight       NUMERIC(6,4),
    strategy     VARCHAR(50)  NOT NULL DEFAULT 'multi_factor',
    UNIQUE (date, symbol, strategy)
);

CREATE INDEX IF NOT EXISTS idx_positions_date ON portfolio_positions (strategy, date DESC);

-- ============================================================
-- 6. Trades
-- ============================================================
CREATE TABLE IF NOT EXISTS trades (
    id            SERIAL       PRIMARY KEY,
    date          DATE         NOT NULL,
    symbol        VARCHAR(20)  NOT NULL,
    side          VARCHAR(4)   NOT NULL CHECK (side IN ('BUY', 'SELL')),
    shares        NUMERIC(12,4) NOT NULL,
    price         NUMERIC(12,4) NOT NULL,
    commission    NUMERIC(8,2) DEFAULT 0,
    strategy      VARCHAR(50)  NOT NULL DEFAULT 'multi_factor',
    signal_source VARCHAR(100)
);

CREATE INDEX IF NOT EXISTS idx_trades_date ON trades (strategy, date DESC);

-- ============================================================
-- 7. Portfolio Daily Snapshots (for performance tracking)
-- ============================================================
CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    date             DATE         NOT NULL,
    strategy         VARCHAR(50)  NOT NULL,
    total_value      NUMERIC(15,2),
    cash             NUMERIC(15,2),
    daily_return     NUMERIC(10,6),
    benchmark_return NUMERIC(10,6),
    PRIMARY KEY (date, strategy)
);

SELECT create_hypertable('portfolio_snapshots', 'date', if_not_exists => TRUE);

-- ============================================================
-- 8. Document Embeddings (pgvector for RAG)
-- ============================================================
CREATE TABLE IF NOT EXISTS document_embeddings (
    id             SERIAL       PRIMARY KEY,
    document_type  VARCHAR(50),   -- '10-K', '10-Q', 'earnings_call', 'news'
    symbol         VARCHAR(20),
    document_date  DATE,
    chunk_index    INT,
    chunk_text     TEXT,
    embedding      vector(1536),  -- Dimension depends on embedding model
    metadata       JSONB,
    created_at     TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_doc_embed_symbol ON document_embeddings (symbol, document_type);
CREATE INDEX IF NOT EXISTS idx_doc_embed_vector
    ON document_embeddings USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- ============================================================
-- 9. Data Quality Log (track ingestion issues)
-- ============================================================
CREATE TABLE IF NOT EXISTS data_quality_log (
    id          SERIAL       PRIMARY KEY,
    source      VARCHAR(50)  NOT NULL,
    check_name  VARCHAR(100) NOT NULL,
    status      VARCHAR(20)  NOT NULL CHECK (status IN ('pass', 'warn', 'fail')),
    details     JSONB,
    checked_at  TIMESTAMPTZ  DEFAULT NOW()
);

-- ============================================================
-- 10. News Articles (for sentiment analysis)
-- ============================================================
CREATE TABLE IF NOT EXISTS news_articles (
    id             SERIAL       PRIMARY KEY,
    symbol         VARCHAR(20),
    title          TEXT         NOT NULL,
    description    TEXT,
    url            TEXT,
    source_name    VARCHAR(100),
    published_at   TIMESTAMPTZ,
    sentiment      NUMERIC(5,3),  -- -1.0 to 1.0
    ingested_at    TIMESTAMPTZ  DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_news_symbol ON news_articles (symbol, published_at DESC);

-- ============================================================
-- 11. Research Reports (generated by AI agents)
-- ============================================================
CREATE TABLE IF NOT EXISTS research_reports (
    id              SERIAL       PRIMARY KEY,
    symbol          VARCHAR(20)  NOT NULL,
    report_type     VARCHAR(50)  DEFAULT 'full_analysis',
    recommendation  VARCHAR(20),
    conviction      VARCHAR(10),
    report_data     JSONB        NOT NULL,  -- Full structured report
    generated_at    TIMESTAMPTZ  DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_reports_symbol ON research_reports (symbol, generated_at DESC);
