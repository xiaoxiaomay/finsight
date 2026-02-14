# Data Dictionary

## Database Tables

### market_data

Daily OHLCV price data. TimescaleDB hypertable partitioned by date.

| Column | Type | Description | Source |
|--------|------|-------------|--------|
| symbol | VARCHAR(10) | Stock ticker (e.g., AAPL) | yfinance |
| date | DATE | Trading date | yfinance |
| open | NUMERIC | Opening price | yfinance |
| high | NUMERIC | Intraday high | yfinance |
| low | NUMERIC | Intraday low | yfinance |
| close | NUMERIC | Closing price | yfinance |
| adj_close | NUMERIC | Split/dividend adjusted close | yfinance |
| volume | BIGINT | Trading volume | yfinance |
| source | VARCHAR(20) | Data source identifier | System |
| updated_at | TIMESTAMPTZ | Last update time | System |

**Primary Key:** (symbol, date)
**Update Frequency:** Daily at market close (~18:00 ET)
**History:** 2020-01-01 to present

---

### fundamentals

Company financial statements. Point-in-time enforced (filing_date >= report_date).

| Column | Type | Description | Source |
|--------|------|-------------|--------|
| symbol | VARCHAR(10) | Stock ticker | FMP API |
| filing_date | DATE | SEC filing date (when public) | FMP API |
| report_date | DATE | Fiscal period end date | FMP API |
| period_type | VARCHAR(10) | 'annual' or 'quarterly' | FMP API |
| revenue | NUMERIC | Total revenue | Income stmt |
| net_income | NUMERIC | Net income | Income stmt |
| eps | NUMERIC | Earnings per share | Income stmt |
| gross_profit | NUMERIC | Gross profit | Income stmt |
| operating_income | NUMERIC | Operating income | Income stmt |
| total_assets | NUMERIC | Total assets | Balance sheet |
| total_liabilities | NUMERIC | Total liabilities | Balance sheet |
| total_equity | NUMERIC | Shareholders' equity | Balance sheet |
| book_value_per_share | NUMERIC | Equity / shares | Derived |
| shares_outstanding | BIGINT | Diluted shares | Income stmt |
| operating_cash_flow | NUMERIC | Cash from operations | Cash flow |
| capital_expenditure | NUMERIC | CapEx | Cash flow |
| free_cash_flow | NUMERIC | OCF - CapEx | Derived |
| source | VARCHAR(20) | Data source identifier | System |
| updated_at | TIMESTAMPTZ | Last update time | System |

**Primary Key:** (symbol, filing_date, period_type)
**Update Frequency:** Quarterly (as filings are released)
**PIT Rule:** Fundamentals only visible after filing_date, preventing look-ahead bias

---

### macro_indicators

Economic and market indicators from Federal Reserve (FRED).

| Column | Type | Description | Source |
|--------|------|-------------|--------|
| indicator_id | VARCHAR(20) | FRED series ID (e.g., GDP, UNRATE) | FRED |
| date | DATE | Observation date | FRED |
| value | NUMERIC | Indicator value | FRED |
| source | VARCHAR(20) | Always 'FRED' | System |
| updated_at | TIMESTAMPTZ | Last update time | System |

**Primary Key:** (indicator_id, date)
**Update Frequency:** Daily (most indicators monthly/quarterly)

**Key Indicators:**

| Series ID | Name | Frequency |
|-----------|------|-----------|
| GDP | Real GDP | Quarterly |
| UNRATE | Unemployment Rate | Monthly |
| CPIAUCSL | CPI (All Urban) | Monthly |
| FEDFUNDS | Federal Funds Rate | Daily |
| DGS10 | 10-Year Treasury | Daily |
| T10Y2Y | 10Y-2Y Spread | Daily |
| VIXCLS | VIX Index | Daily |
| SP500 | S&P 500 | Daily |
| UMCSENT | Consumer Sentiment | Monthly |
| INDPRO | Industrial Production | Monthly |

---

### factor_signals

Cross-sectional factor scores for all symbols.

| Column | Type | Description | Source |
|--------|------|-------------|--------|
| symbol | VARCHAR(10) | Stock ticker | Computed |
| date | DATE | Signal date | Computed |
| factor_name | VARCHAR(30) | Factor identifier | Computed |
| raw_value | NUMERIC | Raw factor value | Factor engine |
| z_score | NUMERIC | Cross-sectional z-score | Factor engine |
| percentile | NUMERIC | Cross-sectional percentile (0-100) | Factor engine |

**Primary Key:** (symbol, date, factor_name)
**Update Frequency:** Daily

**Factor Universe:**

| Factor | Category | Description |
|--------|----------|-------------|
| momentum_12_1 | Momentum | 12-month return, skip last month |
| short_term_reversal | Momentum | Past 1-month return (negated) |
| earnings_yield | Value | E/P ratio |
| book_to_market | Value | Book value / Market cap |
| ev_ebitda | Value | EV/EBITDA (negated, lower = cheaper) |
| roe | Quality | Return on equity |
| gross_profitability | Quality | Gross profit / Total assets |
| accruals | Quality | (NI - OCF) / Assets (negated) |
| asset_growth | Quality | YoY asset growth (negated) |
| volatility_60d | Low Vol | 60-day realized vol (negated) |

---

### sec_filings

SEC filing metadata and text for RAG pipeline.

| Column | Type | Description | Source |
|--------|------|-------------|--------|
| id | SERIAL | Auto-increment ID | System |
| symbol | VARCHAR(10) | Stock ticker | EDGAR |
| filing_type | VARCHAR(10) | '10-K' or '10-Q' | EDGAR |
| filing_date | DATE | Filed with SEC | EDGAR |
| fiscal_year | INTEGER | Fiscal year | EDGAR |
| text_content | TEXT | Full filing text | EDGAR |
| embedding | VECTOR(384) | all-MiniLM-L6-v2 embedding | Computed |

**Embedding Model:** sentence-transformers/all-MiniLM-L6-v2 (384 dimensions)
**Retrieval:** Hybrid — cosine similarity + BM25, fused via RRF

---

### data_quality_log

Results from automated quality checks.

| Column | Type | Description |
|--------|------|-------------|
| id | SERIAL | Auto-increment ID |
| check_name | VARCHAR(50) | Check identifier |
| status | VARCHAR(10) | 'pass', 'warning', 'fail' |
| details | TEXT | Human-readable result |
| checked_at | TIMESTAMPTZ | When the check ran |
| duration_ms | INTEGER | Check execution time |

**Quality Checks:**

| Check | Table | Rule |
|-------|-------|------|
| market_data_coverage | market_data | Symbol count and total rows |
| market_data_freshness | market_data | Latest date within 5 trading days |
| market_data_gaps | market_data | No symbol missing > 5 consecutive days |
| fundamentals_pit_integrity | fundamentals | All filing_date >= report_date |
| fundamentals_coverage | fundamentals | Symbol and period coverage |
| macro_coverage | macro_indicators | Indicator count and observations |
| macro_freshness | macro_indicators | No indicator > 30 days stale |

---

## File-Based Data

| Path | Format | Description |
|------|--------|-------------|
| reports/*.json | JSON | AI research reports |
| reports/*.html | HTML | Formatted research reports |
| reports/tearsheet.html | HTML | Performance tearsheet |
| data/snapshots/*.json | JSON | Daily portfolio signal snapshots |
| data/trades/paper_trades.jsonl | JSONL | Paper trade log |
