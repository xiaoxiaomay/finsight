# FinSight — AI-Powered Investment Research & Quantitative Analysis Platform

## Project Design Document v2.0

> **Purpose**: This document serves as the master blueprint for Claude Code to execute the FinSight project. It covers architecture, implementation details, phased delivery plan, and quality standards. The project has three goals: (1) serve as a compelling portfolio project for job applications in Vancouver's finance/AI sector, (2) function as a real investment system with verifiable returns, and (3) lay groundwork for potential future commercialization.
>
> **v2.0 Changes**: Added Claude API setup guide, upgraded Quant Engine with 3-tier factor library + ML pipeline + factor neutralization + regime detection, upgraded AI Agent System with LangGraph supervisor pattern + agentic RAG + tool-using agents, added Proof of Performance section with live track record design, added Interview Preparation section.

---

## 0. Prerequisites: Claude API Setup Guide

The AI Agent system requires a Claude API key. Here is how to get one:

### Step 1: Create Anthropic Developer Account
1. Go to **console.anthropic.com** (注意：这和你日常使用的 claude.ai 是不同的账号系统)
2. Click "Start Building" or "Sign Up"
3. You can use Google account or email to register
4. Even if you already have a paid Claude.ai subscription, you still need a separate developer account

### Step 2: Generate API Key
1. Login to console.anthropic.com
2. In the left sidebar, click **"API Keys"**
3. Click **"+Create Key"** in the top right
4. Name it something descriptive like `finsight-project`
5. **CRITICAL**: Copy the key immediately — it is shown only once. Store it securely (e.g., password manager)
6. The key format looks like: `sk-ant-api03-...`

### Step 3: Add Credits (API is Pay-As-You-Go)
1. In the left sidebar, click **"Billing"**
2. Add a credit card and purchase credits
3. **Start with $5** — this is more than enough for initial development and testing
4. Optionally set up auto-reload (e.g., reload $10 when balance drops below $2)

### Cost Estimate for FinSight
| Usage | Model | Estimated Monthly Cost |
|-------|-------|----------------------|
| Agent research reports (20/month) | Claude Sonnet 4.5 | ~$3-5 |
| RAG document analysis | Claude Sonnet 4.5 | ~$2-3 |
| Interactive chat queries | Claude Sonnet 4.5 | ~$2-3 |
| Embedding generation | Voyage/local model | ~$1-2 |
| **Total estimated** | | **~$10-15/month** |

### Step 4: Test the Key
```bash
curl https://api.anthropic.com/v1/messages \
  -H "x-api-key: YOUR_API_KEY_HERE" \
  -H "anthropic-version: 2023-06-01" \
  -H "content-type: application/json" \
  -d '{
    "model": "claude-sonnet-4-5-20250929",
    "max_tokens": 256,
    "messages": [{"role": "user", "content": "Say hello in one sentence."}]
  }'
```

### Step 5: Store the Key for FinSight
Add to your `.env` file:
```bash
ANTHROPIC_API_KEY=sk-ant-api03-your-key-here
```
Never commit `.env` to git. The `.gitignore` file will exclude it.

---

## 1. Market Context & Strategic Positioning

### 1.1 Vancouver Finance + AI Landscape (as of early 2026)

Vancouver's financial AI ecosystem has several key players and trends that this project must align with:

**Key Employers & What They Value:**

- **Connor, Clark & Lunn (CC&L)** — Vancouver's largest quant firm, managing $167B+ in assets. Their Quantitative Equity team actively hires Quant Developers ($90K-$200K) and Quant Research Analysts. They use machine learning, cloud computing, optimization engines, and non-traditional high-performance databases. They explicitly value "interest in investment management" and collaborative research. CC&L also has a Data Engineer role focused on "pipelines, governance controls, and semantic context that power enterprise-wide AI and analytics."
- **Block/Square** — Major Vancouver tech employer with fintech focus. Hiring ML Engineers, Data Engineers. Uses Airflow, Databricks, dbt, Snowflake, Python stack.
- **Brex** — AI-powered spend platform with Vancouver office. Hiring data engineers with Airflow/Databricks/Snowflake stack.
- **Motive** — Hiring AI Operations Product Managers and AI engineers in Vancouver.
- **RBC, BCAA, Nicola Wealth** — Traditional finance with growing AI teams. Looking for data scientists, risk analysts, quantitative analysts.

**Industry Trends Shaping Hiring:**

1. **Agentic RAG is the hottest topic in enterprise AI** — The RAG market is projected to grow from $1.9B (2025) to $9.9B (2030) at 38.4% CAGR. Finance is a top adoption sector. Employers want people who can build production RAG systems, not just toy demos.
2. **Multi-agent systems are moving from research to production** — Deloitte predicts 25% of GenAI-using companies will launch agentic pilots by 2025, growing to 50% by 2027. Financial firms are early adopters for compliance, research, and customer support.
3. **Data engineering is the foundation** — Every Vancouver fintech job listing emphasizes: Python, SQL, Airflow/Prefect, Snowflake/Databricks, dbt, cloud infrastructure. This is table stakes.
4. **ML in finance is maturing** — Not just about building models, but about MLOps, model validation, performance monitoring, and explainability.
5. **Domain expertise differentiates** — Companies explicitly say "interest in investment management is key." A portfolio project that demonstrates genuine financial understanding (not just generic ML) stands out.

### 1.2 Project Positioning for Job Applications

This project should demonstrate EXACTLY the skills that Vancouver financial AI employers are looking for:

| Employer Need | How FinSight Demonstrates It |
|---|---|
| Data engineering (pipelines, governance) | Automated multi-source financial data pipeline with scheduling, validation, and lineage tracking |
| Quantitative research rigor | Factor analysis framework with proper backtesting methodology (no look-ahead bias, walk-forward validation) |
| Modern AI engineering (RAG, Agents) | Multi-agent research system with financial document RAG, real-time data retrieval, and structured output generation |
| ML in production | End-to-end ML pipeline: feature engineering → training → validation → monitoring → retraining triggers |
| Software engineering quality | Clean architecture, comprehensive tests, CI/CD, Docker containerization, proper documentation |
| Domain knowledge in finance | Real understanding of equity markets, factor investing, risk management, portfolio construction |
| Product thinking | Polished dashboard, user-facing research reports, intuitive workflow design |

### 1.3 Key Resume Talking Points (Design for These)

When presenting this project to employers, the candidate should be able to articulate these specific stories:

1. **"I built an end-to-end financial data platform"** — Multi-source ingestion (market data, fundamentals, SEC filings, macro indicators), automated quality checks, temporal consistency handling (point-in-time data to prevent look-ahead bias), and time-series optimized storage. *This maps to CC&L's Data Engineer role.*

2. **"I designed a multi-agent AI research system for financial analysis"** — Not a toy chatbot, but a production-grade system where specialized agents (earnings analysis, macro assessment, news sentiment, peer comparison) coordinate through an orchestrator to produce institutional-quality research reports. Used Agentic RAG with financial document retrieval, structured output validation, and citation tracking. *This maps to every AI Engineer role in Vancouver.*

3. **"I implemented a rigorous quantitative research framework"** — Factor-based equity selection with walk-forward validation, transaction cost modeling, and proper statistical testing (not just "my backtest shows 30% returns"). Can explain why most backtests are wrong and how to build ones that aren't. *This maps to CC&L's Quant Research Analyst role.*

4. **"I have a live track record"** — The system runs with real capital (even if small, in a TFSA), producing verifiable performance metrics. Can show equity curve, drawdown analysis, and attribution against benchmarks. *This proves the system isn't just theoretical.*

5. **"I built it with production-grade engineering practices"** — Docker containerized, scheduled with cron/Airflow, monitored with alerts, documented with architecture diagrams, tested with unit and integration tests. The codebase is clean enough to onboard another developer. *This maps to senior engineer expectations.*

6. **"I understand the full investment pipeline"** — From data collection to signal generation to portfolio construction to execution to performance attribution. Can discuss trade-offs between alpha decay and transaction costs, factor crowding, regime changes. *This shows domain depth, not just technical ability.*

---

## 2. System Architecture

### 2.1 High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          FinSight Platform v2.0                           │
├──────────────┬───────────────┬───────────────────┬───────────────────────┤
│  Data Layer  │  Quant Engine │  AI Agent System   │  Dashboard &          │
│              │               │                    │  Performance Proof    │
├──────────────┼───────────────┼───────────────────┼───────────────────────┤
│ • Ingestion  │ • Tier 1:     │ • Supervisor       │ • Streamlit UI        │
│   (multi-src)│   Classic     │   (LangGraph)      │ • Portfolio Overview  │
│ • TimescaleDB│   Factors     │ • Earnings Agent   │ • AI Research Chat    │
│ • Quality    │ • Tier 2:     │   [RAG + Tools]    │ • Backtest Lab        │
│   Validation │   ML-Enhanced │ • Macro Agent      │ • Risk Dashboard      │
│ • Scheduling │ • Factor      │   [FRED + Tools]   │ • Live Track Record ★│
│ • Point-in-  │   Neutralizer │ • News Agent       │ • Performance Proof   │
│   Time Data  │ • ML Pipeline │   [NLP + Tools]    │                       │
│              │ • Walk-Forward│ • Peer Agent       │                       │
│              │   Backtester  │ • Quant Signal     │                       │
│              │ • Portfolio   │   Agent [DB Tools] │                       │
│              │   Optimizer   │ • Agentic RAG      │                       │
│              │ • Regime      │   Pipeline         │                       │
│              │   Detector    │ • HITL Gate        │                       │
└──────────────┴───────────────┴───────────────────┴───────────────────────┘
         │              │              │                      │
         └──────────────┴──────────────┴──────────────────────┘
                                │
                   ┌────────────┴────────────┐
                   │  PostgreSQL 16 +         │
                   │  TimescaleDB (time-series)│
                   │  + pgvector (embeddings) │
                   │  + Redis (cache)         │
                   └─────────────────────────┘
```

★ = The "Live Track Record" page is the key differentiator for job applications

### 2.2 Technology Stack

```yaml
# Core Language
language: Python 3.11+

# Data Layer
database: PostgreSQL 16 + TimescaleDB extension
vector_store: pgvector (PostgreSQL extension) # keeps infra simple, one DB
cache: Redis (for rate limiting, caching API responses)
data_sources:
  market_data: yfinance (free), Alpha Vantage (free tier)
  fundamentals: SEC EDGAR API (free), Financial Modeling Prep (free tier)
  macro: FRED API (free)
  news: NewsAPI (free tier), RSS feeds
  canadian_markets: TMX Data (for TSX listings)

# Scheduling & Orchestration
scheduler: APScheduler (lightweight) → migrate to Prefect if needed
task_queue: None initially → Celery if needed

# Quant Engine
backtesting: vectorbt (fast, vectorized) + custom framework for walk-forward
optimization: scipy.optimize, cvxpy (for portfolio optimization)
statistics: statsmodels, scipy.stats
ml_models: scikit-learn, lightgbm, optionally pytorch

# AI Agent System
llm_provider: Anthropic Claude API (claude-sonnet-4-20250514)
agent_framework: LangGraph (for stateful multi-agent workflows)
rag_framework: LangChain for document processing + pgvector for retrieval
embeddings: Anthropic's embedding model or sentence-transformers
document_parsing: unstructured, pdfplumber (for SEC filings)
structured_output: Pydantic models for all agent outputs

# Frontend & Visualization
dashboard: Streamlit (fast to build, good for data apps)
charts: Plotly (interactive), matplotlib (static for reports)
reporting: Jinja2 templates → HTML/PDF reports

# DevOps & Quality
containerization: Docker + docker-compose
testing: pytest, pytest-cov
linting: ruff, mypy
ci_cd: GitHub Actions
documentation: mkdocs with mkdocs-material theme
```

### 2.3 Project Directory Structure

```
finsight/
├── README.md                          # Project overview with architecture diagram
├── pyproject.toml                     # Project config (dependencies, tools)
├── docker-compose.yml                 # Full stack: postgres, redis, app
├── Dockerfile                         # Application container
├── .env.example                       # Environment variable template
├── .github/
│   └── workflows/
│       ├── test.yml                   # CI: lint + test on push
│       └── deploy.yml                 # CD: build and deploy
├── docs/
│   ├── architecture.md                # System architecture documentation
│   ├── data_dictionary.md             # All data fields, sources, update frequency
│   ├── backtesting_methodology.md     # Statistical rigor documentation
│   └── agent_design.md               # Agent system design & prompts
├── src/
│   ├── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   ├── settings.py                # Pydantic settings (env-based config)
│   │   └── logging_config.py          # Structured logging setup
│   ├── data/
│   │   ├── __init__.py
│   │   ├── models.py                  # SQLAlchemy ORM models
│   │   ├── database.py                # DB connection & session management
│   │   ├── ingestion/
│   │   │   ├── __init__.py
│   │   │   ├── base.py                # Abstract ingestion interface
│   │   │   ├── market_data.py         # Price/volume data (yfinance)
│   │   │   ├── fundamentals.py        # Financial statements (SEC EDGAR, FMP)
│   │   │   ├── macro.py               # Macro indicators (FRED)
│   │   │   ├── news.py                # News articles & sentiment
│   │   │   └── canadian.py            # TSX-specific data sources
│   │   ├── quality/
│   │   │   ├── __init__.py
│   │   │   ├── validators.py          # Data validation rules
│   │   │   ├── cleaners.py            # Data cleaning transformations
│   │   │   └── monitors.py            # Data quality monitoring & alerts
│   │   └── scheduler.py               # Data pipeline scheduling
│   ├── quant/
│   │   ├── __init__.py
│   │   ├── factors/
│   │   │   ├── __init__.py
│   │   │   ├── base.py                # Abstract factor interface
│   │   │   ├── momentum.py            # Momentum factors (12-1, reversal)
│   │   │   ├── value.py               # Value factors (P/E, P/B, EV/EBITDA)
│   │   │   ├── quality.py             # Quality factors (ROE, margins, accruals)
│   │   │   ├── low_volatility.py      # Low vol / min variance factors
│   │   │   ├── composite.py           # Multi-factor combination
│   │   │   └── neutralizer.py         # Barra-style factor neutralization (NEW)
│   │   ├── ml/                         # ML-Enhanced Alpha (NEW - Tier 2)
│   │   │   ├── __init__.py
│   │   │   ├── feature_engineering.py # Multi-source feature construction
│   │   │   ├── alpha_model.py         # LightGBM/Ridge ensemble pipeline
│   │   │   ├── cross_validation.py    # Time-series aware CV (expanding window)
│   │   │   └── explainability.py      # SHAP feature importance
│   │   ├── backtest/
│   │   │   ├── __init__.py
│   │   │   ├── engine.py              # Core backtesting engine
│   │   │   ├── walk_forward.py        # Walk-forward validation
│   │   │   ├── transaction_costs.py   # Realistic cost modeling
│   │   │   └── statistics.py          # Performance stats & significance tests
│   │   ├── portfolio/
│   │   │   ├── __init__.py
│   │   │   ├── optimizer.py           # Mean-variance, risk parity, Black-Litterman
│   │   │   ├── constructor.py         # Cross-sectional portfolio construction (NEW)
│   │   │   ├── risk.py                # Risk metrics (VaR, CVaR, drawdown)
│   │   │   ├── rebalance.py           # Rebalancing logic & triggers
│   │   │   ├── attribution.py         # Performance attribution (Brinson, factor-based)
│   │   │   └── regime.py              # Regime detection & dynamic allocation (NEW)
│   │   └── signals/
│   │       ├── __init__.py
│   │       ├── generator.py           # Signal generation from factors
│   │       └── combiner.py            # Signal combination & weighting
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── supervisor.py              # Supervisor/router agent (LangGraph) (RENAMED)
│   │   ├── earnings_agent.py          # Financial statement analysis agent
│   │   ├── macro_agent.py             # Macroeconomic environment agent
│   │   ├── news_agent.py              # News & sentiment analysis agent
│   │   ├── peer_agent.py              # Peer comparison & industry analysis
│   │   ├── quant_signal_agent.py      # Quant signal summary agent (NEW)
│   │   ├── tools/                      # Agent tools (NEW)
│   │   │   ├── __init__.py
│   │   │   ├── sec_edgar.py           # SEC filing download/parse tool
│   │   │   ├── fred_api.py            # FRED data query tool
│   │   │   ├── financial_calc.py      # Financial calculation tool
│   │   │   ├── database_query.py      # DB query tool for agents
│   │   │   └── news_api.py            # News search tool
│   │   ├── rag/
│   │   │   ├── __init__.py
│   │   │   ├── document_processor.py  # SEC filing / annual report processing
│   │   │   ├── embeddings.py          # Embedding generation & management
│   │   │   ├── retriever.py           # Hybrid search (semantic + BM25 + metadata)
│   │   │   ├── reranker.py            # Cross-encoder reranking (NEW)
│   │   │   └── chunking.py            # Intelligent document chunking
│   │   ├── prompts/
│   │   │   ├── earnings_analysis.py   # Structured prompts for earnings analysis
│   │   │   ├── macro_assessment.py    # Macro environment assessment prompts
│   │   │   ├── news_synthesis.py      # News synthesis prompts
│   │   │   └── research_report.py     # Full research report generation
│   │   └── schemas/
│   │       ├── __init__.py
│   │       ├── earnings.py            # Pydantic models for earnings output
│   │       ├── macro.py               # Pydantic models for macro output
│   │       ├── research_report.py     # Pydantic models for research reports
│   │       └── signals.py             # Pydantic models for trading signals
│   ├── dashboard/
│   │   ├── __init__.py
│   │   ├── app.py                     # Main Streamlit application
│   │   ├── pages/
│   │   │   ├── portfolio.py           # Portfolio overview & performance
│   │   │   ├── research.py            # AI research interface
│   │   │   ├── backtest.py            # Backtest results & analysis
│   │   │   ├── risk.py                # Risk dashboard
│   │   │   ├── live_performance.py    # Live track record & proof (NEW - KEY)
│   │   │   ├── data_explorer.py       # Data exploration tools
│   │   │   └── agent_chat.py          # Interactive agent chat
│   │   └── components/
│   │       ├── charts.py              # Reusable chart components
│   │       ├── tables.py              # Reusable table components
│   │       └── sidebar.py             # Navigation sidebar
│   └── utils/
│       ├── __init__.py
│       ├── date_utils.py              # Trading calendar, date handling
│       ├── financial_utils.py         # Common financial calculations
│       └── rate_limiter.py            # API rate limiting
├── tests/
│   ├── conftest.py                    # Shared fixtures
│   ├── test_data/
│   │   ├── test_ingestion.py
│   │   └── test_quality.py
│   ├── test_quant/
│   │   ├── test_factors.py
│   │   ├── test_backtest.py
│   │   └── test_portfolio.py
│   ├── test_agents/
│   │   ├── test_orchestrator.py
│   │   ├── test_rag.py
│   │   └── test_earnings_agent.py
│   └── test_integration/
│       └── test_pipeline.py           # End-to-end pipeline tests
├── notebooks/
│   ├── 01_data_exploration.ipynb      # EDA on financial data
│   ├── 02_factor_research.ipynb       # Factor analysis research
│   ├── 03_backtest_analysis.ipynb     # Backtest results deep dive
│   └── 04_agent_demo.ipynb            # Agent system demo notebook
└── scripts/
    ├── setup_db.py                    # Database initialization
    ├── seed_data.py                   # Initial data loading
    ├── run_backtest.py                # CLI for running backtests
    └── generate_report.py             # CLI for generating research reports
```

---

## 3. Module Specifications

### 3.1 Data Layer

#### 3.1.1 Database Schema (Key Tables)

```sql
-- TimescaleDB hypertable for OHLCV data
CREATE TABLE market_data (
    symbol VARCHAR(20) NOT NULL,
    date DATE NOT NULL,
    open NUMERIC(12,4),
    high NUMERIC(12,4),
    low NUMERIC(12,4),
    close NUMERIC(12,4),
    adj_close NUMERIC(12,4),
    volume BIGINT,
    source VARCHAR(50),
    ingested_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (symbol, date)
);
SELECT create_hypertable('market_data', 'date');

-- Point-in-time fundamental data (CRITICAL for avoiding look-ahead bias)
CREATE TABLE fundamentals (
    symbol VARCHAR(20) NOT NULL,
    report_date DATE NOT NULL,          -- When the period ended
    filing_date DATE NOT NULL,          -- When it was actually available (point-in-time)
    period_type VARCHAR(10),            -- 'annual', 'quarterly'
    revenue NUMERIC(15,2),
    net_income NUMERIC(15,2),
    total_assets NUMERIC(15,2),
    total_equity NUMERIC(15,2),
    operating_cash_flow NUMERIC(15,2),
    eps NUMERIC(10,4),
    book_value_per_share NUMERIC(10,4),
    -- ... additional fields
    source VARCHAR(50),
    ingested_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (symbol, report_date, period_type)
);

-- Macro indicators
CREATE TABLE macro_indicators (
    indicator_id VARCHAR(50) NOT NULL,  -- e.g., 'GDP', 'CPI', 'FED_FUNDS_RATE'
    date DATE NOT NULL,
    value NUMERIC(15,4),
    source VARCHAR(50) DEFAULT 'FRED',
    ingested_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (indicator_id, date)
);

-- Factor signals (precomputed)
CREATE TABLE factor_signals (
    symbol VARCHAR(20) NOT NULL,
    date DATE NOT NULL,
    factor_name VARCHAR(50) NOT NULL,
    raw_value NUMERIC(15,6),
    z_score NUMERIC(8,4),              -- Cross-sectional z-score
    percentile NUMERIC(5,2),
    PRIMARY KEY (symbol, date, factor_name)
);

-- Portfolio positions & trades
CREATE TABLE portfolio_positions (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    shares NUMERIC(12,4),
    avg_cost NUMERIC(12,4),
    market_value NUMERIC(15,2),
    weight NUMERIC(6,4),
    strategy VARCHAR(50)
);

CREATE TABLE trades (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(4),                   -- 'BUY' or 'SELL'
    shares NUMERIC(12,4),
    price NUMERIC(12,4),
    commission NUMERIC(8,2),
    strategy VARCHAR(50),
    signal_source VARCHAR(100)         -- Which signal triggered this trade
);

-- Portfolio daily snapshots for performance tracking
CREATE TABLE portfolio_snapshots (
    date DATE NOT NULL,
    strategy VARCHAR(50) NOT NULL,
    total_value NUMERIC(15,2),
    cash NUMERIC(15,2),
    daily_return NUMERIC(10,6),
    benchmark_return NUMERIC(10,6),    -- SPY or relevant benchmark
    PRIMARY KEY (date, strategy)
);

-- Vector store for document embeddings (using pgvector)
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE document_embeddings (
    id SERIAL PRIMARY KEY,
    document_type VARCHAR(50),         -- '10-K', '10-Q', 'earnings_call', 'news'
    symbol VARCHAR(20),
    document_date DATE,
    chunk_index INT,
    chunk_text TEXT,
    embedding vector(1536),            -- Dimension depends on model
    metadata JSONB,                    -- Flexible metadata storage
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX ON document_embeddings USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
```

#### 3.1.2 Ingestion Pipeline Design

Each data source implements this interface:

```python
from abc import ABC, abstractmethod
from datetime import date
from typing import List

class DataIngestor(ABC):
    """Base class for all data ingestion sources."""
    
    @abstractmethod
    async def fetch(self, symbols: List[str], start_date: date, end_date: date) -> pd.DataFrame:
        """Fetch raw data from source."""
        pass
    
    @abstractmethod
    def validate(self, df: pd.DataFrame) -> tuple[pd.DataFrame, List[str]]:
        """Validate data quality. Returns (clean_data, list_of_issues)."""
        pass
    
    @abstractmethod
    async def ingest(self, symbols: List[str], start_date: date, end_date: date) -> dict:
        """Full pipeline: fetch → validate → store. Returns summary stats."""
        pass
```

**Key data quality rules to implement:**
- No future data leaks (point-in-time enforcement)
- Adjusted price consistency (handle splits, dividends)
- Missing data detection and interpolation policy
- Outlier detection (>5 sigma daily moves → flag for review)
- Cross-source validation (compare yfinance vs FMP for same data)

#### 3.1.3 Scheduling

```python
# Daily schedule (all times in ET, aligned with market hours)
SCHEDULE = {
    "market_data": {
        "cron": "0 18 * * 1-5",       # 6 PM ET, after market close
        "description": "Fetch daily OHLCV for all tracked symbols"
    },
    "fundamentals": {
        "cron": "0 20 * * 1-5",       # 8 PM ET
        "description": "Check for new SEC filings and earnings"
    },
    "macro_indicators": {
        "cron": "0 12 * * 1-5",       # Noon ET
        "description": "Update macro indicators from FRED"
    },
    "news_sentiment": {
        "cron": "0 */4 * * 1-5",      # Every 4 hours on weekdays
        "description": "Fetch and analyze news for tracked symbols"
    },
    "factor_signals": {
        "cron": "0 19 * * 5",         # Friday 7 PM ET (weekly recompute)
        "description": "Recompute all factor signals"
    },
    "portfolio_snapshot": {
        "cron": "30 16 * * 1-5",      # 4:30 PM ET, after close
        "description": "Snapshot portfolio values and compute daily returns"
    }
}
```

---

### 3.2 Quant Engine (Aligned with Modern Quant Fund Architecture)

> **Design Philosophy**: This engine follows the "Quant 4.0" paradigm — combining traditional factor investing with ML-enhanced alpha signals, automated factor discovery, cross-sectional neutralization, and regime-aware portfolio construction. The goal is to demonstrate the full pipeline that firms like CC&L, AQR, and Two Sigma use in production.

#### 3.2.1 Factor Library (Three Tiers)

**Tier 1: Classic Academic Factors (Proven, Well-Documented)**

These are the baseline — every quant should know them:

| Factor | Category | Calculation | Academic Source |
|--------|----------|-------------|----------------|
| 12-1 Month Momentum | Momentum | Past 12-month return, skip most recent month | Jegadeesh & Titman (1993) |
| Short-Term Reversal | Reversal | Past 1-month return (negative signal) | Jegadeesh (1990) |
| Earnings Yield | Value | E/P ratio using trailing 12M earnings | Basu (1977) |
| Book-to-Market | Value | Book value / market cap | Fama & French (1993) |
| EV/EBITDA | Value | Enterprise value / EBITDA | Loughran & Wellman (2011) |
| ROE | Quality | Net income / shareholders equity | Novy-Marx (2013) |
| Gross Profitability | Quality | Gross profit / total assets | Novy-Marx (2013) |
| Accruals | Quality | Change in non-cash working capital / total assets | Sloan (1996) |
| Asset Growth | Investment | Year-over-year total asset growth (negative signal) | Cooper et al. (2008) |
| 60-Day Volatility | Low Vol | Rolling 60-day return standard deviation (negative signal) | Ang et al. (2006) |

**Tier 2: ML-Enhanced Factors (The Differentiator)**

These demonstrate modern quant capabilities:

| Factor | Method | Description |
|--------|--------|-------------|
| Nonlinear Momentum | LightGBM | Interaction effects between momentum at different horizons, volume, and volatility |
| Earnings Quality ML | Gradient Boosted Trees | Ensemble of 15+ accounting features to predict earnings persistence |
| Sector-Relative Value | Cross-sectional regression residuals | Value signal after neutralizing sector and size effects |
| News Sentiment Alpha | LLM + NLP | Claude-generated sentiment scores from earnings calls and news (unique AI-driven factor) |
| Volatility Regime Factor | Hidden Markov Model | Factor that adapts to current market regime (risk-on vs risk-off) |

**Tier 3: LLM-Assisted Alpha Discovery (Cutting-Edge, Interview Talking Point)**

Inspired by the Alpha-GPT research (2025), use Claude to assist in alpha signal discovery:

```python
class LLMAlphaDiscovery:
    """
    Uses LLM to generate hypotheses for new alpha factors,
    then validates them with rigorous statistical testing.
    
    This is a 'Quant 4.0' approach: "AI creates AI"
    — The LLM proposes factor formulas, the backtesting engine validates them.
    
    Interview Talking Point: "I built an LLM-assisted alpha mining system
    inspired by Alpha-GPT. The LLM generates factor hypotheses from financial
    theory, and my backtesting engine validates them with walk-forward testing
    and multiple hypothesis correction."
    """
    
    async def generate_alpha_hypotheses(
        self,
        market_context: str,          # Current regime description
        existing_factors: List[str],   # Factors already in the library
        data_available: List[str],     # What data fields are available
        num_hypotheses: int = 10
    ) -> List[AlphaHypothesis]:
        """Ask Claude to propose new factor formulas based on financial theory."""
        pass
    
    async def validate_hypothesis(
        self,
        hypothesis: AlphaHypothesis,
        data: pd.DataFrame
    ) -> AlphaValidationResult:
        """
        Rigorous validation:
        1. Compute factor values
        2. Cross-sectional neutralization (remove sector/size bias)
        3. Information Coefficient (IC) analysis
        4. Walk-forward backtest
        5. Multiple hypothesis correction (Bonferroni or BH)
        6. Turnover and capacity analysis
        """
        pass
```

#### 3.2.2 Factor Neutralization & Bias Correction (Critical for Real-World Performance)

> **Why this matters**: Raw factors embed systematic biases (size, sector, country). A "value" signal might really just be a "small-cap" signal in disguise. Modern quant funds neutralize these biases. This is a KEY interview topic.

```python
class FactorNeutralizer:
    """
    Cross-sectional neutralization following Barra-style risk model approach.
    
    For each factor signal:
    1. Regress factor values against risk factors (market, sector, size, country)
    2. Take the residual as the "pure" alpha signal
    3. Winsorize extremes and z-score normalize
    
    Interview Talking Point: "I implemented Barra-style factor neutralization 
    to ensure alpha signals aren't contaminated by sector or size exposure.
    This is the same methodology used by MSCI Barra and major quant funds."
    """
    
    def neutralize(
        self,
        factor_values: pd.Series,       # Raw factor signal
        sector: pd.Series,              # GICS sector labels
        market_cap: pd.Series,          # For size neutralization
        additional_risks: pd.DataFrame = None  # Optional: country, currency, etc.
    ) -> pd.Series:
        """Returns neutralized, z-scored factor values."""
        # 1. Create dummy variables for sectors
        # 2. Log-transform market cap
        # 3. Run cross-sectional OLS: factor ~ sector_dummies + log(mcap) + risks
        # 4. Return residuals, winsorized at 3σ, then z-scored
        pass
    
    def compute_factor_exposure(
        self,
        portfolio_weights: pd.Series,
        factor_values: pd.DataFrame
    ) -> pd.Series:
        """Compute portfolio's exposure to each risk factor."""
        pass
```

#### 3.2.3 ML Model Pipeline (Full MLOps Lifecycle)

```python
class AlphaModelPipeline:
    """
    End-to-end ML pipeline for return prediction.
    
    Architecture follows modern quant fund practices:
    - Feature engineering from multi-source data
    - Time-series aware cross-validation (expanding window)
    - Ensemble of models (LightGBM + Ridge + simple NN)
    - SHAP-based feature importance for explainability
    - Model monitoring and drift detection
    
    Interview Talking Point: "My ML pipeline uses time-series cross-validation 
    with expanding windows, ensemble prediction, and SHAP explainability. 
    I can show you exactly which features drive each prediction."
    """
    
    # Feature engineering
    feature_groups = {
        "momentum": ["ret_1m", "ret_3m", "ret_6m", "ret_12m", "ret_12m_skip1m"],
        "value": ["ep_ratio", "bp_ratio", "ev_ebitda", "fcf_yield"],
        "quality": ["roe", "gross_margin", "asset_turnover", "accruals", "debt_equity"],
        "growth": ["revenue_growth_yoy", "earnings_growth_yoy", "asset_growth"],
        "risk": ["volatility_60d", "beta_252d", "idio_vol", "max_drawdown_60d"],
        "technical": ["volume_ratio_20d", "rsi_14d", "price_to_52w_high"],
        "sentiment": ["news_sentiment_7d", "earnings_call_sentiment"],  # LLM-derived
    }
    
    # Model ensemble
    models = {
        "lightgbm": LGBMRegressor(n_estimators=500, max_depth=6, learning_rate=0.05),
        "ridge": Ridge(alpha=1.0),
        "elastic_net": ElasticNet(alpha=0.1, l1_ratio=0.5),
    }
    ensemble_weights = {"lightgbm": 0.5, "ridge": 0.3, "elastic_net": 0.2}
    
    # Cross-validation scheme (CRITICAL: time-series aware)
    cv_config = {
        "method": "expanding_window",       # NOT random K-fold!
        "initial_train_months": 36,         # 3 years minimum training
        "validation_months": 6,             # 6 months validation
        "test_months": 6,                   # 6 months true out-of-sample
        "step_months": 6,                   # Roll forward 6 months each fold
        "purge_gap_days": 5,                # Gap between train/test to prevent leakage
    }
```

#### 3.2.4 Backtesting Framework

**Critical methodology requirements (these are interview questions):**

```python
class BacktestConfig:
    """Configuration for a rigorous backtest."""
    
    # Universe
    universe: str = "SP500"             # or "SP1500", "Russell1000", "TSX60"
    min_market_cap: float = 1e9         # $1B minimum
    min_avg_volume: float = 1e6         # $1M daily minimum
    
    # Timing
    start_date: date                     # Backtest start
    end_date: date                       # Backtest end
    rebalance_frequency: str = "monthly" # 'daily', 'weekly', 'monthly', 'quarterly'
    rebalance_day: int = 1              # Day of month for rebalancing
    
    # Portfolio construction
    num_holdings: int = 50               # Number of stocks to hold
    max_sector_weight: float = 0.25     # 25% max per sector
    max_position_weight: float = 0.05   # 5% max per position
    
    # Transaction costs (REALISTIC)
    commission_per_share: float = 0.005  # $0.005/share
    spread_cost_bps: float = 5.0        # 5 bps half-spread
    market_impact_bps: float = 10.0     # 10 bps market impact for mid-caps
    
    # Validation
    walk_forward_windows: int = 5        # Number of IS/OOS splits
    in_sample_months: int = 36           # 3 years in-sample
    out_of_sample_months: int = 12       # 1 year out-of-sample
    
    # Statistical testing
    min_sharpe_ratio: float = 0.5        # Minimum acceptable Sharpe
    min_num_trades: int = 100            # Minimum trades for significance
    significance_level: float = 0.05     # p-value threshold
```

**Performance metrics to compute:**
- Annualized return, volatility, Sharpe ratio, Sortino ratio
- Maximum drawdown, average drawdown, recovery time
- Win rate, profit factor, average win/loss ratio
- Turnover, transaction costs as % of return
- Information ratio vs benchmark (SPY, equal-weight SPY)
- Factor exposure analysis (how much return comes from which factors)
- t-statistic of alpha (is it statistically significant?)
- Calmar ratio (return / max drawdown)

#### 3.2.5 Portfolio Optimization (Cross-Sectional Approach)

> **Key insight from latest research**: Modern quant funds use cross-sectional portfolio construction — ranking stocks within the universe and going long the top quintile, rather than predicting absolute returns. This naturally hedges market risk.

Implement four approaches (increasing sophistication):

1. **Equal-Weight Long-Only** — Baseline. Select top 50 stocks by composite score, equal weight. Simple but surprisingly hard to beat.

2. **Risk Parity** — Weight inversely proportional to volatility. Each stock contributes equal risk. Good in volatile markets.

3. **Mean-Variance with Constraints** — Classic Markowitz with practical constraints. Use shrinkage estimator for covariance matrix (Ledoit-Wolf) to avoid extreme weights.

4. **Black-Litterman with Factor Views** — Incorporate factor model views into equilibrium weights. The LLM Macro Agent outputs feed directly into BL views. *Interview Talking Point: "My system uses Black-Litterman to combine equilibrium market views with our AI-generated macro outlook."*

```python
class CrossSectionalPortfolioConstructor:
    """
    Cross-sectional portfolio construction:
    1. Score all stocks using factor/ML signals
    2. Rank within universe
    3. Select top quintile
    4. Apply optimization within selected stocks
    5. Apply risk constraints
    
    This approach naturally produces market-neutral-ish returns
    because alpha comes from RELATIVE stock selection, not market timing.
    """
    
    def construct(
        self,
        scores: pd.Series,              # Alpha scores for all stocks
        covariance: pd.DataFrame,        # Return covariance matrix
        constraints: PortfolioConstraints,
        method: str = "risk_parity"
    ) -> pd.Series:
        """Returns optimal weight vector."""
        pass

class PortfolioConstraints:
    max_position_weight: float = 0.05
    max_sector_weight: float = 0.25
    max_turnover: float = 0.30          # 30% max monthly turnover
    max_factor_exposure: float = 0.50   # Limit unintended factor bets
    min_holdings: int = 30
    max_holdings: int = 60
```

#### 3.2.6 Regime Detection (Market Awareness)

```python
class RegimeDetector:
    """
    Detect market regime to adjust strategy parameters.
    
    Uses Hidden Markov Model on market features to identify:
    - Bull (low vol, positive trend)
    - Bear (high vol, negative trend)  
    - High Volatility (elevated vol, unclear direction)
    - Recovery (declining vol, turning positive)
    
    Portfolio adjusts: in Bear regime, reduce equity allocation,
    tighten stop-losses, favor quality/low-vol factors.
    In Bull regime, favor momentum/growth factors.
    
    Interview Talking Point: "My system detects market regimes using
    an HMM and dynamically adjusts factor weights. In the 2022 drawdown,
    the regime detector shifted to quality/low-vol factors, reducing
    drawdown by X% compared to a static allocation."
    """
    
    features_used = [
        "sp500_return_21d",             # Market trend
        "sp500_volatility_21d",         # Realized volatility
        "vix_level",                    # Implied volatility
        "yield_curve_slope",            # 10Y - 2Y treasury spread
        "credit_spread",               # BAA - AAA spread
        "market_breadth",              # % stocks above 200d MA
    ]
    
    def detect_regime(self, market_data: pd.DataFrame) -> str:
        """Returns current regime label."""
        pass
    
    def get_regime_adjustments(self, regime: str) -> dict:
        """Returns parameter adjustments for current regime."""
        adjustments = {
            "bull": {"equity_allocation": 0.95, "factor_tilt": "momentum_growth"},
            "bear": {"equity_allocation": 0.60, "factor_tilt": "quality_low_vol"},
            "high_vol": {"equity_allocation": 0.75, "factor_tilt": "low_vol_defensive"},
            "recovery": {"equity_allocation": 0.85, "factor_tilt": "value_momentum"},
        }
        return adjustments[regime]
```

---

### 3.3 AI Agent System (The #1 Differentiator)

> **Design Philosophy**: This is the module that will land you the job. It demonstrates Agentic RAG, multi-agent coordination with LangGraph, structured outputs, tool use, human-in-the-loop patterns, and MCP integration — exactly the skills every AI engineering role in Vancouver demands.
>
> **Architecture Reference**: Follows the AWS reference architecture for financial analysis agents (LangGraph + Strands Agents, published Aug 2025) and the Alpha-GPT paper (EMNLP 2025) for LLM-enhanced alpha mining.

This module must demonstrate:
- **Agentic RAG** — not simple retrieve-and-generate, but agents that decide WHEN to retrieve, WHAT to retrieve, and can do multi-hop reasoning across documents
- **Multi-agent coordination** — specialized agents with a supervisor/orchestrator pattern
- **Structured outputs** — all agent outputs validated by Pydantic schemas
- **Tool use** — agents can call data APIs, run calculations, query databases
- **Human-in-the-loop** — agent can pause and ask for human review on high-stakes decisions
- **Production-grade** — error handling, retries, token usage monitoring, cost tracking

#### 3.3.1 Agent Architecture (LangGraph Supervisor Pattern)

```
User Query: "Analyze AAPL for potential investment"
    │
    ▼
┌──────────────────────────────┐
│   Supervisor / Router Agent   │ ← Analyzes query, plans execution, routes to specialists
│   (LangGraph StateGraph)     │    Decides which agents to invoke (not always all of them)
└───┬───┬───┬───┬───┬──────────┘
    │   │   │   │   │
    ▼   ▼   ▼   ▼   ▼
┌──────┐┌──────┐┌──────┐┌──────┐┌─────────┐
│Earnings││Macro ││News  ││Peer  ││Quant    │ ← Specialized agents with TOOLS
│Agent  ││Agent ││Agent ││Agent ││Signal   │    Each can: call APIs, query DB,
│       ││      ││      ││      ││Agent    │    run calculations, do RAG retrieval
│[Tools]││[Tools]││[Tools]││[Tools]││[Tools] │
└──┬───┘└──┬───┘└──┬───┘└──┬───┘└──┬──────┘
   │       │       │       │       │
   ▼       ▼       ▼       ▼       ▼
┌─────────────────────────────────────────┐
│          Result Aggregator               │ ← Validates structured outputs
│          + Quality Checker               │    Detects conflicts, asks for clarification
└────────────────┬────────────────────────┘
                 │
          ┌──────┴──────┐
          │ Human Review │ ← Optional: pause for high-conviction decisions
          │ (HITL Gate)  │    or when agents disagree significantly
          └──────┬──────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│   Research Report Generator              │ ← Produces institutional-quality report
│   + Alpha Signal Emitter                 │    AND emits trading signals to Quant Engine
└─────────────────────────────────────────┘
```

**Key Design Patterns:**

1. **Conditional Routing**: The Supervisor doesn't always invoke all agents. For a simple price check, it might only call the Quant Signal Agent. For a full analysis, it orchestrates all five.

2. **Parallel Execution**: Independent agents (Earnings, News, Peer) run in parallel via LangGraph's `fan_out` pattern, reducing latency.

3. **Tool-Using Agents**: Each agent has access to specific tools:
   - Earnings Agent: SEC EDGAR API tool, financial calculation tool, RAG retrieval tool
   - Macro Agent: FRED API tool, regime detector tool
   - News Agent: NewsAPI tool, sentiment scoring tool
   - Peer Agent: financial data API tool, comparison calculation tool
   - Quant Signal Agent: database query tool, factor computation tool

4. **Feedback Loop**: The Quality Checker can send results back to agents for refinement if confidence is low or outputs conflict.

#### 3.3.2 Agent Specifications

**Earnings Analysis Agent:**
```python
class EarningsAnalysisOutput(BaseModel):
    """Structured output from earnings analysis."""
    
    symbol: str
    analysis_date: date
    
    # Revenue analysis
    revenue_trend: str           # 'accelerating', 'stable', 'decelerating', 'declining'
    revenue_growth_yoy: float
    revenue_surprise_pct: float  # vs consensus if available
    
    # Profitability
    margin_trend: str
    gross_margin: float
    operating_margin: float
    net_margin: float
    margin_expansion: bool
    
    # Cash flow
    fcf_yield: float
    cash_conversion: float       # FCF / Net Income
    capex_trend: str
    
    # Balance sheet
    debt_to_equity: float
    current_ratio: float
    balance_sheet_quality: str    # 'strong', 'adequate', 'concerning', 'weak'
    
    # Management commentary (from RAG on earnings call/10-K)
    key_guidance_points: List[str]
    risk_factors_highlighted: List[str]
    management_tone: str         # 'confident', 'cautious', 'defensive', 'optimistic'
    
    # Overall assessment
    fundamental_score: float     # 0-100
    investment_thesis: str       # 2-3 sentence summary
    key_risks: List[str]
    catalysts: List[str]
    
    # Citations
    sources_used: List[str]      # Document references
    confidence_level: str        # 'high', 'medium', 'low'
```

**Macro Environment Agent:**
```python
class MacroAssessmentOutput(BaseModel):
    """Structured macro environment assessment."""
    
    assessment_date: date
    
    # Economic cycle
    cycle_phase: str             # 'expansion', 'peak', 'contraction', 'trough'
    cycle_confidence: float
    
    # Key indicators
    gdp_growth_trend: str
    inflation_trend: str
    employment_trend: str
    yield_curve_status: str      # 'normal', 'flat', 'inverted'
    
    # Policy environment
    fed_policy_stance: str       # 'hawkish', 'neutral', 'dovish'
    rate_direction: str          # 'hiking', 'pausing', 'cutting'
    
    # Market implications
    equity_outlook: str          # 'bullish', 'neutral', 'bearish'
    sector_preferences: List[str]
    risk_factors: List[str]
    
    # Actionable for portfolio
    suggested_equity_allocation: float  # 0.0 to 1.0
    defensive_tilt: bool
    reasoning: str
```

#### 3.3.3 Agentic RAG Pipeline Design (Not Simple RAG)

> **Key distinction**: Simple RAG = retrieve → generate. Agentic RAG = the agent DECIDES when/what to retrieve, can do multi-hop reasoning, self-corrects, and reformulates queries. This is what employers want to see.

```python
class AgenticFinancialRAG:
    """
    Agentic RAG pipeline optimized for financial documents.
    
    Key capabilities beyond simple RAG:
    1. Query Decomposition — breaks complex questions into sub-queries
    2. Adaptive Retrieval — decides whether retrieval is needed at all
    3. Multi-hop Reasoning — chains multiple retrievals for complex questions
    4. Self-Reflection — evaluates retrieved context quality, re-retrieves if needed
    5. Hybrid Search — combines semantic (vector) + lexical (BM25) + metadata filtering
    6. Citation Tracking — every generated claim traces to specific source chunks
    
    Interview Talking Point: "My RAG system uses agentic retrieval — the LLM
    decides what to search for, evaluates retrieval quality, and can do multi-hop
    reasoning. For example, when asked 'How does AAPL's margin compare to its 
    sector average?', it first retrieves AAPL's margins from the 10-K, then 
    retrieves peer margins, then synthesizes the comparison."
    """
    
    def __init__(self):
        self.embeddings = ...         # Embedding model
        self.vector_store = ...       # pgvector
        self.bm25_index = ...         # BM25 for keyword search
        self.reranker = ...           # Cross-encoder reranker
    
    async def ingest_document(
        self,
        document_path: str,
        document_type: str,           # '10-K', '10-Q', 'earnings_call', 'annual_report'
        symbol: str,
        document_date: date
    ):
        """
        Process and store a financial document.
        
        Steps:
        1. Parse document (handle SEC XBRL, PDF, HTML)
        2. Section detection (Risk Factors, MD&A, Financial Statements, etc.)
        3. Intelligent chunking:
           - Respect section boundaries
           - Keep tables as single chunks (tables are critical in financial docs)
           - Overlap with context from section headers
           - Attach metadata: symbol, date, section_name, page_number
        4. Generate embeddings (semantic)
        5. Build BM25 index (keyword — important for ticker symbols, numbers)
        6. Store with rich metadata in pgvector
        """
        pass
    
    async def agentic_retrieve(
        self,
        query: str,
        symbol: str = None,
        document_types: List[str] = None,
        date_range: tuple = None,
        max_hops: int = 3
    ) -> AgenticRetrievalResult:
        """
        Agentic retrieval with multi-hop reasoning.
        
        Flow:
        1. LLM analyzes query → decides if retrieval needed
        2. If yes: decompose into sub-queries
        3. For each sub-query:
           a. Hybrid search (semantic + BM25 + metadata filter)
           b. Cross-encoder reranking
           c. LLM evaluates relevance: "Is this sufficient?"
           d. If not: reformulate query and retrieve again (up to max_hops)
        4. Combine all retrieved chunks with deduplication
        5. Return with citation metadata
        """
        pass
```

**RAG Quality Measures (Production-Grade):**
- **Chunk quality**: Section-aware splitting; tables kept intact; overlap with header context
- **Retrieval quality**: Hybrid search (semantic + BM25) with cross-encoder reranking
- **Answer quality**: Every factual claim must cite a specific source chunk
- **Hallucination guard**: Post-generation check — does the answer contain claims not supported by retrieved context?
- **Evaluation framework**: Track retrieval precision@k, answer faithfulness, citation accuracy across test queries

#### 3.3.4 Research Report Generation

The system's crown jewel — generates institutional-quality research reports:

```python
class ResearchReportOutput(BaseModel):
    """Full research report combining all agent outputs."""
    
    symbol: str
    company_name: str
    generated_date: date
    
    # Executive Summary
    executive_summary: str           # 3-5 sentences
    investment_recommendation: str   # 'Strong Buy', 'Buy', 'Hold', 'Sell', 'Strong Sell'
    target_return_12m: float        # Expected 12-month return
    conviction_level: str           # 'High', 'Medium', 'Low'
    
    # Section: Business Overview (from RAG on 10-K)
    business_description: str
    competitive_advantages: List[str]
    key_risks: List[str]
    
    # Section: Financial Analysis (from Earnings Agent)
    financial_analysis: EarningsAnalysisOutput
    
    # Section: Industry & Peers (from Peer Agent)
    peer_comparison: dict
    industry_position: str
    
    # Section: Macro Context (from Macro Agent)
    macro_context: MacroAssessmentOutput
    
    # Section: News & Sentiment (from News Agent)
    recent_news_summary: str
    sentiment_score: float
    
    # Section: Quantitative Signals (from Quant Engine)
    factor_scores: dict              # Factor name → score
    backtest_performance: dict       # Key metrics from relevant strategy
    
    # Section: Conclusion
    bull_case: str
    bear_case: str
    key_monitoring_points: List[str]
    
    # Meta
    sources_cited: List[str]
    disclaimer: str = "This report is generated by an AI system for informational purposes only. Not investment advice."
```

---

### 3.4 Dashboard & Visualization

#### 3.4.1 Streamlit Pages

**Page 1: Portfolio Overview**
- Current portfolio positions with P&L
- Equity curve vs benchmark (interactive Plotly chart)
- Key metrics: total return, Sharpe, max drawdown, win rate
- Sector allocation pie chart
- Recent trades table

**Page 2: AI Research Assistant**
- Text input: "Analyze [SYMBOL]" or ask any investment question
- Agent execution progress indicator (shows which agents are working)
- Rendered research report with collapsible sections
- Source citations with links to original documents
- Chat history for follow-up questions

**Page 3: Backtest Lab**
- Factor selection and parameter configuration
- Run backtest with progress bar
- Results dashboard: equity curve, drawdown chart, monthly returns heatmap
- Walk-forward validation results
- Statistical significance test results
- Comparison against benchmark strategies

**Page 4: Risk Dashboard**
- Real-time portfolio risk metrics (VaR, CVaR, beta, correlation matrix)
- Factor exposure breakdown
- Concentration analysis (position, sector, factor)
- Stress test scenarios (what-if: rates +200bps, market -20%, etc.)
- Drawdown analysis with recovery tracking

**Page 5: Data Explorer**
- Browse all available data (prices, fundamentals, macro)
- Data quality dashboard (coverage, freshness, issues)
- Custom query interface
- Export capabilities

---

## 4. Phased Delivery Plan

### Phase 1: Foundation (Days 1-5)

**Goal**: Data platform + quant engine with factor neutralization + project infrastructure

**Deliverables:**
- [ ] Project scaffolding (directory structure, pyproject.toml, Docker setup)
- [ ] PostgreSQL + TimescaleDB + pgvector + Redis Docker compose
- [ ] Database schema creation script (all tables including portfolio_snapshots)
- [ ] Market data ingestion (yfinance) with quality validation
- [ ] Fundamental data ingestion (Financial Modeling Prep free API, or SEC EDGAR)
- [ ] Macro data ingestion (FRED) — yield curve, VIX, credit spreads
- [ ] Basic scheduling with APScheduler
- [ ] **Tier 1 factors**: All 10 classic factors implemented with proper point-in-time data
- [ ] **Factor neutralization**: Barra-style sector/size neutralization
- [ ] **Backtesting engine**: Walk-forward validation with realistic transaction costs
- [ ] **Portfolio constructor**: Equal-weight + risk parity with constraints
- [ ] Pytest setup with fixtures for data layer and factor tests
- [ ] README with architecture diagram

**Definition of Done:**
- `docker-compose up` brings up the full stack
- `python scripts/seed_data.py` loads 3 years of data for SP500 + TSX60
- Walk-forward backtest of momentum factor runs end-to-end, produces performance tearsheet with Sharpe, drawdown, factor attribution
- Factor neutralization demonstrably removes sector bias (show before/after)
- All tests pass, ruff/mypy clean

### Phase 2: AI Agent System (Days 6-10)

**Goal**: Multi-agent research system with agentic RAG + regime detection + ML signals

**Deliverables:**
- [ ] **Agentic RAG pipeline**: SEC filing download → parse → section-aware chunk → embed → pgvector + BM25 index
- [ ] **Earnings Analysis Agent** with structured Pydantic output + RAG tool + financial calc tool
- [ ] **Macro Environment Agent** with FRED API tool + regime detector integration
- [ ] **News Sentiment Agent** (using free news APIs + LLM sentiment scoring)
- [ ] **Quant Signal Agent** that queries factor DB and summarizes quantitative outlook
- [ ] **Supervisor/Router** (LangGraph StateGraph) with conditional routing + parallel fan-out
- [ ] **Research Report generation** (HTML output with citations)
- [ ] **Regime Detector**: HMM-based market regime classification
- [ ] **ML Alpha Model** (stretch): LightGBM + Ridge ensemble with time-series CV
- [ ] Agent unit tests with mocked LLM responses
- [ ] Agent tools documented in `docs/agent_design.md`

**Definition of Done:**
- Running `python scripts/generate_report.py AAPL` produces a structured research report
- The report includes RAG-sourced citations from actual SEC filings
- Supervisor correctly routes simple queries (1 agent) vs complex queries (all agents)
- Agent outputs conform to Pydantic schemas (no unstructured text dumps)
- Regime detector correctly identifies current market state
- At least one agent can use tools to fetch live data

### Phase 3: Dashboard & Polish (Days 11-14)

**Goal**: Polished frontend with live track record + complete documentation + paper trading

**Deliverables:**
- [ ] Streamlit dashboard with 6 pages:
  - [ ] Portfolio Overview (positions, equity curve, sector allocation)
  - [ ] AI Research Chat (agent interaction with streaming output)
  - [ ] Backtest Lab (configurable parameters, walk-forward results)
  - [ ] Risk Dashboard (VaR, factor exposure, stress tests)
  - [ ] **Live Track Record ★** (backtest vs live comparison, tearsheet, factor attribution)
  - [ ] Data Explorer (browse data, quality metrics)
- [ ] Interactive charts with Plotly
- [ ] **Performance tearsheet generator** (the PDF/HTML you show to employers)
- [ ] GitHub Actions CI (lint + test + type check)
- [ ] Complete documentation:
  - [ ] architecture.md with system diagram
  - [ ] data_dictionary.md (all fields, sources, update frequency)
  - [ ] backtesting_methodology.md (statistical rigor)
  - [ ] agent_design.md (prompts, schemas, tools)
- [ ] Demo notebook with full walkthrough
- [ ] Paper trading setup with daily scheduling + snapshot recording
- [ ] Performance monitoring (basic alerts for data failures or drawdown limits)

**Definition of Done:**
- `streamlit run src/dashboard/app.py` shows a polished, professional dashboard
- The **Live Track Record** page shows equity curve, key metrics, factor attribution
- A hiring manager can understand the system's capabilities within 2 minutes
- All documentation is written and the GitHub repo looks institutional-grade
- Paper trading is running and recording daily snapshots automatically
- `python scripts/generate_tearsheet.py` produces a PDF performance report

---

## 5. Quality Standards

### 5.1 Code Quality

- **Type hints everywhere** — all function signatures fully typed
- **Docstrings** — Google-style docstrings on all public classes and functions
- **Ruff** for linting (replaces flake8 + black + isort)
- **Mypy** for static type checking (strict mode)
- **Test coverage** — target 80%+ on core modules (data, quant, agents)
- **No magic numbers** — all constants in config or named variables
- **Async where appropriate** — data ingestion, API calls, agent execution
- **Error handling** — structured exception hierarchy, not bare `except:`

### 5.2 Data Quality

- All data validated at ingestion time
- Point-in-time enforcement on all fundamental data
- Data lineage tracked (source, ingestion time, transformations)
- Missing data clearly flagged (NaN, not silently filled)
- Quality monitoring dashboard showing coverage and freshness

### 5.3 Quantitative Rigor

- No look-ahead bias (point-in-time data only)
- Walk-forward validation (never test on training data)
- Transaction costs included in all backtests
- Statistical significance testing on all alpha claims
- Survivorship bias handling (use historical index constituents if available)
- Clear documentation of all assumptions and limitations

### 5.4 AI System Quality

- All agent outputs validated against Pydantic schemas
- Hallucination guardrails (agents must cite sources for factual claims)
- Token usage monitoring and cost tracking
- Prompt versioning (all prompts stored in code, not hardcoded strings)
- Fallback behavior when LLM is unavailable or returns errors
- Response quality evaluation framework (optional: LLM-as-judge for report quality)

### 5.5 Documentation Quality

- Architecture diagram (Mermaid or draw.io, embedded in README)
- Data dictionary with all fields, sources, update frequencies
- Backtesting methodology document explaining statistical approach
- Agent design document explaining each agent's purpose, inputs, outputs, prompts
- Setup guide (from zero to running system in <30 minutes)
- Decision log (why certain technologies were chosen over alternatives)

---

## 6. Configuration & Environment

### 6.1 Environment Variables (.env)

```bash
# Database
DATABASE_URL=postgresql://finsight:finsight@localhost:5432/finsight

# Redis
REDIS_URL=redis://localhost:6379

# API Keys
ANTHROPIC_API_KEY=sk-ant-...          # Required for agent system
FMP_API_KEY=...                        # Financial Modeling Prep (free tier)
FRED_API_KEY=...                       # Federal Reserve Economic Data (free)
NEWS_API_KEY=...                       # NewsAPI.org (free tier)
ALPHA_VANTAGE_API_KEY=...             # Alpha Vantage (free tier, 5 calls/min)

# Configuration
MARKET_DATA_START_DATE=2020-01-01     # How far back to load data
UNIVERSE=SP500                         # Stock universe
REBALANCE_FREQUENCY=monthly           # Portfolio rebalance cadence
LOG_LEVEL=INFO
```

### 6.2 Docker Compose

```yaml
version: "3.8"
services:
  db:
    image: timescale/timescaledb:latest-pg16
    environment:
      POSTGRES_DB: finsight
      POSTGRES_USER: finsight
      POSTGRES_PASSWORD: finsight
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data
      - ./scripts/init_extensions.sql:/docker-entrypoint-initdb.d/init.sql
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U finsight"]
      interval: 5s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  app:
    build: .
    depends_on:
      db:
        condition: service_healthy
      redis:
        condition: service_started
    env_file: .env
    ports:
      - "8501:8501"  # Streamlit
    volumes:
      - ./src:/app/src
      - ./data:/app/data  # Persistent data storage

volumes:
  pgdata:
```

---

## 7. Market Coverage

### 7.1 Initial Universe (Phase 1)

Start focused, expand later:

- **Primary**: S&P 500 constituents (US large cap)
- **Secondary**: S&P/TSX 60 (Canadian large cap, relevant for Vancouver market)
- **Benchmarks**: SPY, ^GSPTSE (TSX Composite), AGG (bonds)

### 7.2 Data Coverage Targets

| Data Type | Coverage | Update Frequency | Source |
|-----------|----------|------------------|--------|
| Daily OHLCV | S&P 500 + TSX 60 | Daily after close | yfinance |
| Quarterly financials | S&P 500 | After each filing | FMP / SEC EDGAR |
| Annual financials | S&P 500 | After annual filing | FMP / SEC EDGAR |
| Macro indicators | 15 key series | Daily/Monthly | FRED |
| News articles | Top 50 holdings | Every 4 hours | NewsAPI |
| SEC filings (10-K, 10-Q) | Top 100 by market cap | As filed | SEC EDGAR |

---

## 8. Proof of Performance: How to Demonstrate Verifiable Returns

> **This section is critical.** A backtested system is not proof. Employers and investors have seen too many "backtests that show 50% returns." What ACTUALLY impresses people is a live track record — even a small one — with proper attribution.

### 8.1 Realistic Return Expectations

Be honest with yourself and with employers about what's achievable:

| Strategy Type | Realistic Annual Return | Realistic Sharpe | Notes |
|---|---|---|---|
| Multi-factor long-only (vs SPY) | +2-5% alpha over benchmark | 0.5-1.0 IR | Most realistic for individual |
| Factor-timed long-only | +3-7% alpha | 0.6-1.2 IR | Requires regime detection |
| ML-enhanced factor | +4-8% alpha | 0.7-1.3 IR | Higher turnover, more complexity |
| Market-neutral (long/short) | +5-10% absolute | 0.8-1.5 Sharpe | Requires short selling capability |

**What employers actually want to see:**
- NOT: "My system returns 40% per year" (they'll assume it's overfit)
- YES: "My system generates a Sharpe ratio of 0.8 out-of-sample with an Information Ratio of 0.6 against SPY, and I can explain exactly where the alpha comes from"
- YES: "I've been running this live for 3 months. Here's the equity curve, drawdown, and factor attribution. The live performance is within 1 standard deviation of my backtest, which tells me the backtest is realistic"

### 8.2 Live Verification Setup

```python
class PerformanceVerification:
    """
    Generate verifiable performance reports.
    
    Key metrics that prove the system works:
    1. LIVE equity curve vs benchmark (even 3 months is meaningful)
    2. Backtest vs Live comparison (proves backtest isn't overfit)
    3. Factor attribution (explains WHERE returns come from)
    4. Risk metrics (proves risk management works)
    5. Transaction log (proves it's real, not cherry-picked)
    """
    
    def generate_tearsheet(self, strategy: str) -> PerformanceTearsheet:
        """
        Generates a professional performance report including:
        
        PAGE 1: Summary
        - Cumulative return chart vs benchmark
        - Key metrics table (return, vol, Sharpe, Sortino, MaxDD, Calmar)
        - Monthly returns heatmap
        
        PAGE 2: Risk Analysis
        - Drawdown chart
        - Rolling Sharpe ratio (is performance consistent or lumpy?)
        - Rolling beta (is it actually market-neutral?)
        - VaR and CVaR
        
        PAGE 3: Attribution
        - Factor exposure chart over time
        - Brinson attribution (allocation vs selection effect)
        - Sector contribution analysis
        - Top/bottom individual contributors
        
        PAGE 4: Backtest Validation
        - Walk-forward OOS equity curves (each fold plotted separately)
        - Backtest vs Live comparison chart
        - Statistical significance of alpha (t-stat, p-value)
        - Parameter sensitivity heatmap (shows it's not overfit to specific params)
        """
        pass
```

### 8.3 TFSA Paper Trading → Live Strategy

**Recommended approach for building a track record:**

1. **Weeks 1-2**: Build system, backtest extensively
2. **Weeks 3-4**: Paper trading — system generates signals, you record them but don't trade
3. **Month 2**: Compare paper signals to what would have happened → validate signal quality
4. **Month 3+**: Deposit into TFSA (tax-free, up to $7,000/year contribution room), begin live trading with real money following system signals
5. **Ongoing**: Record every trade, every signal, every daily portfolio value

**Why TFSA is perfect:**
- Tax-free gains (no capital gains tax on profits)
- Limited size forces discipline ($7K-$95K depending on cumulative room)
- Legitimate investment vehicle (not gambling)
- Easy to explain to employers

### 8.4 The Performance Dashboard That Gets You Hired

Your dashboard should have a "Live Track Record" page that shows:

```
┌─────────────────────────────────────────────────────────┐
│  FinSight Live Performance Dashboard                     │
│                                                          │
│  Strategy: Multi-Factor Equity (Long-Only)               │
│  Live Since: March 2026                                  │
│  Benchmark: S&P 500 (SPY)                               │
│                                                          │
│  ┌─────────────────────────────────────────────────────┐│
│  │ [Interactive Equity Curve: Strategy vs SPY]          ││
│  │ Shows daily portfolio value since inception          ││
│  └─────────────────────────────────────────────────────┘│
│                                                          │
│  Key Metrics (Live Period)                               │
│  ┌──────────────┬───────────┬───────────┐              │
│  │              │ Strategy  │ Benchmark │              │
│  │ Return (Ann.)│ +14.2%    │ +11.5%    │              │
│  │ Volatility   │ 13.8%     │ 15.2%     │              │
│  │ Sharpe Ratio │ 1.03      │ 0.76      │              │
│  │ Max Drawdown │ -8.3%     │ -12.1%    │              │
│  │ Info Ratio   │ 0.62      │ —         │              │
│  │ Alpha (Ann.) │ +2.7%     │ —         │              │
│  └──────────────┴───────────┴───────────┘              │
│                                                          │
│  Why It Works: Factor Attribution                        │
│  Momentum: +1.2% │ Quality: +0.8% │ Value: +0.5%       │
│  ML Signal: +0.9% │ Regime Timing: +0.3%                │
│  Transaction Costs: -1.0%                                │
│                                                          │
│  [Backtest vs Live Comparison] [Drawdown Chart]          │
│  [Monthly Returns Heatmap]    [Factor Exposure]          │
└─────────────────────────────────────────────────────────┘
```

---

## 9. Risk Management & Compliance

### 9.1 Portfolio Risk Constraints

```python
RISK_CONSTRAINTS = {
    "max_position_weight": 0.05,      # 5% max per stock
    "max_sector_weight": 0.25,        # 25% max per sector
    "max_drawdown_trigger": 0.15,     # -15% drawdown → reduce risk
    "min_cash_reserve": 0.05,         # Always keep 5% cash
    "max_leverage": 1.0,              # No leverage
    "max_turnover_monthly": 0.20,     # 20% max monthly turnover
    "stop_loss_per_position": 0.20,   # -20% per position → review
}
```

### 9.2 Disclaimer

All outputs from the system include:
> "FinSight is an AI-powered research and analysis tool for educational and informational purposes only. It does not constitute investment advice. Past performance, whether backtested or live, does not guarantee future results. Always conduct your own research and consider consulting a registered financial advisor before making investment decisions."

---

## 10. Future Enhancements (Post v1.0)

These are not in scope for the two-week sprint but should be architecturally possible:

- **MCP (Model Context Protocol) integration** — Expose your financial data as MCP tools so any LLM can query your database, run backtests, generate reports. *This is extremely hot in 2025-2026 and a massive interview talking point.*
- **Alternative data**: Satellite imagery (parking lot counts), social media sentiment, insider trading filings (SEC Form 4)
- **Options analysis**: Implied volatility surfaces, unusual options activity detection, put-call ratio as signal
- **Crypto coverage**: BTC, ETH with on-chain metrics as alternative factors
- **Chinese language interface**: 中文投资研究助手 for Vancouver Chinese community
- **Interactive Brokers API integration**: Automated trade execution via IB TWS API
- **Multi-strategy framework**: Run multiple strategies in parallel with portfolio-level risk management
- **LLM-as-Judge evaluation**: Automated quality scoring of agent outputs for continuous improvement
- **Reinforcement Learning for execution**: Optimal order execution to minimize market impact
- **Streaming data pipeline**: Kafka/Redis Streams for real-time market data processing
- **Model monitoring dashboard**: Track feature drift, prediction quality, factor decay over time

---

## 11. Interview Preparation: Technical Questions This Project Prepares You For

> Building this project equips you to answer these common interview questions at firms like CC&L, RBC, or any Vancouver fintech:

### Quantitative Finance Questions
- "Walk me through your backtesting methodology. How do you prevent overfitting?"
  → Walk-forward validation, transaction costs, multiple hypothesis correction, parameter sensitivity
- "What is look-ahead bias and how do you prevent it?"
  → Point-in-time data, filing_date vs report_date, purge gap between train/test
- "Explain factor neutralization. Why is it important?"
  → Barra-style regression, removing sector/size exposure, pure alpha extraction
- "How do you construct a portfolio from alpha signals?"
  → Cross-sectional ranking, constrained optimization, risk parity, Black-Litterman
- "What's your Sharpe ratio? Is it statistically significant?"
  → t-stat of alpha, multiple testing correction, walk-forward OOS performance

### AI / ML Engineering Questions
- "Describe your RAG architecture. How is it different from simple retrieve-and-generate?"
  → Agentic RAG, multi-hop reasoning, adaptive retrieval, self-reflection, hybrid search
- "How do you handle hallucinations in your financial analysis agent?"
  → Citation tracking, source grounding, structured output validation, post-generation checks
- "How do your agents coordinate?"
  → LangGraph supervisor pattern, conditional routing, parallel fan-out, HITL gates
- "How do you evaluate your RAG system's quality?"
  → Retrieval precision@k, answer faithfulness, citation accuracy, A/B testing
- "What's your experience with LLM-based feature engineering?"
  → News sentiment as alpha factor, LLM-assisted alpha discovery, earnings call analysis

### Data Engineering Questions
- "How do you handle financial data quality issues?"
  → Validation at ingestion, cross-source verification, outlier detection, missing data policy
- "Describe your data pipeline architecture."
  → Multi-source ingestion, TimescaleDB for time-series, scheduling, monitoring, lineage
- "How do you ensure point-in-time data consistency?"
  → filing_date tracking, temporal joins, as-of-date queries

---

## 12. Key Dependencies & API Limits

| Service | Free Tier Limits | Workaround |
|---------|------------------|------------|
| yfinance | Unofficial, may throttle | Rate limit to 1 req/sec, cache aggressively |
| Financial Modeling Prep | 250 calls/day | Cache all responses, batch requests |
| FRED | 120 calls/min | More than sufficient |
| NewsAPI | 100 calls/day | Focus on portfolio holdings only |
| Anthropic Claude | Pay per token | Use Sonnet for all agents (cost-effective), budget ~$20/month |
| Alpha Vantage | 25 calls/day | Supplementary source only |

---

## Appendix A: Instructions for Claude Code

When implementing this project, please follow these priorities:

1. **Get the skeleton working first** — A running system that does something simple end-to-end is more valuable than a perfect module that doesn't connect to anything.

2. **Data layer is the foundation** — Nothing else works without clean, reliable data. Prioritize this.

3. **Make it demo-able at every stage** — After each phase, there should be something impressive to show. A chart, a report, a dashboard page.

4. **Test as you go** — Don't leave testing for the end. Write tests alongside implementation.

5. **Commit frequently with good messages** — This is a portfolio project. The git history tells a story.

6. **Documentation is first-class** — Treat docs like code. Architecture diagram in README is mandatory.

7. **Error handling from day one** — No bare `except:`, no silent failures. Log everything.

8. **Use async for I/O** — All data fetching, API calls should be async.

9. **Type everything** — Full type hints on all function signatures. Use Pydantic for data validation.

10. **Keep it simple** — If choosing between a clever solution and a readable solution, choose readable. This code needs to be explainable in an interview.

### Implementation-Specific Notes

**For the Quant Engine:**
- Start with Tier 1 (classic) factors only. Get the full pipeline working: data → factor computation → backtesting → performance report.
- Add Tier 2 (ML-enhanced) factors only after Tier 1 is solid.
- Tier 3 (LLM alpha discovery) is a stretch goal — nice to have, not required for v1.0.
- Factor neutralization is MANDATORY — it's a key interview differentiator.
- Use vectorbt for initial backtesting (it's fast). Custom engine can come later.

**For the Agent System:**
- Start with ONE agent (Earnings Agent) working end-to-end with RAG.
- Then add the orchestrator to coordinate multiple agents.
- The Supervisor pattern in LangGraph is the correct architecture — don't use a simple sequential chain.
- ALL agent outputs MUST use Pydantic schemas — no unstructured text responses.
- Citation tracking is non-negotiable — every claim must trace to a source.

**For the Dashboard:**
- Streamlit is the right choice for speed. Don't over-engineer the frontend.
- The "Live Performance" page is the most important page for job applications.
- Make the Research Agent chat interface feel polished — this is what employers will interact with during demos.

**For Performance Proof:**
- Set up automated daily portfolio snapshots from Day 1 of going live.
- The comparison of backtest vs live performance is the MOST IMPORTANT chart.
- Even 2-3 months of live data showing backtest consistency is extremely impressive.

### Critical Dependencies to Install First
```bash
# Core
pip install sqlalchemy asyncpg pandas numpy scipy statsmodels

# Quant
pip install yfinance vectorbt lightgbm scikit-learn cvxpy shap

# AI/Agent
pip install langchain langgraph langchain-anthropic chromadb pdfplumber

# Dashboard
pip install streamlit plotly

# DevOps
pip install pytest ruff mypy pydantic

# Database
# (via docker-compose: PostgreSQL + TimescaleDB + pgvector + Redis)
```
