# System Architecture

## Overview

FinSight is a four-layer platform: **Data → Quant → AI → Dashboard**.
Each layer is independently testable and communicates via well-defined interfaces.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         DASHBOARD (Streamlit)                          │
│  Portfolio │ AI Research │ Backtest Lab │ Risk │ Live Perf │ Explorer  │
└──────┬──────────┬───────────┬────────────┬──────────┬──────────┬──────┘
       │          │           │            │          │          │
┌──────▼──────────▼───────────▼────────────▼──────────▼──────────▼──────┐
│                        AI AGENT SYSTEM                                │
│  ┌────────────┐  ┌───────┐  ┌───────┐  ┌──────┐  ┌─────────┐        │
│  │ Supervisor │──│Earnings│──│ Macro │──│ News │──│Quant Sig│        │
│  │ (LangGraph)│  │ Agent  │  │ Agent │  │Agent │  │  Agent  │        │
│  └──────┬─────┘  └───┬───┘  └───┬───┘  └──┬───┘  └────┬────┘        │
│         │            │          │          │           │              │
│         ▼            ▼          │          │           │              │
│    Aggregator   RAG Pipeline    │          │           │              │
│    + Report     (chunk→embed    │          │           │              │
│    Generator     →retrieve)     │          │           │              │
└──────┬──────────────┬───────────┼──────────┼───────────┼──────────────┘
       │              │           │          │           │
┌──────▼──────────────▼───────────▼──────────▼───────────▼──────────────┐
│                        QUANT ENGINE                                   │
│  ┌──────────────┐  ┌───────────────┐  ┌────────────┐  ┌───────────┐  │
│  │ Factor Lib   │  │  Backtest     │  │ Portfolio   │  │  Regime   │  │
│  │ 10 Classic   │  │  Walk-Forward │  │ Optimizer   │  │ Detector  │  │
│  │ + Composite  │  │  + Statistics │  │ + Risk      │  │           │  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬──────┘  └─────┬─────┘  │
└─────────┼─────────────────┼─────────────────┼────────────────┼────────┘
          │                 │                 │                │
┌─────────▼─────────────────▼─────────────────▼────────────────▼────────┐
│                        DATA PLATFORM                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                │
│  │ Market Data  │  │ Fundamentals │  │    Macro     │                │
│  │  (yfinance)  │  │    (FMP)     │  │   (FRED)     │                │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘                │
│         │                 │                 │                         │
│         ▼                 ▼                 ▼                         │
│    Quality Validation (PIT enforcement, gap detection, monitoring)    │
└──────────────────────────────┬────────────────────────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │   PostgreSQL 16     │
                    │  + TimescaleDB      │
                    │  + pgvector         │
                    │  ───────────────    │
                    │   Redis 7 Cache     │
                    └─────────────────────┘
```

## Data Flow

### Ingestion Pipeline
```
External APIs → Fetcher → Validator → Quality Logger → PostgreSQL
                 │                       │
                 └── Rate Limiter ◄──── Redis
```

1. **Market Data**: yfinance → daily OHLCV → `market_data` table (TimescaleDB hypertable)
2. **Fundamentals**: FMP API → income/balance/cashflow → `fundamentals` table (PIT-enforced)
3. **Macro**: FRED API → economic indicators → `macro_indicators` table
4. **SEC Filings**: EDGAR → 10-K/10-Q text → chunked → embedded → `sec_filings` + pgvector

### Factor Pipeline
```
market_data + fundamentals → Factor Library (10 factors)
                              → Cross-sectional Z-Score
                              → Composite Score
                              → factor_signals table
```

### AI Research Pipeline
```
User Query → Supervisor (classify) → Fan-out to Agents
                                       → Earnings (RAG + calc)
                                       → Macro (FRED)
                                       → News (sentiment)
                                       → Quant (factor DB)
                                       → Peer (comparison)
                                     → Aggregator (score + conflicts)
                                     → Report Generator (HTML)
```

### Dashboard Pipeline
```
PostgreSQL → Streamlit Cache (@st.cache_data)
           → Plotly Charts
           → Interactive UI
           → User
```

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Database | PostgreSQL + TimescaleDB | Time-series optimization, mature ecosystem |
| Embeddings | pgvector | Co-located with data, no separate vector DB |
| LLM | Claude API | Structured output, long context for filings |
| Agent framework | LangGraph | StateGraph for complex routing, fan-out |
| Dashboard | Streamlit | Rapid prototyping, native Python, Plotly support |
| Factor framework | Custom | Full control over PIT enforcement, walk-forward |
| Testing | pytest + mocks | Fast CI, no external dependencies needed |

## Deployment

```
docker-compose up -d          # PostgreSQL + Redis
pip install -e ".[dev]"       # Python environment
python scripts/seed_data.py   # Initial data load
streamlit run src/dashboard/app.py  # Launch dashboard
```

For production:
- Add Nginx reverse proxy with SSL
- Use Kubernetes or ECS for scaling
- Add Prometheus/Grafana for monitoring
- Schedule daily_update.py via Airflow or cron
