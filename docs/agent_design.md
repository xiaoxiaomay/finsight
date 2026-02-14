# FinSight Agent System Architecture

## Overview

FinSight uses a multi-agent architecture orchestrated by a LangGraph StateGraph supervisor.
Each agent specializes in a domain (fundamentals, macro, news, quant, peers) and produces
structured Pydantic outputs that are aggregated into comprehensive research reports.

## Architecture Diagram

```
User Query
    │
    ▼
┌─────────────────────┐
│   Supervisor/Router  │  Classifies query → selects agents
│   (LangGraph)        │
└─────┬───────────────┘
      │
      ├─ Simple query ──► Single Agent ──► Aggregator ──► Report
      │
      └─ Full analysis ─► Fan-out to ALL agents (parallel)
                          │
            ┌─────────────┼─────────────┐
            ▼             ▼             ▼
      ┌──────────┐ ┌──────────┐ ┌──────────┐
      │ Earnings │ │  Macro   │ │   News   │ ...
      │  Agent   │ │  Agent   │ │  Agent   │
      └────┬─────┘ └────┬─────┘ └────┬─────┘
           │             │             │
           └─────────────┼─────────────┘
                         ▼
               ┌──────────────────┐
               │   Aggregator     │  Conflict detection + scoring
               └────────┬────────┘
                         ▼
               ┌──────────────────┐
               │ Report Generator │  Professional HTML output
               └──────────────────┘
```

## Agents

### 1. Earnings Analysis Agent (`src/agents/earnings_agent.py`)

**Purpose:** Analyze company fundamentals using SEC filings and financial data.

**Tools:**
- RAG Retriever (SEC 10-K/10-Q filings via hybrid search)
- Financial Calculator (`src/agents/tools/financial_calc.py`)
- SEC EDGAR downloader (`src/agents/tools/sec_edgar.py`)

**Output:** `EarningsAnalysisOutput`
- Revenue analysis (trend, growth, surprise)
- Profitability (margins, expansion)
- Cash flow (FCF yield, cash conversion)
- Balance sheet quality
- Management commentary from RAG
- Fundamental score (0-100)

**Prompt:** `src/agents/prompts/earnings_analysis.py`

---

### 2. Macro Environment Agent (`src/agents/macro_agent.py`)

**Purpose:** Assess macroeconomic conditions and market implications.

**Tools:**
- FRED API client (`src/agents/tools/fred_api.py`)
- Yield curve analysis
- Fed policy stance assessment

**Output:** `MacroAssessmentOutput`
- Economic cycle phase (expansion/peak/contraction/trough)
- Key indicator trends (GDP, inflation, employment)
- Fed policy and rate direction
- Equity outlook and sector preferences
- Suggested equity allocation (0.0-1.0)

**Prompt:** `src/agents/prompts/macro_assessment.py`

---

### 3. News Sentiment Agent (`src/agents/news_agent.py`)

**Purpose:** Analyze recent news sentiment for a company.

**Tools:**
- NewsAPI client (`src/agents/tools/news_api.py`)

**Output:** `NewsSentimentOutput`
- Overall sentiment score (-1.0 to +1.0)
- Per-article sentiment and relevance scores
- Key themes extraction
- Sentiment trend (improving/stable/deteriorating)
- Market impact assessment

**Prompt:** `src/agents/prompts/news_synthesis.py`

---

### 4. Quant Signal Agent (`src/agents/quant_signal_agent.py`)

**Purpose:** Analyze quantitative factor signals for systematic trading decisions.

**Tools:**
- Factor signal database (10 classic factors)
- Composite scoring engine
- Historical backtest statistics

**Output:** `QuantSignalOutput`
- Individual factor scores (z-score, percentile)
- Weighted composite score
- Signal direction (strong_long to strong_short)
- Strongest/weakest factor exposures
- Historical backtest context

**Prompt:** `src/agents/prompts/quant_signal.py`

**Factor Universe (10 Factors):**

| Category  | Factor              | Description                        |
|-----------|---------------------|------------------------------------|
| Momentum  | momentum_12_1       | 12-month return skip last month    |
| Momentum  | short_term_reversal | Past 1-month return (negated)      |
| Value     | earnings_yield      | E/P ratio                          |
| Value     | book_to_market      | Book value / Price                 |
| Value     | ev_ebitda           | EV/EBITDA (negated)                |
| Quality   | roe                 | Return on equity                   |
| Quality   | gross_profitability | Gross profit / Total assets        |
| Quality   | accruals            | (NI - OCF) / Assets (negated)      |
| Quality   | asset_growth        | YoY asset growth (negated)         |
| Low Vol   | volatility_60d      | 60-day rolling vol (negated)       |

---

### 5. Peer Comparison Agent (`src/agents/peer_comparison_agent.py`)

**Purpose:** Compare a company against sector peers on key metrics.

**Tools:**
- GICS sector peer mapping (11 sectors, ~10 peers each)
- Financial metrics comparison engine

**Output:** `PeerComparisonOutput`
- Target company metrics (valuation, profitability, growth)
- Peer company metrics
- Relative positioning (premium/discount/in_line)
- Profitability and growth rankings
- Competitive advantages and weaknesses

**Prompt:** `src/agents/prompts/peer_comparison.py`

---

## Supervisor / Router (`src/agents/supervisor.py`)

Built with **LangGraph StateGraph**.

### State Definition

```python
class AgentState(TypedDict):
    query: str              # User input
    symbol: str             # Stock ticker
    analysis_date: str      # ISO date
    plan: list[str]         # Agents to invoke
    earnings_result: dict   # Agent outputs
    macro_result: dict
    news_result: dict
    quant_result: dict
    peer_result: dict
    regime_result: dict
    final_report: dict      # Aggregated output
```

### Query Classification Rules

| Query Pattern                          | Agents Invoked        |
|----------------------------------------|-----------------------|
| "Analyze X" / "full report"            | All 5 agents          |
| "factor scores" / "quant signal"       | Quant Signal only     |
| "earnings" / "revenue" / "margins"     | Earnings only         |
| "macro" / "GDP" / "interest rates"     | Macro only            |
| "news" / "sentiment" / "headlines"     | News only             |
| "peer" / "competitor" / "comparison"   | Peer only             |
| Multiple keywords                      | Multiple agents       |

### Graph Nodes

1. **Router** - Classifies query, sets plan
2. **Agent nodes** (5) - Each wraps a domain agent
3. **Aggregator** - Combines results, detects conflicts

### Conditional Routing

```
router → [selected_agent_1, selected_agent_2, ...] → aggregator → END
```

---

## Result Aggregator (`src/agents/aggregator.py`)

### Scoring Weights

| Agent        | Weight |
|--------------|--------|
| Earnings     | 35%    |
| Quant Signal | 25%    |
| News         | 15%    |
| Macro        | 15%    |
| Peer         | 10%    |

### Score Normalization

- Earnings: `fundamental_score` (already 0-100)
- Quant: `composite_percentile` (already 0-100)
- News: `(sentiment + 1) * 50` (maps [-1,1] to [0,100])
- Macro: `equity_allocation * 100` (maps [0,1] to [0,100])
- Peer: `(1 - rank/count) * 100` (inverse rank)

### Recommendation Thresholds

| Score Range | Recommendation |
|-------------|----------------|
| >= 80       | Strong Buy     |
| >= 65       | Buy            |
| >= 45       | Hold           |
| >= 30       | Sell           |
| < 30        | Strong Sell    |

### Conflict Detection

Checks for disagreements between:
- Earnings (positive) vs News (negative)
- Quant (bullish) vs Macro (bearish)

---

## Regime Detector (`src/agents/regime_detector.py`)

Rule-based market regime classification.

### Regime Rules

| Regime    | Conditions                                           |
|-----------|------------------------------------------------------|
| Bull      | SP500 > 200DMA AND VIX < 20 AND yield curve > 0     |
| Bear      | SP500 < 200DMA AND VIX > 25                          |
| High Vol  | VIX > 30 (takes priority)                             |
| Recovery  | SP500 crossing above 200DMA                          |
| Neutral   | None of the above                                    |

### Factor Tilts by Regime

| Regime    | Recommended Factors                | Avoid              |
|-----------|------------------------------------|---------------------|
| Bull      | momentum, earnings_yield, ROE      | low_vol, reversal   |
| Bear      | low_vol, quality, accruals         | momentum, value     |
| High Vol  | low_vol, accruals, quality         | momentum, reversal  |
| Recovery  | value, reversal, earnings_yield    | low_vol             |
| Neutral   | ROE, profitability, earnings_yield | (none)              |

---

## Report Generator (`src/agents/report_generator.py`)

Produces professional HTML research reports with:

1. **Header** - FinSight branding, date
2. **Title Bar** - Symbol, recommendation badge (color-coded)
3. **Score Cards** - Composite + 5 component scores with progress bars
4. **Executive Summary** - Auto-generated from agent outputs
5. **Bull/Bear Case** - Side-by-side columns
6. **Signal Conflicts** - Highlighted disagreements
7. **Financial Analysis** - Metrics grid, thesis, guidance
8. **Quantitative Signals** - Factor table with z-scores
9. **Industry & Peers** - Peer comparison table
10. **Macro Environment** - Indicator grid, assessment
11. **News & Sentiment** - Article table, themes
12. **Risk Factors & Catalysts** - Side-by-side
13. **Market Regime** - Current regime and factor tilt
14. **Source Citations** - Data provenance
15. **Footer** - Disclaimer

### Usage

```bash
python scripts/generate_report.py AAPL
python scripts/generate_report.py MSFT --output reports/msft.html
```

---

## RAG Pipeline (`src/agents/rag/`)

### Components

| Module              | Purpose                                    |
|---------------------|--------------------------------------------|
| `chunking.py`       | Section-aware text splitting (~1000 tokens) |
| `embeddings.py`     | all-MiniLM-L6-v2 (384d) + hash fallback   |
| `retriever.py`      | Hybrid: semantic + BM25 + metadata filters |
| `reranker.py`       | Cross-encoder reranking (optional)         |
| `document_processor.py` | End-to-end: download → parse → embed  |

### Retrieval Flow

```
Query → Embed → Vector Search (cosine)  ─┐
                                          ├─ RRF Fusion → Top-K → [Rerank] → Results
Query → Tokenize → BM25 Search          ─┘
```

---

## Testing Strategy

All agents tested with **mocked LLM responses** (MagicMock on `anthropic.Anthropic`).

| Test File                    | Tests | What's Tested                    |
|------------------------------|-------|----------------------------------|
| test_earnings_agent.py       | 7     | Structured output, RAG, citations|
| test_macro_agent.py          | 5     | Data formatting, LLM prompt      |
| test_news_agent.py           | 7     | Sentiment scoring, empty articles|
| test_quant_signal_agent.py   | 7     | Factor scoring, signal direction |
| test_peer_comparison_agent.py| 6     | Peer ranking, relative valuation |
| test_supervisor.py           | 14    | Query routing, graph construction|
| test_aggregator.py           | 13    | Scoring, conflicts, conviction   |
| test_report_generator.py     | 10    | HTML output, all sections        |
| test_regime_detector.py      | 9     | Regime rules, factor tilts       |
| test_schemas.py              | 8     | Pydantic validation              |
| test_rag.py                  | 21    | Chunking, embeddings, retrieval  |

**Total: 185+ tests** across the full project.

---

## Configuration

Agent settings in `src/config/settings.py`:

```python
llm_model = "claude-sonnet-4-20250514"
llm_max_tokens = 4096
llm_temperature = 0.1
```

All API keys loaded from `.env`:
- `ANTHROPIC_API_KEY` - Claude API
- `FRED_API_KEY` - Federal Reserve data
- `NEWS_API_KEY` - NewsAPI.org
- `FMP_API_KEY` - Financial Modeling Prep
