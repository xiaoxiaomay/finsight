"""Pydantic schemas for Peer Comparison Agent output."""

from datetime import date

from pydantic import BaseModel, Field


class PeerMetrics(BaseModel):
    """Key metrics for a single peer company."""

    symbol: str
    company_name: str = ""
    # Valuation
    pe_ratio: float | None = None
    pb_ratio: float | None = None
    ev_ebitda: float | None = None
    # Profitability
    roe: float | None = None
    gross_margin: float | None = None
    operating_margin: float | None = None
    net_margin: float | None = None
    # Growth
    revenue_growth_yoy: float | None = None
    earnings_growth_yoy: float | None = None
    # Size
    market_cap: float | None = None


class PeerComparisonOutput(BaseModel):
    """Structured output from peer comparison analysis."""

    symbol: str
    analysis_date: date
    sector: str = ""
    industry: str = ""

    # Target company metrics
    target_metrics: PeerMetrics

    # Peer companies
    peers: list[PeerMetrics] = Field(default_factory=list)
    peer_count: int = 0

    # Relative valuation
    pe_vs_peers: str = Field(
        default="in_line",
        description="One of: premium, in_line, discount"
    )
    pb_vs_peers: str = Field(default="in_line")
    ev_ebitda_vs_peers: str = Field(default="in_line")
    valuation_summary: str = ""

    # Relative quality
    profitability_rank: int = Field(
        default=0, description="Rank among peers (1 = best)"
    )
    growth_rank: int = Field(default=0)
    overall_rank: int = Field(default=0)

    # Key takeaways
    competitive_advantages: list[str] = Field(default_factory=list)
    competitive_weaknesses: list[str] = Field(default_factory=list)
    peer_comparison_summary: str = ""

    confidence_level: str = Field(default="medium")
