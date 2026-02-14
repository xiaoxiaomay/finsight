"""Pydantic schemas for Research Report output."""

from datetime import datetime

from pydantic import BaseModel, Field


class ConflictNote(BaseModel):
    """Conflict detected between agent outputs."""

    agents: list[str]
    description: str
    resolution: str = ""


class ResearchReportOutput(BaseModel):
    """Full research report combining all agent outputs."""

    symbol: str
    company_name: str = ""
    generated_at: datetime = Field(default_factory=datetime.now)
    report_version: str = "1.0"

    # Recommendation
    recommendation: str = Field(
        description="One of: Strong Buy, Buy, Hold, Sell, Strong Sell"
    )
    conviction: str = Field(
        default="medium",
        description="One of: high, medium, low"
    )
    composite_score: float = Field(
        ge=0, le=100, description="Overall score 0-100"
    )
    target_price: float | None = None

    # Executive summary
    executive_summary: str = ""
    bull_case: list[str] = Field(default_factory=list)
    bear_case: list[str] = Field(default_factory=list)

    # Component scores
    fundamental_score: float | None = None
    quant_score: float | None = None
    sentiment_score: float | None = None
    macro_score: float | None = None
    peer_score: float | None = None

    # Agent outputs (raw data)
    earnings_analysis: dict | None = None
    macro_assessment: dict | None = None
    news_sentiment: dict | None = None
    quant_signals: dict | None = None
    peer_comparison: dict | None = None

    # Quality
    conflicts: list[ConflictNote] = Field(default_factory=list)
    agents_used: list[str] = Field(default_factory=list)
    data_sources: list[str] = Field(default_factory=list)

    # Risk
    key_risks: list[str] = Field(default_factory=list)
    catalysts: list[str] = Field(default_factory=list)

    # Regime context
    market_regime: str = ""
    regime_factor_tilt: str = ""
