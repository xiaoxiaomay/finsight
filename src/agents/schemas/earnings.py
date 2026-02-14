"""Pydantic schemas for Earnings Analysis Agent output."""

from datetime import date

from pydantic import BaseModel, Field


class EarningsAnalysisOutput(BaseModel):
    """Structured output from earnings analysis."""

    symbol: str
    analysis_date: date

    # Revenue analysis
    revenue_trend: str = Field(
        description="One of: accelerating, stable, decelerating, declining"
    )
    revenue_growth_yoy: float = Field(description="Year-over-year revenue growth rate")
    revenue_surprise_pct: float = Field(
        default=0.0, description="Revenue surprise vs consensus if available"
    )

    # Profitability
    margin_trend: str = Field(
        description="One of: expanding, stable, compressing"
    )
    gross_margin: float
    operating_margin: float
    net_margin: float
    margin_expansion: bool

    # Cash flow
    fcf_yield: float = Field(description="Free cash flow yield")
    cash_conversion: float = Field(description="FCF / Net Income ratio")
    capex_trend: str = Field(
        description="One of: increasing, stable, decreasing"
    )

    # Balance sheet
    debt_to_equity: float
    current_ratio: float
    balance_sheet_quality: str = Field(
        description="One of: strong, adequate, concerning, weak"
    )

    # Management commentary (from RAG on earnings call / 10-K)
    key_guidance_points: list[str] = Field(default_factory=list)
    risk_factors_highlighted: list[str] = Field(default_factory=list)
    management_tone: str = Field(
        description="One of: confident, cautious, defensive, optimistic"
    )

    # Overall assessment
    fundamental_score: float = Field(ge=0, le=100, description="0-100 score")
    investment_thesis: str = Field(description="2-3 sentence summary")
    key_risks: list[str] = Field(default_factory=list)
    catalysts: list[str] = Field(default_factory=list)

    # Citations
    sources_used: list[str] = Field(default_factory=list, description="Document references")
    confidence_level: str = Field(
        description="One of: high, medium, low"
    )
