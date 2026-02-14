"""Pydantic schemas for Regime Detector output."""

from datetime import date

from pydantic import BaseModel, Field


class RegimeOutput(BaseModel):
    """Market regime detection output."""

    assessment_date: date
    regime: str = Field(
        description="One of: bull, bear, high_vol, recovery, neutral"
    )
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)

    # Indicator readings
    sp500_vs_200dma: float = Field(
        default=0.0,
        description="Percentage above/below 200-day moving average",
    )
    vix_level: float = Field(default=0.0)
    yield_curve_spread: float = Field(
        default=0.0, description="10Y - 2Y Treasury spread"
    )
    sp500_trend: str = Field(
        default="flat", description="One of: up, flat, down"
    )

    # Factor tilt recommendations
    factor_tilt: str = Field(
        default="balanced",
        description="Recommended factor emphasis for current regime",
    )
    recommended_factors: list[str] = Field(default_factory=list)
    factors_to_avoid: list[str] = Field(default_factory=list)

    # Portfolio guidance
    suggested_equity_weight: float = Field(ge=0.0, le=1.0, default=0.6)
    defensive_tilt: bool = False
    reasoning: str = ""
