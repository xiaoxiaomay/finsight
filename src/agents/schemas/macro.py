"""Pydantic schemas for Macro Environment Agent output."""

from datetime import date

from pydantic import BaseModel, Field


class MacroAssessmentOutput(BaseModel):
    """Structured macro environment assessment."""

    assessment_date: date

    # Economic cycle
    cycle_phase: str = Field(
        description="One of: expansion, peak, contraction, trough"
    )
    cycle_confidence: float = Field(ge=0.0, le=1.0)

    # Key indicators
    gdp_growth_trend: str = Field(
        description="One of: accelerating, stable, decelerating, contracting"
    )
    inflation_trend: str = Field(
        description="One of: rising, stable, falling"
    )
    employment_trend: str = Field(
        description="One of: strengthening, stable, weakening"
    )
    yield_curve_status: str = Field(
        description="One of: normal, flat, inverted"
    )

    # Policy environment
    fed_policy_stance: str = Field(
        description="One of: hawkish, neutral, dovish"
    )
    rate_direction: str = Field(
        description="One of: hiking, pausing, cutting"
    )

    # Market implications
    equity_outlook: str = Field(
        description="One of: bullish, neutral, bearish"
    )
    sector_preferences: list[str] = Field(default_factory=list)
    risk_factors: list[str] = Field(default_factory=list)

    # Actionable for portfolio
    suggested_equity_allocation: float = Field(
        ge=0.0, le=1.0, description="Suggested equity allocation (0.0 to 1.0)"
    )
    defensive_tilt: bool
    reasoning: str
