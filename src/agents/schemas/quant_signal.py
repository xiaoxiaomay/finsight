"""Pydantic schemas for Quant Signal Agent output."""

from datetime import date

from pydantic import BaseModel, Field


class FactorScore(BaseModel):
    """Individual factor score for a stock."""

    factor_name: str
    raw_value: float = 0.0
    z_score: float = 0.0
    percentile: float = Field(default=50.0, ge=0, le=100)


class BacktestStats(BaseModel):
    """Historical backtest performance statistics."""

    total_return: float = 0.0
    annualized_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    period: str = ""


class QuantSignalOutput(BaseModel):
    """Structured output from quant signal analysis."""

    symbol: str
    analysis_date: date

    # Factor scores
    factor_scores: list[FactorScore] = Field(default_factory=list)
    composite_score: float = Field(
        default=0.0, description="Weighted composite factor z-score"
    )
    composite_percentile: float = Field(
        default=50.0, ge=0, le=100,
        description="Percentile rank among universe"
    )

    # Signal
    signal_direction: str = Field(
        default="neutral",
        description="One of: strong_long, long, neutral, short, strong_short"
    )
    signal_strength: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Signal conviction 0.0 to 1.0"
    )

    # Historical backtest
    backtest_stats: BacktestStats | None = None

    # Factor tilts
    strongest_factors: list[str] = Field(default_factory=list)
    weakest_factors: list[str] = Field(default_factory=list)

    # Context
    sector: str = ""
    market_cap_bucket: str = Field(
        default="", description="One of: mega, large, mid, small"
    )
    confidence_level: str = Field(default="medium", description="One of: high, medium, low")
