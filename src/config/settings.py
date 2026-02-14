"""Application settings loaded from environment variables.

Uses Pydantic Settings for validation and type coercion.
All config flows through this single module.
"""

from datetime import date
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """FinSight application configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Database
    database_url: str = "postgresql://finsight:finsight@localhost:5432/finsight"
    redis_url: str = "redis://localhost:6379"

    # API Keys
    anthropic_api_key: str = ""
    fmp_api_key: str = ""
    fred_api_key: str = ""
    news_api_key: str = ""
    alpha_vantage_api_key: str = ""

    # Project Configuration
    market_data_start_date: date = Field(default=date(2020, 1, 1))
    universe: str = "SP500"
    rebalance_frequency: str = "monthly"
    log_level: str = "INFO"

    # Derived async database URL (for asyncpg)
    @property
    def async_database_url(self) -> str:
        return self.database_url.replace("postgresql://", "postgresql+asyncpg://")

    # Quant Engine Defaults
    backtest_commission_per_share: float = 0.005
    backtest_spread_cost_bps: float = 5.0
    backtest_market_impact_bps: float = 10.0
    max_position_weight: float = 0.05
    max_sector_weight: float = 0.25

    # Agent System
    llm_model: str = "claude-sonnet-4-20250514"
    llm_max_tokens: int = 4096
    llm_temperature: float = 0.1

    # Rate Limiting
    yfinance_rate_limit: float = 1.0  # seconds between requests
    fmp_daily_limit: int = 250
    news_api_daily_limit: int = 100


@lru_cache
def get_settings() -> Settings:
    """Get cached application settings singleton."""
    return Settings()
