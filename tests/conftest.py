"""Shared pytest fixtures for FinSight test suite."""

from datetime import date

import pandas as pd
import pytest


@pytest.fixture
def sample_market_data() -> pd.DataFrame:
    """Sample OHLCV data for testing."""
    return pd.DataFrame({
        "symbol": ["AAPL"] * 5,
        "date": pd.date_range("2024-01-02", periods=5, freq="B"),
        "open": [185.0, 186.5, 187.0, 185.5, 188.0],
        "high": [187.0, 188.0, 188.5, 187.0, 190.0],
        "low": [184.0, 185.5, 186.0, 184.5, 187.5],
        "close": [186.5, 187.0, 186.0, 186.5, 189.0],
        "adj_close": [186.5, 187.0, 186.0, 186.5, 189.0],
        "volume": [50_000_000, 48_000_000, 52_000_000, 45_000_000, 55_000_000],
    })


@pytest.fixture
def sample_fundamentals() -> dict:
    """Sample fundamental data for a single filing."""
    return {
        "symbol": "AAPL",
        "report_date": date(2024, 9, 30),
        "filing_date": date(2024, 10, 31),
        "period_type": "annual",
        "revenue": 391_035_000_000,
        "net_income": 93_736_000_000,
        "total_assets": 352_583_000_000,
        "total_equity": 62_146_000_000,
        "operating_cash_flow": 110_543_000_000,
        "eps": 6.13,
        "book_value_per_share": 4.38,
    }


@pytest.fixture
def sample_factor_scores() -> pd.DataFrame:
    """Sample cross-sectional factor scores."""
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
    return pd.DataFrame({
        "symbol": symbols,
        "momentum_12_1": [0.15, 0.22, -0.05, 0.30, 0.10],
        "earnings_yield": [0.035, 0.030, 0.025, 0.015, 0.040],
        "roe": [1.47, 0.38, 0.29, 0.17, 0.33],
        "volatility_60d": [0.22, 0.20, 0.25, 0.28, 0.30],
    })


@pytest.fixture
def sample_macro_data() -> pd.DataFrame:
    """Sample macro indicator time series."""
    dates = pd.date_range("2024-01-01", periods=12, freq="MS")
    return pd.DataFrame({
        "date": dates,
        "fed_funds_rate": [5.33] * 3 + [5.33] * 3 + [5.08] * 3 + [4.83] * 3,
        "cpi_yoy": [3.1, 3.2, 3.5, 3.4, 3.3, 3.0, 2.9, 2.5, 2.4, 2.6, 2.7, 2.9],
        "unemployment": [3.7, 3.7, 3.8, 3.9, 4.0, 4.1, 4.2, 4.2, 4.1, 4.1, 4.2, 4.2],
    })
