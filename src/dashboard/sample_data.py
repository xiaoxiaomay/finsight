"""Sample data generators for dashboard demo mode.

Provides realistic synthetic data so the dashboard works
without a live database connection.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st


@st.cache_data(ttl=3600)
def get_sample_prices(
    n_symbols: int = 50,
    n_days: int = 1260,
    start_date: str = "2020-01-01",
) -> pd.DataFrame:
    """Generate synthetic price matrix (wide format)."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range(start_date, periods=n_days)
    symbols = [f"SYM{i:03d}" for i in range(n_symbols)]

    market = rng.normal(0.0003, 0.01, n_days)
    returns = np.zeros((n_days, n_symbols))
    for j in range(n_symbols):
        beta = 0.7 + (j % 10) * 0.1
        idio = 0.005 + (j % 5) * 0.003
        returns[:, j] = beta * market + rng.normal(0.00005, idio, n_days)

    prices = 100 * np.exp(np.cumsum(returns, axis=0))
    return pd.DataFrame(prices, index=dates, columns=symbols)


@st.cache_data(ttl=3600)
def get_sample_fundamentals(symbols: list[str]) -> pd.DataFrame:
    """Generate synthetic fundamentals data."""
    rng = np.random.default_rng(42)
    rows = []
    for sym in symbols:
        for year in range(2019, 2026):
            rows.append({
                "symbol": sym,
                "filing_date": pd.Timestamp(f"{year}-03-15"),
                "report_date": pd.Timestamp(f"{year - 1}-12-31"),
                "period_type": "annual",
                "eps": rng.uniform(2, 15),
                "book_value_per_share": rng.uniform(10, 80),
                "net_income": rng.uniform(1e8, 5e9),
                "total_equity": rng.uniform(5e8, 50e9),
                "total_assets": rng.uniform(1e9, 100e9),
                "total_liabilities": rng.uniform(5e8, 60e9),
                "gross_profit": rng.uniform(5e8, 20e9),
                "operating_income": rng.uniform(2e8, 10e9),
                "operating_cash_flow": rng.uniform(3e8, 15e9),
                "shares_outstanding": rng.integers(int(1e8), int(5e9)),
            })
    return pd.DataFrame(rows)


@st.cache_data(ttl=3600)
def get_sample_portfolio() -> dict:
    """Generate sample portfolio holdings and performance."""
    rng = np.random.default_rng(42)
    symbols = [f"SYM{i:03d}" for i in range(20)]
    sectors = [
        "Tech", "Health", "Finance", "Consumer", "Energy",
        "Tech", "Health", "Finance", "Consumer", "Energy",
        "Industrial", "Materials", "Tech", "Health", "Finance",
        "Consumer", "Energy", "Industrial", "Materials", "Utilities",
    ]
    weights = rng.dirichlet(np.ones(20))

    holdings = pd.DataFrame({
        "Symbol": symbols,
        "Sector": sectors,
        "Shares": rng.integers(100, 5000, size=20),
        "Avg Cost": rng.uniform(50, 200, size=20).round(2),
        "Current Price": rng.uniform(50, 250, size=20).round(2),
        "Weight": (weights * 100).round(2),
    })
    holdings["Market Value"] = (holdings["Shares"] * holdings["Current Price"]).round(2)
    holdings["P&L"] = (
        (holdings["Current Price"] - holdings["Avg Cost"]) * holdings["Shares"]
    ).round(2)
    holdings["P&L %"] = (
        (holdings["Current Price"] / holdings["Avg Cost"] - 1) * 100
    ).round(2)

    # Generate portfolio returns (daily, ~3 years)
    n_days = 756
    dates = pd.bdate_range("2023-01-01", periods=n_days)
    port_ret = rng.normal(0.0004, 0.009, n_days)
    bench_ret = rng.normal(0.0003, 0.010, n_days)

    port_value = 1_000_000 * np.cumprod(1 + port_ret)
    bench_value = 1_000_000 * np.cumprod(1 + bench_ret)

    # Recent trades
    trade_dates = pd.bdate_range("2025-01-01", periods=10)
    trades = pd.DataFrame({
        "Date": trade_dates,
        "Symbol": rng.choice(symbols, 10),
        "Side": rng.choice(["BUY", "SELL"], 10),
        "Shares": rng.integers(50, 500, size=10),
        "Price": rng.uniform(80, 200, size=10).round(2),
    })

    return {
        "holdings": holdings,
        "portfolio_returns": pd.Series(port_ret, index=dates),
        "benchmark_returns": pd.Series(bench_ret, index=dates),
        "portfolio_value": pd.Series(port_value, index=dates),
        "benchmark_value": pd.Series(bench_value, index=dates),
        "trades": trades,
    }


def get_sector_map(symbols: list[str]) -> dict[str, str]:
    """Map symbols to sectors."""
    sector_names = [
        "Tech", "Health", "Finance", "Consumer", "Energy",
        "Industrial", "Materials", "Utilities", "RealEstate", "Telecom",
    ]
    return {sym: sector_names[i % len(sector_names)] for i, sym in enumerate(symbols)}
