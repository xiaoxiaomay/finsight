"""Tests for dashboard sample data generators."""

import pandas as pd

from src.dashboard.sample_data import (
    get_sample_fundamentals,
    get_sample_portfolio,
    get_sample_prices,
    get_sector_map,
)


class TestGetSamplePrices:
    def test_returns_dataframe(self):
        df = get_sample_prices()
        assert isinstance(df, pd.DataFrame)

    def test_shape(self):
        df = get_sample_prices(n_symbols=10, n_days=100)
        assert df.shape == (100, 10)

    def test_columns_are_symbols(self):
        df = get_sample_prices(n_symbols=5)
        assert list(df.columns) == [f"SYM{i:03d}" for i in range(5)]

    def test_index_is_business_days(self):
        df = get_sample_prices(n_days=10)
        assert isinstance(df.index, pd.DatetimeIndex)
        # Business days have no Saturday (5) or Sunday (6)
        assert all(d.weekday() < 5 for d in df.index)

    def test_prices_positive(self):
        df = get_sample_prices()
        assert (df > 0).all().all()

    def test_deterministic_with_seed(self):
        df1 = get_sample_prices.__wrapped__(n_symbols=5, n_days=50)
        df2 = get_sample_prices.__wrapped__(n_symbols=5, n_days=50)
        pd.testing.assert_frame_equal(df1, df2)


class TestGetSampleFundamentals:
    def test_returns_dataframe(self):
        syms = ["A", "B", "C"]
        df = get_sample_fundamentals.__wrapped__(syms)
        assert isinstance(df, pd.DataFrame)

    def test_has_required_columns(self):
        df = get_sample_fundamentals.__wrapped__(["A"])
        required = {"symbol", "filing_date", "report_date", "eps", "net_income"}
        assert required.issubset(set(df.columns))

    def test_rows_per_symbol(self):
        syms = ["X", "Y"]
        df = get_sample_fundamentals.__wrapped__(syms)
        # 2019-2025 = 7 years per symbol
        assert len(df) == 2 * 7


class TestGetSamplePortfolio:
    def test_returns_dict(self):
        data = get_sample_portfolio.__wrapped__()
        assert isinstance(data, dict)

    def test_has_required_keys(self):
        data = get_sample_portfolio.__wrapped__()
        keys = {"holdings", "portfolio_returns", "benchmark_returns",
                "portfolio_value", "benchmark_value", "trades"}
        assert keys.issubset(set(data.keys()))

    def test_holdings_shape(self):
        data = get_sample_portfolio.__wrapped__()
        assert data["holdings"].shape[0] == 20

    def test_holdings_has_pnl(self):
        data = get_sample_portfolio.__wrapped__()
        assert "P&L" in data["holdings"].columns
        assert "P&L %" in data["holdings"].columns

    def test_returns_length(self):
        data = get_sample_portfolio.__wrapped__()
        assert len(data["portfolio_returns"]) == 756

    def test_trades_shape(self):
        data = get_sample_portfolio.__wrapped__()
        assert data["trades"].shape[0] == 10


class TestGetSectorMap:
    def test_returns_dict(self):
        result = get_sector_map(["A", "B", "C"])
        assert isinstance(result, dict)
        assert len(result) == 3

    def test_maps_all_symbols(self):
        syms = ["X", "Y", "Z"]
        result = get_sector_map(syms)
        assert set(result.keys()) == set(syms)

    def test_sector_names_valid(self):
        result = get_sector_map(["A"])
        assert result["A"] in {
            "Tech", "Health", "Finance", "Consumer", "Energy",
            "Industrial", "Materials", "Utilities", "RealEstate", "Telecom",
        }
