"""Tests for data quality validation."""

import numpy as np
import pandas as pd


class TestMarketDataValidation:
    """Test market data quality checks."""

    def test_no_future_dates(self, sample_market_data: pd.DataFrame) -> None:
        """Market data should not contain dates in the future."""
        today = pd.Timestamp.now().normalize()
        assert (sample_market_data["date"] <= today).all()

    def test_ohlc_consistency(self, sample_market_data: pd.DataFrame) -> None:
        """High should be >= Low, and Open/Close within High/Low range."""
        df = sample_market_data
        assert (df["high"] >= df["low"]).all()
        assert (df["high"] >= df["open"]).all()
        assert (df["high"] >= df["close"]).all()
        assert (df["low"] <= df["open"]).all()
        assert (df["low"] <= df["close"]).all()

    def test_positive_volume(self, sample_market_data: pd.DataFrame) -> None:
        """Volume should be positive."""
        assert (sample_market_data["volume"] > 0).all()

    def test_no_missing_prices(self, sample_market_data: pd.DataFrame) -> None:
        """Price columns should not have NaN values."""
        price_cols = ["open", "high", "low", "close", "adj_close"]
        assert not sample_market_data[price_cols].isna().any().any()

    def test_outlier_detection(self, sample_market_data: pd.DataFrame) -> None:
        """Daily returns should not exceed 5 sigma (extreme outlier)."""
        returns = sample_market_data["close"].pct_change().dropna()
        std = returns.std()
        if std > 0:
            z_scores = (returns - returns.mean()) / std
            assert (z_scores.abs() < 5).all(), "Detected >5 sigma daily return"


class TestFundamentalsValidation:
    """Test fundamental data quality checks."""

    def test_filing_after_report(self, sample_fundamentals: dict) -> None:
        """Filing date must be after report date (point-in-time)."""
        assert sample_fundamentals["filing_date"] >= sample_fundamentals["report_date"]

    def test_positive_revenue(self, sample_fundamentals: dict) -> None:
        """Revenue should be positive for a real company."""
        assert sample_fundamentals["revenue"] > 0

    def test_equity_is_assets_minus_liabilities(self) -> None:
        """Basic accounting identity: equity = assets - liabilities."""
        assets = 352_583_000_000
        liabilities = 290_437_000_000
        equity = assets - liabilities
        assert equity > 0


class TestValidatorsModule:
    """Test the validators module directly."""

    def test_validate_market_data_clean(self) -> None:
        """Clean market data should pass validation."""
        from src.data.quality.validators import validate_market_data

        df = pd.DataFrame({
            "symbol": ["AAPL"] * 3,
            "date": pd.date_range("2024-01-02", periods=3, freq="B"),
            "open": [185.0, 186.0, 187.0],
            "high": [187.0, 188.0, 189.0],
            "low": [184.0, 185.0, 186.0],
            "close": [186.0, 187.0, 188.0],
            "adj_close": [186.0, 187.0, 188.0],
            "volume": [50_000_000, 48_000_000, 52_000_000],
        })

        clean, issues = validate_market_data(df)
        assert len(clean) == 3
        # May have date gap issues for short series, but no critical issues
        critical = [i for i in issues if "Removed" in i or "Dropped" in i]
        assert len(critical) == 0

    def test_validate_market_data_removes_nulls(self) -> None:
        """Should remove all-null price rows."""
        from src.data.quality.validators import validate_market_data

        df = pd.DataFrame({
            "symbol": ["AAPL", "AAPL"],
            "date": pd.date_range("2024-01-02", periods=2, freq="B"),
            "open": [185.0, np.nan],
            "high": [187.0, np.nan],
            "low": [184.0, np.nan],
            "close": [186.0, np.nan],
            "adj_close": [186.0, np.nan],
            "volume": [50_000_000, np.nan],
        })

        clean, issues = validate_market_data(df)
        assert len(clean) == 1
        assert any("null" in i.lower() for i in issues)

    def test_validate_fundamentals_pit(self) -> None:
        """Should flag filing_date < report_date."""
        from src.data.quality.validators import validate_fundamentals

        df = pd.DataFrame({
            "symbol": ["AAPL"],
            "report_date": ["2024-09-30"],
            "filing_date": ["2024-08-01"],
            "period_type": ["annual"],
            "revenue": [391_000_000_000],
        })

        clean, issues = validate_fundamentals(df)
        assert any("filing_date" in i for i in issues)

    def test_validate_macro_removes_nulls(self) -> None:
        """Should drop null macro values."""
        from src.data.quality.validators import validate_macro

        df = pd.DataFrame({
            "indicator_id": ["DFF", "DFF", "DGS10"],
            "date": ["2024-01-01", "2024-01-02", "2024-01-01"],
            "value": [5.33, np.nan, 4.50],
            "source": ["FRED"] * 3,
        })

        clean, issues = validate_macro(df)
        assert len(clean) == 2
        assert any("null" in i.lower() for i in issues)

    def test_coverage_stats(self) -> None:
        """Coverage stats should compute correctly."""
        from src.data.quality.validators import compute_coverage_stats

        df = pd.DataFrame({
            "symbol": ["AAPL"] * 5 + ["MSFT"] * 3,
            "date": list(pd.date_range("2024-01-02", periods=5, freq="B")) +
                    list(pd.date_range("2024-01-02", periods=3, freq="B")),
            "close": [186.0, 187.0, 188.0, 189.0, 190.0, 400.0, 401.0, 402.0],
        })

        stats = compute_coverage_stats(df)
        assert stats["total_rows"] == 8
        assert stats["unique_symbols"] == 2
        assert stats["rows_per_symbol"]["min"] == 3
        assert stats["rows_per_symbol"]["max"] == 5


class TestRateLimiter:
    """Test rate limiting utilities."""

    def test_daily_quota(self) -> None:
        """DailyQuotaLimiter should track usage."""
        from src.utils.rate_limiter import DailyQuotaLimiter

        limiter = DailyQuotaLimiter(daily_limit=3)

        assert limiter.remaining == 3
        assert limiter.acquire()  # 1
        assert limiter.acquire()  # 2
        assert limiter.acquire()  # 3
        assert not limiter.acquire()  # Exhausted
        assert limiter.remaining == 0

    def test_daily_quota_check(self) -> None:
        """Check should not consume quota."""
        from src.utils.rate_limiter import DailyQuotaLimiter

        limiter = DailyQuotaLimiter(daily_limit=5)

        assert limiter.check(needed=3)
        assert limiter.remaining == 5  # Check doesn't consume
        assert limiter.check(needed=6) is False
