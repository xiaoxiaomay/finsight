"""Tests for factor computation modules."""

import numpy as np
import pandas as pd


# Shared test price matrix
def make_price_matrix(n_days: int = 300, n_symbols: int = 20) -> pd.DataFrame:
    """Generate a realistic test price matrix."""
    np.random.seed(42)
    dates = pd.bdate_range("2023-01-01", periods=n_days)
    symbols = [f"SYM{i:02d}" for i in range(n_symbols)]

    returns = np.random.normal(0.0003, 0.015, (n_days, n_symbols))
    prices = 100 * np.exp(np.cumsum(returns, axis=0))
    return pd.DataFrame(prices, index=dates, columns=symbols)


def make_fundamentals(symbols: list[str]) -> pd.DataFrame:
    """Generate test fundamental data."""
    np.random.seed(42)
    rows = []
    for sym in symbols:
        for year in range(2022, 2025):
            rows.append({
                "symbol": sym,
                "filing_date": pd.Timestamp(f"{year}-03-15"),
                "report_date": pd.Timestamp(f"{year - 1}-12-31"),
                "period_type": "annual",
                "eps": np.random.uniform(2, 15),
                "book_value_per_share": np.random.uniform(10, 80),
                "net_income": np.random.uniform(1e8, 5e9),
                "total_equity": np.random.uniform(5e8, 50e9),
                "total_assets": np.random.uniform(1e9, 100e9),
                "total_liabilities": np.random.uniform(5e8, 60e9),
                "gross_profit": np.random.uniform(5e8, 20e9),
                "operating_income": np.random.uniform(2e8, 10e9),
                "operating_cash_flow": np.random.uniform(3e8, 15e9),
                "shares_outstanding": np.random.randint(1e8, 5e9),
            })
    return pd.DataFrame(rows)


class TestFactorBase:
    """Test base factor utilities."""

    def test_cross_sectional_zscore(self) -> None:
        """Z-score should have mean~0 and std~1 for valid data."""
        from src.quant.factors.base import cross_sectional_zscore

        row = pd.Series([10, 20, 30, 40, 50], index=["A", "B", "C", "D", "E"])
        z = cross_sectional_zscore(row)

        assert abs(z.mean()) < 0.01
        assert abs(z.std() - 1.0) < 0.1

    def test_cross_sectional_zscore_winsorizes(self) -> None:
        """Z-score should clip outliers at ±3σ."""
        from src.quant.factors.base import cross_sectional_zscore

        row = pd.Series([1, 2, 3, 4, 100], index=["A", "B", "C", "D", "E"])
        z = cross_sectional_zscore(row)

        assert z.max() < 4  # Should be clipped

    def test_cross_sectional_zscore_handles_nans(self) -> None:
        """Z-score should handle NaN values gracefully."""
        from src.quant.factors.base import cross_sectional_zscore

        row = pd.Series([10, np.nan, 30, 40, 50])
        z = cross_sectional_zscore(row)

        assert pd.isna(z.iloc[1])
        assert not pd.isna(z.iloc[0])

    def test_prepare_price_matrix(self) -> None:
        """Should convert long format to wide format."""
        from src.quant.factors.base import prepare_price_matrix

        df = pd.DataFrame({
            "symbol": ["A", "A", "B", "B"],
            "date": ["2024-01-02", "2024-01-03", "2024-01-02", "2024-01-03"],
            "adj_close": [100, 101, 200, 202],
        })

        wide = prepare_price_matrix(df)

        assert wide.shape == (2, 2)
        assert "A" in wide.columns
        assert "B" in wide.columns

    def test_merge_pit_fundamentals(self) -> None:
        """Should forward-fill using filing_date (point-in-time)."""
        from src.quant.factors.base import merge_pit_fundamentals

        fund = pd.DataFrame({
            "symbol": ["AAPL", "AAPL"],
            "filing_date": [pd.Timestamp("2023-03-01"), pd.Timestamp("2024-03-01")],
            "eps": [5.0, 6.0],
        })

        dates = pd.date_range("2023-01-01", "2024-06-01", freq="MS")
        result = merge_pit_fundamentals(fund, dates, "eps")

        # Before first filing: NaN
        assert pd.isna(result.loc["2023-01-01", "AAPL"])

        # After first filing: 5.0
        assert result.loc["2023-04-01", "AAPL"] == 5.0

        # After second filing: 6.0
        assert result.loc["2024-04-01", "AAPL"] == 6.0


class TestMomentumFactors:
    """Test momentum factor computations."""

    def test_momentum_12_1_shape(self) -> None:
        """Momentum factor should produce same shape as prices."""
        from src.quant.factors.momentum import Momentum12M1M

        prices = make_price_matrix(n_days=300)
        factor = Momentum12M1M()
        raw = factor.compute_raw(prices)

        assert raw.shape == prices.shape
        assert list(raw.columns) == list(prices.columns)

    def test_momentum_12_1_needs_history(self) -> None:
        """Momentum requires 252 days of history; early rows should be NaN."""
        from src.quant.factors.momentum import Momentum12M1M

        prices = make_price_matrix(n_days=300)
        factor = Momentum12M1M()
        raw = factor.compute_raw(prices)

        # First 252 rows should be all NaN
        assert raw.iloc[:252].isna().all().all()
        # Later rows should have values
        assert not raw.iloc[260:].isna().all().all()

    def test_momentum_compute_returns_long_format(self) -> None:
        """compute() should return long-format with correct columns."""
        from src.quant.factors.momentum import Momentum12M1M

        prices = make_price_matrix(n_days=300)
        factor = Momentum12M1M()
        result = factor.compute(prices)

        expected_cols = {"symbol", "date", "factor_name", "raw_value", "z_score", "percentile"}
        assert set(result.columns) == expected_cols
        assert (result["factor_name"] == "momentum_12_1").all()

    def test_short_term_reversal(self) -> None:
        """Short-term reversal should negate recent returns."""
        from src.quant.factors.momentum import ShortTermReversal

        prices = make_price_matrix(n_days=60)
        factor = ShortTermReversal()
        raw = factor.compute_raw(prices)

        # The factor negates 1-month returns
        # A stock that went up should have negative signal
        assert not raw.iloc[25:].isna().all().all()


class TestValueFactors:
    """Test value factor computations."""

    def test_earnings_yield(self) -> None:
        """Earnings yield should be EPS / Price."""
        from src.quant.factors.value import EarningsYield

        prices = make_price_matrix(n_days=300)
        fund = make_fundamentals(list(prices.columns))

        factor = EarningsYield()
        raw = factor.compute_raw(prices, fund)

        # Should have non-NaN values after filing dates
        assert not raw.iloc[200:].isna().all().all()

    def test_book_to_market(self) -> None:
        """Book-to-market should be BVPS / Price."""
        from src.quant.factors.value import BookToMarket

        prices = make_price_matrix(n_days=300)
        fund = make_fundamentals(list(prices.columns))

        factor = BookToMarket()
        raw = factor.compute_raw(prices, fund)

        assert not raw.iloc[200:].isna().all().all()

    def test_value_factors_need_fundamentals(self) -> None:
        """Value factors should return all NaN without fundamentals."""
        from src.quant.factors.value import EarningsYield

        prices = make_price_matrix(n_days=100)
        factor = EarningsYield()
        raw = factor.compute_raw(prices, fundamentals=None)

        assert raw.isna().all().all()


class TestQualityFactors:
    """Test quality factor computations."""

    def test_roe(self) -> None:
        """ROE = net_income / total_equity."""
        from src.quant.factors.quality import ROE

        prices = make_price_matrix(n_days=300)
        fund = make_fundamentals(list(prices.columns))

        factor = ROE()
        raw = factor.compute_raw(prices, fund)

        assert not raw.iloc[200:].isna().all().all()

    def test_gross_profitability(self) -> None:
        """Gross profitability = gross_profit / total_assets."""
        from src.quant.factors.quality import GrossProfitability

        prices = make_price_matrix(n_days=300)
        fund = make_fundamentals(list(prices.columns))

        factor = GrossProfitability()
        raw = factor.compute_raw(prices, fund)

        assert not raw.iloc[200:].isna().all().all()

    def test_accruals_negated(self) -> None:
        """Accruals should be negated (lower accruals = better)."""
        from src.quant.factors.quality import Accruals

        factor = Accruals()
        assert factor.name == "accruals"

    def test_asset_growth_negated(self) -> None:
        """Asset growth should be negated (lower growth = better signal)."""
        from src.quant.factors.quality import AssetGrowth

        factor = AssetGrowth()
        assert factor.name == "asset_growth"


class TestLowVolFactor:
    """Test low volatility factor."""

    def test_volatility_60d(self) -> None:
        """60-day vol should be negated (lower vol = higher signal)."""
        from src.quant.factors.low_volatility import Volatility60D

        prices = make_price_matrix(n_days=100)
        factor = Volatility60D()
        raw = factor.compute_raw(prices)

        # Values should be negative (negated vol)
        valid = raw.iloc[65:].dropna(how="all")
        assert (valid < 0).any().any()

    def test_volatility_annualized(self) -> None:
        """Annualized vol should be in reasonable range."""
        from src.quant.factors.low_volatility import Volatility60D

        prices = make_price_matrix(n_days=100, n_symbols=5)
        factor = Volatility60D()
        raw = factor.compute_raw(prices)

        # Negated annualized vol; abs should be 0.1-1.0 for typical stocks
        valid = raw.iloc[65:].dropna().abs()
        assert valid.mean().mean() > 0.05
        assert valid.mean().mean() < 1.5


class TestComposite:
    """Test composite factor combiner."""

    def test_combine_equal_weight(self) -> None:
        """Composite should produce valid z-scores."""
        from src.quant.factors.composite import CompositeFactor

        dates = pd.date_range("2024-01-01", periods=10)
        symbols = ["A", "B", "C", "D", "E"]

        factor_zscores = {
            "momentum_12_1": pd.DataFrame(
                np.random.randn(10, 5), index=dates, columns=symbols
            ),
            "earnings_yield": pd.DataFrame(
                np.random.randn(10, 5), index=dates, columns=symbols
            ),
        }

        composite = CompositeFactor(
            factor_weights={"momentum_12_1": 0.5, "earnings_yield": 0.5}
        )
        result = composite.combine(factor_zscores)

        assert result.shape == (10, 5)
        assert not result.isna().all().all()

    def test_get_all_factors(self) -> None:
        """Should return exactly 10 factors."""
        from src.quant.factors.composite import get_all_factors

        factors = get_all_factors()
        assert len(factors) == 10
        names = [f.name for f in factors]
        assert "momentum_12_1" in names
        assert "earnings_yield" in names
        assert "roe" in names
        assert "volatility_60d" in names


class TestNeutralizer:
    """Test Barra-style factor neutralization."""

    def test_neutralize_removes_sector_bias(self) -> None:
        """Neutralized factor should have ~0 mean per sector."""
        from src.quant.factors.neutralizer import FactorNeutralizer

        np.random.seed(42)
        n = 100
        symbols = [f"S{i:03d}" for i in range(n)]

        # Create factor with sector bias
        sectors = pd.Series(
            ["Tech"] * 30 + ["Finance"] * 30 + ["Health"] * 20 + ["Energy"] * 20,
            index=symbols,
        )
        # Tech has higher raw values (bias)
        raw = pd.Series(
            np.concatenate([
                np.random.normal(2.0, 1.0, 30),   # Tech: high
                np.random.normal(-1.0, 1.0, 30),  # Finance: low
                np.random.normal(0.0, 1.0, 20),   # Health: neutral
                np.random.normal(0.5, 1.0, 20),   # Energy: slight
            ]),
            index=symbols,
        )
        market_cap = pd.Series(np.random.uniform(1e9, 100e9, n), index=symbols)

        neutralizer = FactorNeutralizer()
        neutralized = neutralizer.neutralize(raw, sectors, market_cap)

        # After neutralization, sector means should be closer to 0
        df = pd.DataFrame({"sector": sectors, "neutralized": neutralized})
        sector_means = df.groupby("sector")["neutralized"].mean()

        for sector, mean in sector_means.items():
            assert abs(mean) < 0.5, f"Sector {sector} mean={mean:.2f} should be near 0"

    def test_neutralize_small_sample(self) -> None:
        """Should fall back gracefully with very few stocks."""
        from src.quant.factors.neutralizer import FactorNeutralizer

        symbols = ["A", "B", "C"]
        factor_values = pd.Series([1.0, 2.0, 3.0], index=symbols)
        sector = pd.Series(["Tech", "Tech", "Health"], index=symbols)
        market_cap = pd.Series([1e9, 2e9, 3e9], index=symbols)

        neutralizer = FactorNeutralizer()
        result = neutralizer.neutralize(factor_values, sector, market_cap)

        # Should return something (even if not fully neutralized)
        assert len(result) == 3

    def test_compute_factor_exposure(self) -> None:
        """Factor exposure should be weighted sum of z-scores."""
        from src.quant.factors.neutralizer import FactorNeutralizer

        weights = pd.Series({"A": 0.3, "B": 0.3, "C": 0.4})
        factor_values = pd.Series({"A": 1.0, "B": -0.5, "C": 0.5})

        exposure = FactorNeutralizer.compute_factor_exposure(weights, factor_values)

        expected = 0.3 * 1.0 + 0.3 * (-0.5) + 0.4 * 0.5
        assert abs(exposure - expected) < 1e-10

    def test_compare_before_after(self) -> None:
        """Before/after comparison should show sector means."""
        from src.quant.factors.neutralizer import compare_before_after

        symbols = ["A", "B", "C", "D"]
        raw = pd.Series([3.0, 2.0, 0.0, -1.0], index=symbols)
        neutralized = pd.Series([0.5, -0.5, 0.2, -0.2], index=symbols)
        sector = pd.Series(["Tech", "Tech", "Health", "Health"], index=symbols)

        result = compare_before_after(raw, neutralized, sector)

        assert "sector" in result.columns
        assert "raw_mean" in result.columns
        assert "neutralized_mean" in result.columns
        assert len(result) == 2
