"""Tests for Live Performance page data generation."""

import numpy as np
import pandas as pd

from src.dashboard._pages.live_performance import _calc_metrics, _generate_live_data


class TestGenerateLiveData:
    def test_returns_dict(self):
        data = _generate_live_data.__wrapped__()
        assert isinstance(data, dict)

    def test_has_all_keys(self):
        data = _generate_live_data.__wrapped__()
        required = {
            "is_returns", "is_bench_returns", "oos_returns", "oos_bench_returns",
            "full_returns", "full_bench_returns", "full_port_value", "full_bench_value",
            "factors", "factor_attribution", "live_start",
        }
        assert required.issubset(set(data.keys()))

    def test_is_period_length(self):
        data = _generate_live_data.__wrapped__()
        assert len(data["is_returns"]) == 1008

    def test_oos_period_length(self):
        data = _generate_live_data.__wrapped__()
        assert len(data["oos_returns"]) == 504

    def test_full_combines_is_oos(self):
        data = _generate_live_data.__wrapped__()
        assert len(data["full_returns"]) == 1008 + 504

    def test_portfolio_value_increasing_trend(self):
        data = _generate_live_data.__wrapped__()
        # Overall should be higher at end than start (positive drift)
        assert data["full_port_value"].iloc[-1] > data["full_port_value"].iloc[0]

    def test_ten_factors(self):
        data = _generate_live_data.__wrapped__()
        assert len(data["factors"]) == 10
        assert len(data["factor_attribution"]) == 10


class TestCalcMetrics:
    def setup_method(self):
        rng = np.random.default_rng(42)
        n = 252
        self.returns = pd.Series(rng.normal(0.0004, 0.01, n))
        self.bench = pd.Series(rng.normal(0.0003, 0.01, n))

    def test_returns_dict(self):
        m = _calc_metrics(self.returns, self.bench)
        assert isinstance(m, dict)

    def test_has_all_metrics(self):
        m = _calc_metrics(self.returns, self.bench)
        keys = {"ann_return", "ann_volatility", "sharpe_ratio", "sortino_ratio",
                "max_drawdown", "calmar_ratio", "information_ratio", "alpha",
                "total_return", "win_rate"}
        assert keys == set(m.keys())

    def test_sharpe_reasonable(self):
        m = _calc_metrics(self.returns, self.bench)
        assert -5 < m["sharpe_ratio"] < 5

    def test_max_drawdown_negative(self):
        m = _calc_metrics(self.returns, self.bench)
        assert m["max_drawdown"] <= 0

    def test_win_rate_between_0_and_1(self):
        m = _calc_metrics(self.returns, self.bench)
        assert 0 <= m["win_rate"] <= 1
