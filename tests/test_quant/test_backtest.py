"""Tests for backtesting engine and statistics."""

import numpy as np
import pandas as pd


def make_backtest_data(
    n_days: int = 500,
    n_symbols: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate test data for backtesting."""
    np.random.seed(42)
    dates = pd.bdate_range("2022-01-01", periods=n_days)
    symbols = [f"SYM{i:02d}" for i in range(n_symbols)]

    # Prices
    returns = np.random.normal(0.0003, 0.015, (n_days, n_symbols))
    prices = pd.DataFrame(
        100 * np.exp(np.cumsum(returns, axis=0)),
        index=dates,
        columns=symbols,
    )

    # Factor scores (random but persistent — some stocks always higher)
    base_scores = np.random.randn(n_symbols)
    noise = np.random.randn(n_days, n_symbols) * 0.3
    scores = pd.DataFrame(
        base_scores + noise,
        index=dates,
        columns=symbols,
    )

    return prices, scores


class TestBacktestEngine:
    """Test the core backtest engine."""

    def test_run_backtest_basic(self) -> None:
        """Should produce valid results for a simple backtest."""
        from src.quant.backtest.engine import BacktestConfig, run_backtest

        prices, scores = make_backtest_data()

        config = BacktestConfig(
            start_date=prices.index[0].date(),
            end_date=prices.index[-1].date(),
            num_holdings=10,
        )

        result = run_backtest(scores, prices, config)

        assert not result.portfolio_returns.empty
        assert not result.portfolio_value.empty
        assert len(result.metrics) > 0
        assert "sharpe_ratio" in result.metrics
        assert "max_drawdown" in result.metrics

    def test_backtest_weights_sum_to_one(self) -> None:
        """Portfolio weights should sum to ~1.0 at each rebalance."""
        from src.quant.backtest.engine import BacktestConfig, run_backtest

        prices, scores = make_backtest_data()
        config = BacktestConfig(
            start_date=prices.index[0].date(),
            end_date=prices.index[-1].date(),
            num_holdings=10,
        )

        result = run_backtest(scores, prices, config)

        for dt in result.weights_history.index:
            row_sum = result.weights_history.loc[dt].sum()
            if row_sum > 0:
                assert abs(row_sum - 1.0) < 0.01, f"Weights sum={row_sum} at {dt}"

    def test_backtest_max_weight_constraint(self) -> None:
        """No single position should exceed max_position_weight."""
        from src.quant.backtest.engine import BacktestConfig, run_backtest

        prices, scores = make_backtest_data()
        max_wt = 0.05

        config = BacktestConfig(
            start_date=prices.index[0].date(),
            end_date=prices.index[-1].date(),
            num_holdings=30,
            max_position_weight=max_wt,
        )

        result = run_backtest(scores, prices, config)

        for dt in result.weights_history.index:
            max_pos = result.weights_history.loc[dt].max()
            assert max_pos <= max_wt + 0.01, f"Max weight={max_pos} at {dt}"

    def test_backtest_with_benchmark(self) -> None:
        """Should compute benchmark-relative metrics."""
        from src.quant.backtest.engine import BacktestConfig, run_backtest

        prices, scores = make_backtest_data()
        bench = prices.mean(axis=1)  # Equal-weight benchmark

        config = BacktestConfig(
            start_date=prices.index[0].date(),
            end_date=prices.index[-1].date(),
        )

        result = run_backtest(scores, prices, config, benchmark_prices=bench)

        assert "information_ratio" in result.metrics
        assert "alpha_t_stat" in result.metrics
        assert not result.benchmark_returns.empty


class TestPerformanceMetrics:
    """Test performance metric computations."""

    def test_compute_metrics_basic(self) -> None:
        """Should compute all expected metrics."""
        from src.quant.backtest.engine import compute_performance_metrics

        np.random.seed(42)
        n = 500
        returns = pd.Series(np.random.normal(0.0004, 0.01, n))
        bench = pd.Series(np.random.normal(0.0003, 0.012, n))

        metrics = compute_performance_metrics(returns, bench)

        expected_keys = {
            "total_return", "ann_return", "ann_volatility",
            "sharpe_ratio", "sortino_ratio", "max_drawdown",
            "calmar_ratio", "win_rate", "alpha_annualized",
            "tracking_error", "information_ratio", "alpha_t_stat",
            "n_days", "n_years",
        }
        assert expected_keys.issubset(set(metrics.keys()))

    def test_sharpe_ratio_positive_for_positive_returns(self) -> None:
        """Sharpe should be positive when avg return is positive."""
        from src.quant.backtest.engine import compute_performance_metrics

        returns = pd.Series(np.random.normal(0.001, 0.01, 252))
        bench = pd.Series(0.0, index=returns.index)

        metrics = compute_performance_metrics(returns, bench)
        assert metrics["sharpe_ratio"] > 0

    def test_max_drawdown_is_negative(self) -> None:
        """Max drawdown should be negative (or zero)."""
        from src.quant.backtest.engine import compute_performance_metrics

        returns = pd.Series(np.random.normal(0, 0.02, 252))
        bench = pd.Series(0.0, index=returns.index)

        metrics = compute_performance_metrics(returns, bench)
        assert metrics["max_drawdown"] <= 0


class TestStatistics:
    """Test statistical significance functions."""

    def test_alpha_significance(self) -> None:
        """Should compute t-stat and p-value."""
        from src.quant.backtest.statistics import alpha_significance

        np.random.seed(42)
        port = pd.Series(np.random.normal(0.001, 0.01, 500))
        bench = pd.Series(np.random.normal(0.0005, 0.012, 500))

        result = alpha_significance(port, bench)

        assert "t_stat" in result
        assert "p_value" in result
        assert "significant" in result
        assert 0 <= result["p_value"] <= 1

    def test_drawdown_analysis(self) -> None:
        """Should compute drawdown metrics."""
        from src.quant.backtest.statistics import drawdown_analysis

        returns = pd.Series(
            [0.01, 0.02, -0.05, -0.03, 0.04, 0.01],
            index=pd.bdate_range("2024-01-02", periods=6),
        )

        result = drawdown_analysis(returns)

        assert "max_drawdown" in result
        assert "drawdown_series" in result
        assert result["max_drawdown"] < 0

    def test_monthly_returns_table(self) -> None:
        """Should produce year × month table."""
        from src.quant.backtest.statistics import monthly_returns_table

        dates = pd.bdate_range("2023-01-01", periods=500)
        returns = pd.Series(np.random.normal(0.0003, 0.01, 500), index=dates)

        table = monthly_returns_table(returns)

        assert "Jan" in table.columns
        assert "Dec" in table.columns
        assert "Year" in table.columns
        assert len(table) >= 1

    def test_rolling_metrics(self) -> None:
        """Should compute rolling Sharpe, vol, return."""
        from src.quant.backtest.statistics import rolling_metrics

        returns = pd.Series(
            np.random.normal(0.0005, 0.01, 200),
            index=pd.bdate_range("2024-01-01", periods=200),
        )

        result = rolling_metrics(returns, window=63)

        assert "rolling_sharpe" in result.columns
        assert "rolling_volatility" in result.columns
        assert len(result) == 200


class TestWalkForward:
    """Test walk-forward validation."""

    def test_generate_folds(self) -> None:
        """Should generate correct number of folds."""
        from src.quant.backtest.walk_forward import WalkForwardConfig, _generate_folds

        index = pd.bdate_range("2018-01-01", periods=1500)
        config = WalkForwardConfig(
            n_folds=3,
            in_sample_months=36,
            out_of_sample_months=12,
            purge_gap_days=5,
            step_months=12,
        )

        folds = _generate_folds(index, config)

        assert len(folds) >= 1
        for _is_start, is_end, oos_start, oos_end in folds:
            assert is_end < oos_start  # Purge gap
            assert oos_end > oos_start

    def test_walk_forward_run(self) -> None:
        """Should run walk-forward and return valid results."""
        from src.quant.backtest.engine import BacktestConfig
        from src.quant.backtest.walk_forward import WalkForwardConfig, run_walk_forward

        prices, scores = make_backtest_data(n_days=1000)

        wf_config = WalkForwardConfig(
            n_folds=2,
            in_sample_months=12,
            out_of_sample_months=6,
            purge_gap_days=5,
            step_months=6,
        )
        bt_config = BacktestConfig(num_holdings=10)

        result = run_walk_forward(scores, prices, wf_config, bt_config)

        assert len(result.folds) >= 1
        assert len(result.per_fold_metrics) >= 1
        assert "oos_sharpe" in result.per_fold_metrics[0]


class TestTransactionCosts:
    """Test transaction cost modeling."""

    def test_estimate_cost(self) -> None:
        """Cost should scale linearly with turnover."""
        from src.quant.backtest.transaction_costs import estimate_transaction_cost

        cost_10 = estimate_transaction_cost(0.10, spread_bps=5, market_impact_bps=10)
        cost_20 = estimate_transaction_cost(0.20, spread_bps=5, market_impact_bps=10)

        assert abs(cost_20 - 2 * cost_10) < 1e-10

    def test_compute_turnover(self) -> None:
        """Turnover should be half the sum of absolute weight changes."""
        from src.quant.backtest.transaction_costs import compute_turnover

        old = pd.Series({"A": 0.5, "B": 0.5})
        new = pd.Series({"A": 0.3, "B": 0.3, "C": 0.4})

        to = compute_turnover(old, new)

        # Changes: A: -0.2, B: -0.2, C: +0.4 → sum=0.8 → turnover=0.4
        assert abs(to - 0.4) < 1e-10
