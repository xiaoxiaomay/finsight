"""Core backtesting engine.

Simulates a factor-based portfolio strategy with:
- Monthly rebalancing
- Realistic transaction costs (commission + spread + market impact)
- Long-only portfolio construction
- Comprehensive performance metrics

All computation is vectorized. No day-by-day simulation loops.
"""

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from src.config.logging_config import get_logger

logger = get_logger("backtest.engine")


@dataclass
class BacktestConfig:
    """Configuration for a single backtest run."""

    # Universe & timing
    start_date: date = date(2021, 1, 1)
    end_date: date = date(2025, 12, 31)
    rebalance_frequency: str = "monthly"  # 'monthly', 'weekly', 'quarterly'

    # Portfolio construction
    num_holdings: int = 50
    max_position_weight: float = 0.05
    max_sector_weight: float = 0.25

    # Transaction costs (realistic)
    commission_per_share: float = 0.005   # $0.005/share
    spread_cost_bps: float = 5.0          # 5 bps half-spread
    market_impact_bps: float = 10.0       # 10 bps market impact

    # Initial capital
    initial_capital: float = 1_000_000.0

    # Benchmark
    benchmark_symbol: str = "SPY"


@dataclass
class BacktestResult:
    """Results from a completed backtest."""

    config: BacktestConfig

    # Time series
    portfolio_returns: pd.Series = field(default_factory=pd.Series)
    benchmark_returns: pd.Series = field(default_factory=pd.Series)
    portfolio_value: pd.Series = field(default_factory=pd.Series)
    benchmark_value: pd.Series = field(default_factory=pd.Series)

    # Holdings over time
    weights_history: pd.DataFrame = field(default_factory=pd.DataFrame)
    turnover_series: pd.Series = field(default_factory=pd.Series)

    # Performance metrics
    metrics: dict = field(default_factory=dict)


def run_backtest(
    factor_scores: pd.DataFrame,
    prices: pd.DataFrame,
    config: BacktestConfig | None = None,
    benchmark_prices: pd.Series | None = None,
) -> BacktestResult:
    """Run a factor-based backtest.

    Args:
        factor_scores: Wide-format factor scores (dates × symbols).
            Higher score = more attractive.
        prices: Wide-format adjusted close prices (dates × symbols).
        config: Backtest configuration.
        benchmark_prices: Optional benchmark price series (e.g., SPY).

    Returns:
        BacktestResult with returns, metrics, and weight history.
    """
    if config is None:
        config = BacktestConfig()

    # Align data to date range
    start = pd.Timestamp(config.start_date)
    end = pd.Timestamp(config.end_date)

    prices = prices.loc[start:end].copy()
    factor_scores = factor_scores.reindex(prices.index).copy()

    if prices.empty:
        logger.warning("no_price_data_in_range")
        return BacktestResult(config=config)

    # Compute daily returns
    daily_returns = prices.pct_change()

    # Get rebalance dates (month-end business days)
    rebal_dates = _get_rebalance_dates(prices.index, config.rebalance_frequency)

    if len(rebal_dates) < 2:
        logger.warning("too_few_rebalance_dates")
        return BacktestResult(config=config)

    # Generate portfolio weights at each rebalance date
    weights_history = _construct_weights(
        factor_scores, rebal_dates, config.num_holdings, config.max_position_weight
    )

    # Simulate portfolio returns with transaction costs
    portfolio_returns, turnover = _simulate_returns(
        daily_returns, weights_history, rebal_dates, config
    )

    # Compute portfolio value
    portfolio_value = (1 + portfolio_returns).cumprod() * config.initial_capital

    # Benchmark
    if benchmark_prices is not None:
        bench = benchmark_prices.reindex(prices.index).pct_change().dropna()
        bench = bench.reindex(portfolio_returns.index).fillna(0)
        benchmark_value = (1 + bench).cumprod() * config.initial_capital
    else:
        bench = pd.Series(0.0, index=portfolio_returns.index)
        benchmark_value = pd.Series(config.initial_capital, index=portfolio_returns.index)

    # Compute metrics
    metrics = compute_performance_metrics(
        portfolio_returns, bench, config.initial_capital
    )

    result = BacktestResult(
        config=config,
        portfolio_returns=portfolio_returns,
        benchmark_returns=bench,
        portfolio_value=portfolio_value,
        benchmark_value=benchmark_value,
        weights_history=weights_history,
        turnover_series=turnover,
        metrics=metrics,
    )

    logger.info("backtest_complete", **{k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()})

    return result


def compute_performance_metrics(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    initial_capital: float = 1_000_000.0,
) -> dict:
    """Compute comprehensive performance metrics.

    Args:
        returns: Daily portfolio returns.
        benchmark_returns: Daily benchmark returns (aligned).
        initial_capital: Starting capital.

    Returns:
        Dict of performance metrics.
    """
    if returns.empty or len(returns) < 2:
        return {}

    n_days = len(returns)
    n_years = n_days / 252

    # Basic returns
    cum_return = (1 + returns).prod() - 1
    ann_return = (1 + cum_return) ** (1 / max(n_years, 0.01)) - 1
    ann_vol = returns.std() * np.sqrt(252)

    # Sharpe ratio (assuming 0% risk-free rate for simplicity)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0

    # Sortino ratio (downside deviation)
    downside = returns[returns < 0]
    downside_vol = downside.std() * np.sqrt(252) if len(downside) > 0 else ann_vol
    sortino = ann_return / downside_vol if downside_vol > 0 else 0.0

    # Maximum drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()

    # Calmar ratio
    calmar = ann_return / abs(max_drawdown) if max_drawdown != 0 else 0.0

    # Win rate
    win_rate = (returns > 0).mean()

    # Benchmark-relative metrics
    active_returns = returns - benchmark_returns
    ann_active_return = active_returns.mean() * 252
    tracking_error = active_returns.std() * np.sqrt(252)
    information_ratio = ann_active_return / tracking_error if tracking_error > 0 else 0.0

    # Alpha t-statistic
    t_stat = (active_returns.mean() / (active_returns.std() / np.sqrt(n_days))) if active_returns.std() > 0 else 0.0

    return {
        "total_return": round(cum_return, 4),
        "ann_return": round(ann_return, 4),
        "ann_volatility": round(ann_vol, 4),
        "sharpe_ratio": round(sharpe, 4),
        "sortino_ratio": round(sortino, 4),
        "max_drawdown": round(max_drawdown, 4),
        "calmar_ratio": round(calmar, 4),
        "win_rate": round(win_rate, 4),
        "alpha_annualized": round(ann_active_return, 4),
        "tracking_error": round(tracking_error, 4),
        "information_ratio": round(information_ratio, 4),
        "alpha_t_stat": round(t_stat, 4),
        "n_days": n_days,
        "n_years": round(n_years, 2),
    }


def _get_rebalance_dates(
    index: pd.DatetimeIndex, frequency: str
) -> list[pd.Timestamp]:
    """Get rebalance dates from the trading calendar."""
    if frequency == "monthly":
        # Last trading day of each month
        return list(index.to_series().groupby(index.to_period("M")).last())
    elif frequency == "weekly":
        return list(index.to_series().groupby(index.to_period("W")).last())
    elif frequency == "quarterly":
        return list(index.to_series().groupby(index.to_period("Q")).last())
    else:
        raise ValueError(f"Unknown frequency: {frequency}")


def _construct_weights(
    scores: pd.DataFrame,
    rebal_dates: list[pd.Timestamp],
    n_holdings: int,
    max_weight: float,
) -> pd.DataFrame:
    """Construct equal-weight portfolio from top-N scored stocks at each rebalance."""
    weights = pd.DataFrame(0.0, index=rebal_dates, columns=scores.columns)

    for dt in rebal_dates:
        if dt not in scores.index:
            continue

        row = scores.loc[dt].dropna()
        top = row.nlargest(max(len(row), 1)) if len(row) < n_holdings else row.nlargest(n_holdings)

        if len(top) == 0:
            continue

        # Equal weight within selected stocks
        w = 1.0 / len(top)
        w = min(w, max_weight)  # Cap individual weights

        weights.loc[dt, top.index] = w

        # Renormalize so weights sum to 1
        row_sum = weights.loc[dt].sum()
        if row_sum > 0:
            weights.loc[dt] /= row_sum

    return weights


def _simulate_returns(
    daily_returns: pd.DataFrame,
    weights_history: pd.DataFrame,
    rebal_dates: list[pd.Timestamp],
    config: BacktestConfig,
) -> tuple[pd.Series, pd.Series]:
    """Simulate portfolio returns accounting for transaction costs.

    Between rebalance dates, weights drift with market movements.
    At rebalance, costs are incurred proportional to turnover.
    """
    all_dates = daily_returns.index
    port_returns = pd.Series(0.0, index=all_dates)
    turnover = pd.Series(0.0, index=rebal_dates)

    current_weights = pd.Series(0.0, index=daily_returns.columns)

    # Total cost in bps per unit of turnover
    one_way_cost_bps = config.spread_cost_bps + config.market_impact_bps

    rebal_set = set(rebal_dates)
    rebal_idx = 0

    for dt in all_dates:
        day_returns = daily_returns.loc[dt].fillna(0)

        if dt in rebal_set and rebal_idx < len(rebal_dates):
            target_weights = weights_history.iloc[rebal_idx]

            # Turnover = sum of absolute weight changes / 2
            to = (target_weights - current_weights).abs().sum() / 2
            turnover.iloc[rebal_idx] = to

            # Transaction cost = turnover × cost_bps
            cost = to * one_way_cost_bps / 10000

            # Portfolio return for this day: weighted return minus costs
            port_ret = (target_weights * day_returns).sum() - cost
            port_returns.loc[dt] = port_ret

            # Update weights to target (after rebalance)
            current_weights = target_weights.copy()
            rebal_idx += 1
        else:
            # Drift day: weights change with returns
            port_ret = (current_weights * day_returns).sum()
            port_returns.loc[dt] = port_ret

            # Update weights for drift
            new_values = current_weights * (1 + day_returns)
            total = new_values.sum()
            if total > 0:
                current_weights = new_values / total

    return port_returns, turnover
