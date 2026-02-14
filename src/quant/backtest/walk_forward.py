"""Walk-forward validation framework.

Splits the backtest period into rolling in-sample / out-of-sample windows.
This prevents overfitting — the model is always tested on unseen data.

Standard configuration:
- 5 folds
- 36 months in-sample
- 12 months out-of-sample
- 5-day purge gap between IS and OOS

The IS period is used to estimate factor weights or model parameters.
The OOS period tests the strategy on truly unseen data.
"""

from dataclasses import dataclass
from datetime import timedelta

import pandas as pd

from src.config.logging_config import get_logger
from src.quant.backtest.engine import (
    BacktestConfig,
    BacktestResult,
    compute_performance_metrics,
    run_backtest,
)

logger = get_logger("backtest.walk_forward")


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward validation."""

    n_folds: int = 5
    in_sample_months: int = 36      # 3 years IS
    out_of_sample_months: int = 12  # 1 year OOS
    purge_gap_days: int = 5         # Gap between IS and OOS to prevent leakage
    step_months: int = 12           # Roll forward by this many months each fold


@dataclass
class WalkForwardFold:
    """Results for a single walk-forward fold."""

    fold_idx: int
    is_start: pd.Timestamp
    is_end: pd.Timestamp
    oos_start: pd.Timestamp
    oos_end: pd.Timestamp
    is_result: BacktestResult
    oos_result: BacktestResult


@dataclass
class WalkForwardResult:
    """Results from walk-forward validation."""

    folds: list[WalkForwardFold]
    oos_combined_returns: pd.Series
    oos_combined_metrics: dict
    per_fold_metrics: list[dict]


def run_walk_forward(
    factor_scores: pd.DataFrame,
    prices: pd.DataFrame,
    wf_config: WalkForwardConfig | None = None,
    bt_config: BacktestConfig | None = None,
    benchmark_prices: pd.Series | None = None,
) -> WalkForwardResult:
    """Run walk-forward validation.

    Args:
        factor_scores: Wide-format factor scores (dates × symbols).
        prices: Wide-format adjusted close prices.
        wf_config: Walk-forward configuration.
        bt_config: Base backtest configuration (applied to each fold).
        benchmark_prices: Optional benchmark price series.

    Returns:
        WalkForwardResult with per-fold and combined metrics.
    """
    if wf_config is None:
        wf_config = WalkForwardConfig()
    if bt_config is None:
        bt_config = BacktestConfig()

    # Generate fold date ranges
    fold_ranges = _generate_folds(
        prices.index, wf_config
    )

    folds: list[WalkForwardFold] = []
    oos_returns_list: list[pd.Series] = []
    per_fold_metrics: list[dict] = []

    for i, (is_start, is_end, oos_start, oos_end) in enumerate(fold_ranges):
        logger.info(
            "running_fold",
            fold=i + 1,
            is_range=f"{is_start.date()} to {is_end.date()}",
            oos_range=f"{oos_start.date()} to {oos_end.date()}",
        )

        # In-sample backtest
        is_config = BacktestConfig(
            start_date=is_start.date(),
            end_date=is_end.date(),
            num_holdings=bt_config.num_holdings,
            max_position_weight=bt_config.max_position_weight,
            commission_per_share=bt_config.commission_per_share,
            spread_cost_bps=bt_config.spread_cost_bps,
            market_impact_bps=bt_config.market_impact_bps,
            initial_capital=bt_config.initial_capital,
        )

        is_result = run_backtest(factor_scores, prices, is_config, benchmark_prices)

        # Out-of-sample backtest (uses same factor scores, no re-fitting)
        oos_config = BacktestConfig(
            start_date=oos_start.date(),
            end_date=oos_end.date(),
            num_holdings=bt_config.num_holdings,
            max_position_weight=bt_config.max_position_weight,
            commission_per_share=bt_config.commission_per_share,
            spread_cost_bps=bt_config.spread_cost_bps,
            market_impact_bps=bt_config.market_impact_bps,
            initial_capital=bt_config.initial_capital,
        )

        oos_result = run_backtest(factor_scores, prices, oos_config, benchmark_prices)

        fold = WalkForwardFold(
            fold_idx=i,
            is_start=is_start,
            is_end=is_end,
            oos_start=oos_start,
            oos_end=oos_end,
            is_result=is_result,
            oos_result=oos_result,
        )

        folds.append(fold)
        per_fold_metrics.append({
            "fold": i + 1,
            "is_sharpe": is_result.metrics.get("sharpe_ratio", 0),
            "oos_sharpe": oos_result.metrics.get("sharpe_ratio", 0),
            "is_return": is_result.metrics.get("ann_return", 0),
            "oos_return": oos_result.metrics.get("ann_return", 0),
            "is_max_dd": is_result.metrics.get("max_drawdown", 0),
            "oos_max_dd": oos_result.metrics.get("max_drawdown", 0),
        })

        if not oos_result.portfolio_returns.empty:
            oos_returns_list.append(oos_result.portfolio_returns)

    # Combine OOS returns
    if oos_returns_list:
        oos_combined = pd.concat(oos_returns_list).sort_index()
        # Remove duplicates (overlapping folds)
        oos_combined = oos_combined[~oos_combined.index.duplicated(keep="first")]

        bench_combined = benchmark_prices.reindex(prices.index).pct_change().reindex(
            oos_combined.index
        ).fillna(0) if benchmark_prices is not None else pd.Series(0.0, index=oos_combined.index)

        oos_metrics = compute_performance_metrics(
            oos_combined, bench_combined, bt_config.initial_capital
        )
    else:
        oos_combined = pd.Series(dtype=float)
        oos_metrics = {}

    logger.info(
        "walk_forward_complete",
        n_folds=len(folds),
        oos_sharpe=oos_metrics.get("sharpe_ratio", "N/A"),
        oos_return=oos_metrics.get("ann_return", "N/A"),
    )

    return WalkForwardResult(
        folds=folds,
        oos_combined_returns=oos_combined,
        oos_combined_metrics=oos_metrics,
        per_fold_metrics=per_fold_metrics,
    )


def _generate_folds(
    index: pd.DatetimeIndex,
    config: WalkForwardConfig,
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """Generate walk-forward fold date ranges.

    Returns list of (is_start, is_end, oos_start, oos_end) tuples.
    """
    data_start = index.min()
    data_end = index.max()

    folds = []
    is_start = data_start

    for _ in range(config.n_folds):
        is_end = is_start + pd.DateOffset(months=config.in_sample_months)
        oos_start = is_end + timedelta(days=config.purge_gap_days)
        oos_end = oos_start + pd.DateOffset(months=config.out_of_sample_months)

        # Clip to data range
        if oos_start > data_end:
            break

        oos_end = min(oos_end, data_end)

        folds.append((
            pd.Timestamp(is_start),
            pd.Timestamp(is_end),
            pd.Timestamp(oos_start),
            pd.Timestamp(oos_end),
        ))

        # Step forward
        is_start = is_start + pd.DateOffset(months=config.step_months)

    return folds
