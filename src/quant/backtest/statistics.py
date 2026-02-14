"""Statistical tests for backtesting results.

Provides significance testing to answer: "Is this alpha real or luck?"
- Alpha t-statistic and p-value
- Sharpe ratio confidence interval
- Drawdown analysis
- Monthly returns heatmap data
"""

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from src.config.logging_config import get_logger

logger = get_logger("backtest.statistics")


def alpha_significance(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> dict:
    """Test whether portfolio alpha is statistically significant.

    Uses a simple t-test on active returns (portfolio - benchmark).

    Returns:
        Dict with alpha, t_stat, p_value, significant (at 5% level).
    """
    active = (portfolio_returns - benchmark_returns).dropna()

    if len(active) < 30:
        return {
            "alpha_daily": 0.0,
            "alpha_annualized": 0.0,
            "t_stat": 0.0,
            "p_value": 1.0,
            "significant": False,
            "n_obs": len(active),
        }

    alpha_daily = active.mean()
    alpha_ann = alpha_daily * 252
    se = active.std() / np.sqrt(len(active))
    t_stat = alpha_daily / se if se > 0 else 0.0
    p_value = 2 * (1 - scipy_stats.t.cdf(abs(t_stat), df=len(active) - 1))

    return {
        "alpha_daily": round(alpha_daily, 6),
        "alpha_annualized": round(alpha_ann, 4),
        "t_stat": round(t_stat, 4),
        "p_value": round(p_value, 4),
        "significant": p_value < 0.05,
        "n_obs": len(active),
    }


def drawdown_analysis(returns: pd.Series) -> dict:
    """Analyze drawdown characteristics.

    Returns:
        Dict with max_drawdown, avg_drawdown, max_drawdown_duration,
        current_drawdown, and drawdown_series.
    """
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max

    max_dd = drawdown.min()

    # Drawdown duration
    in_drawdown = drawdown < 0
    if in_drawdown.any():
        # Find consecutive drawdown periods
        dd_groups = (~in_drawdown).cumsum()
        dd_lengths = in_drawdown.groupby(dd_groups).sum()
        max_dd_duration = int(dd_lengths.max()) if len(dd_lengths) > 0 else 0
    else:
        max_dd_duration = 0

    # Average drawdown (when in drawdown)
    dd_values = drawdown[drawdown < 0]
    avg_dd = dd_values.mean() if len(dd_values) > 0 else 0.0

    return {
        "max_drawdown": round(max_dd, 4),
        "avg_drawdown": round(avg_dd, 4),
        "max_drawdown_duration_days": max_dd_duration,
        "current_drawdown": round(drawdown.iloc[-1], 4) if len(drawdown) > 0 else 0.0,
        "drawdown_series": drawdown,
    }


def monthly_returns_table(returns: pd.Series) -> pd.DataFrame:
    """Compute monthly returns table (heatmap data).

    Returns:
        DataFrame with years as index, months as columns, values are returns.
    """
    monthly = returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)

    table = pd.DataFrame(index=sorted(monthly.index.year.unique()))
    for month in range(1, 13):
        month_data = monthly[monthly.index.month == month]
        for dt, val in month_data.items():
            table.loc[dt.year, month] = val

    table.columns = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]

    # Add yearly total
    yearly = returns.resample("YE").apply(lambda x: (1 + x).prod() - 1)
    for dt, val in yearly.items():
        table.loc[dt.year, "Year"] = val

    return table


def rolling_metrics(
    returns: pd.Series,
    window: int = 63,  # ~3 months
) -> pd.DataFrame:
    """Compute rolling performance metrics.

    Args:
        returns: Daily returns.
        window: Rolling window in trading days.

    Returns:
        DataFrame with rolling Sharpe, volatility, and return.
    """
    rolling_ret = returns.rolling(window).mean() * 252
    rolling_vol = returns.rolling(window).std() * np.sqrt(252)
    rolling_sharpe = rolling_ret / rolling_vol.replace(0, np.nan)

    return pd.DataFrame({
        "rolling_return": rolling_ret,
        "rolling_volatility": rolling_vol,
        "rolling_sharpe": rolling_sharpe,
    })
