"""Transaction cost modeling for realistic backtesting.

Models three components of trading costs:
1. Commission: fixed per-share cost
2. Spread: bid-ask spread (half-spread applied per trade)
3. Market impact: price impact from order flow

Total one-way cost ≈ spread_bps + market_impact_bps (+ commission negligible for large orders)
"""

import pandas as pd

from src.config.logging_config import get_logger

logger = get_logger("backtest.costs")


def estimate_transaction_cost(
    turnover: float,
    spread_bps: float = 5.0,
    market_impact_bps: float = 10.0,
) -> float:
    """Estimate total transaction cost for a given turnover.

    Args:
        turnover: One-way turnover as a fraction (0 to 1).
        spread_bps: Half-spread in basis points.
        market_impact_bps: Market impact in basis points.

    Returns:
        Total cost as a fraction of portfolio value.
    """
    total_bps = spread_bps + market_impact_bps
    return turnover * total_bps / 10000


def compute_turnover(
    old_weights: pd.Series,
    new_weights: pd.Series,
) -> float:
    """Compute one-way turnover between two weight vectors.

    Turnover = sum of absolute weight changes / 2
    (dividing by 2 because buys and sells net out).
    """
    combined = old_weights.reindex(old_weights.index.union(new_weights.index)).fillna(0)
    new_combined = new_weights.reindex(combined.index).fillna(0)
    return float((combined - new_combined).abs().sum() / 2)


def net_of_cost_returns(
    gross_returns: pd.Series,
    turnover_series: pd.Series,
    spread_bps: float = 5.0,
    market_impact_bps: float = 10.0,
) -> pd.Series:
    """Adjust gross returns for transaction costs at rebalance points.

    Args:
        gross_returns: Daily gross portfolio returns.
        turnover_series: Turnover at each rebalance date.
        spread_bps: Half-spread in basis points.
        market_impact_bps: Market impact in basis points.

    Returns:
        Daily net-of-cost returns.
    """
    net = gross_returns.copy()

    for dt, to in turnover_series.items():
        if dt in net.index:
            cost = estimate_transaction_cost(to, spread_bps, market_impact_bps)
            net.loc[dt] -= cost

    return net
