"""Portfolio optimization methods.

1. Equal-Weight: simplest baseline, surprisingly hard to beat.
2. Risk Parity: weight inversely proportional to volatility.

Both methods apply constraints:
- Max 5% per position
- Max 25% per sector (if sector data available)
- Max 30% monthly turnover
"""

import pandas as pd

from src.config.logging_config import get_logger

logger = get_logger("portfolio.optimizer")


def equal_weight(
    scores: pd.Series,
    n_holdings: int = 50,
    max_weight: float = 0.05,
) -> pd.Series:
    """Select top-N stocks by score and equal-weight them.

    Args:
        scores: Factor scores per symbol (higher = more attractive).
        n_holdings: Number of stocks to hold.
        max_weight: Maximum weight per position.

    Returns:
        Weight vector (pd.Series, index=symbol, sums to 1.0).
    """
    valid = scores.dropna().sort_values(ascending=False)

    if len(valid) == 0:
        return pd.Series(dtype=float)

    selected = valid.head(min(n_holdings, len(valid)))
    n = len(selected)

    weight = min(1.0 / n, max_weight)
    weights = pd.Series(weight, index=selected.index)

    # Renormalize
    weights /= weights.sum()

    return weights


def risk_parity(
    scores: pd.Series,
    volatilities: pd.Series,
    n_holdings: int = 50,
    max_weight: float = 0.05,
) -> pd.Series:
    """Select top-N stocks and weight inversely proportional to volatility.

    Risk parity ensures each stock contributes equal risk to the portfolio.

    Args:
        scores: Factor scores per symbol.
        volatilities: Annualized volatility per symbol.
        n_holdings: Number of stocks to hold.
        max_weight: Maximum weight per position.

    Returns:
        Weight vector (pd.Series, index=symbol, sums to 1.0).
    """
    valid = scores.dropna().sort_values(ascending=False)

    if len(valid) == 0:
        return pd.Series(dtype=float)

    selected = valid.head(min(n_holdings, len(valid)))

    # Get volatilities for selected stocks
    vol = volatilities.reindex(selected.index).dropna()

    if vol.empty or (vol <= 0).all():
        return equal_weight(scores, n_holdings, max_weight)

    # Inverse volatility weights
    inv_vol = 1.0 / vol.clip(lower=0.01)
    weights = inv_vol / inv_vol.sum()

    # Apply max weight constraint
    weights = weights.clip(upper=max_weight)

    # Renormalize
    weights /= weights.sum()

    return weights


def apply_sector_constraint(
    weights: pd.Series,
    sector_map: dict[str, str],
    max_sector_weight: float = 0.25,
) -> pd.Series:
    """Apply maximum sector weight constraint.

    Iteratively caps over-weighted sectors and redistributes excess
    to sectors that still have capacity.

    Args:
        weights: Current portfolio weights.
        sector_map: Dict mapping symbol → sector.
        max_sector_weight: Maximum total weight per sector.

    Returns:
        Adjusted weight vector.
    """
    if not sector_map:
        return weights

    weights = weights.copy()
    sectors = pd.Series({s: sector_map.get(s, "Unknown") for s in weights.index})
    capped: set[str] = set()

    for _ in range(20):
        sector_weights = weights.groupby(sectors).sum()
        over = sector_weights[
            (sector_weights > max_sector_weight + 1e-8)
            & (~sector_weights.index.isin(capped))
        ]

        if over.empty:
            break

        for sector in over.index:
            mask = sectors == sector
            sw = weights[mask].sum()
            if sw > max_sector_weight:
                weights[mask] *= max_sector_weight / sw
                capped.add(sector)

        # Redistribute to non-capped sectors so total sums to 1.0
        capped_total = sum(weights[sectors == s].sum() for s in capped)
        uncapped_mask = ~sectors.isin(capped)
        uncapped_total = weights[uncapped_mask].sum()

        target_uncapped = 1.0 - capped_total
        if uncapped_total > 0 and target_uncapped > 0:
            weights[uncapped_mask] *= target_uncapped / uncapped_total

    # Final normalize if needed
    if weights.sum() > 0 and abs(weights.sum() - 1.0) > 0.01:
        weights /= weights.sum()

    return weights


def apply_turnover_constraint(
    target_weights: pd.Series,
    current_weights: pd.Series,
    max_turnover: float = 0.30,
) -> pd.Series:
    """Limit portfolio turnover to control trading costs.

    If target turnover exceeds max_turnover, blend target with current
    weights to stay within the constraint.

    Args:
        target_weights: Desired new weights.
        current_weights: Current portfolio weights.
        max_turnover: Maximum one-way turnover.

    Returns:
        Blended weight vector respecting turnover constraint.
    """
    # Align indices
    all_syms = target_weights.index.union(current_weights.index)
    target = target_weights.reindex(all_syms).fillna(0)
    current = current_weights.reindex(all_syms).fillna(0)

    turnover = (target - current).abs().sum() / 2

    if turnover <= max_turnover:
        return target_weights

    # Blend: new = alpha * target + (1 - alpha) * current
    alpha = max_turnover / max(turnover, 1e-8)
    alpha = min(alpha, 1.0)

    blended = alpha * target + (1 - alpha) * current

    # Renormalize
    if blended.sum() > 0:
        blended /= blended.sum()

    # Remove tiny positions
    blended = blended[blended > 0.001]
    if blended.sum() > 0:
        blended /= blended.sum()

    return blended
