"""Portfolio risk metrics.

Computes VaR, CVaR, beta, and other risk measures for portfolio analysis.
"""

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


def value_at_risk(
    returns: pd.Series,
    confidence: float = 0.95,
    method: str = "historical",
) -> float:
    """Compute Value at Risk.

    Args:
        returns: Daily portfolio returns.
        confidence: Confidence level (e.g., 0.95 for 95% VaR).
        method: 'historical' or 'parametric'.

    Returns:
        VaR as a positive number (max expected loss at confidence level).
    """
    if returns.empty:
        return 0.0

    if method == "historical":
        var = -np.percentile(returns.dropna(), (1 - confidence) * 100)
    else:  # parametric (assumes normality)
        mu = returns.mean()
        sigma = returns.std()
        z = scipy_stats.norm.ppf(1 - confidence)
        var = -(mu + z * sigma)

    return float(var)


def conditional_var(
    returns: pd.Series,
    confidence: float = 0.95,
) -> float:
    """Compute Conditional VaR (Expected Shortfall).

    Average loss in the worst (1-confidence)% of days.
    More informative than VaR for tail risk.
    """
    if returns.empty:
        return 0.0

    threshold = np.percentile(returns.dropna(), (1 - confidence) * 100)
    tail_returns = returns[returns <= threshold]

    if tail_returns.empty:
        return value_at_risk(returns, confidence)

    return float(-tail_returns.mean())


def portfolio_beta(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> float:
    """Compute portfolio beta relative to benchmark."""
    aligned = pd.DataFrame({
        "port": portfolio_returns,
        "bench": benchmark_returns,
    }).dropna()

    if len(aligned) < 30:
        return 1.0

    cov = aligned["port"].cov(aligned["bench"])
    var = aligned["bench"].var()

    if var == 0:
        return 1.0

    return float(cov / var)
