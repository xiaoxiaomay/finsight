"""Momentum factors.

1. 12-1 Month Momentum (Jegadeesh & Titman, 1993)
   - Past 12-month return, skipping the most recent month
   - Rationale: stocks with strong past returns continue outperforming

2. Short-Term Reversal (Jegadeesh, 1990)
   - Past 1-month return (NEGATIVE signal)
   - Rationale: very recent winners tend to reverse in the short term
"""

import pandas as pd

from src.quant.factors.base import Factor


class Momentum12M1M(Factor):
    """12-1 Month Momentum: cumulative return over months [-12, -1].

    Skips the most recent month to avoid the short-term reversal effect.
    This is the canonical momentum factor from academic literature.
    """

    def __init__(self) -> None:
        super().__init__(name="momentum_12_1", category="momentum")

    def compute_raw(
        self,
        prices: pd.DataFrame,
        fundamentals: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        # Return from t-252 to t-21 (approx 12 months, skip last month)
        # Using trading days: 252 ≈ 12 months, 21 ≈ 1 month
        price_12m_ago = prices.shift(252)
        price_1m_ago = prices.shift(21)

        momentum = (price_1m_ago / price_12m_ago) - 1.0

        return momentum


class ShortTermReversal(Factor):
    """Short-Term Reversal: past 1-month return (negative signal).

    Stocks that went up in the last month tend to reverse.
    The raw value is the return itself; the SIGNAL is negative
    (lower recent return → higher expected future return).
    """

    def __init__(self) -> None:
        super().__init__(name="short_term_reversal", category="momentum")

    def compute_raw(
        self,
        prices: pd.DataFrame,
        fundamentals: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        # Past 21 trading days return (negative signal)
        ret_1m = prices.pct_change(21)

        # Negate: lower recent return = higher signal
        return -ret_1m
