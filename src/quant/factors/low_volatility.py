"""Low Volatility factor.

10. 60-Day Volatility (Ang et al., 2006)
    - Rolling 60-day return standard deviation (NEGATIVE signal)
    - Lower volatility → higher risk-adjusted expected return
    - The "low volatility anomaly" — one of the most robust anomalies
"""

import numpy as np
import pandas as pd

from src.quant.factors.base import Factor


class Volatility60D(Factor):
    """60-Day Rolling Volatility (negative signal).

    Lower realized volatility → higher expected risk-adjusted return.
    We negate so that lower-vol stocks get higher signal values.
    """

    def __init__(self) -> None:
        super().__init__(name="volatility_60d", category="low_vol")

    def compute_raw(
        self,
        prices: pd.DataFrame,
        fundamentals: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        # Daily returns
        returns = prices.pct_change()

        # 60-day rolling standard deviation, annualized
        vol_60d = returns.rolling(window=60, min_periods=40).std() * np.sqrt(252)

        # Negate: lower vol = higher signal
        return -vol_60d
