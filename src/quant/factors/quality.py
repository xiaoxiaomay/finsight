"""Quality factors.

6. ROE — Return on Equity (Novy-Marx, 2013)
   - Net Income / Shareholders' Equity
   - Higher ROE → more profitable → positive signal

7. Gross Profitability (Novy-Marx, 2013)
   - Gross Profit / Total Assets
   - "The other side of value" — profitable firms outperform

8. Accruals (Sloan, 1996)
   - Change in non-cash working capital / total assets
   - NEGATIVE signal: high accruals predict lower future returns

9. Asset Growth (Cooper et al., 2008)
   - Year-over-year total asset growth
   - NEGATIVE signal: firms growing assets aggressively tend to underperform

All fundamental data uses point-in-time (filing_date) to prevent look-ahead bias.
"""

import numpy as np
import pandas as pd

from src.quant.factors.base import Factor, merge_pit_fundamentals


class ROE(Factor):
    """Return on Equity: net_income / total_equity.

    Higher ROE = more efficient capital use = positive signal.
    """

    def __init__(self) -> None:
        super().__init__(name="roe", category="quality")

    def compute_raw(
        self,
        prices: pd.DataFrame,
        fundamentals: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        if fundamentals is None or fundamentals.empty:
            return pd.DataFrame(index=prices.index, columns=prices.columns) * np.nan

        ni_pit = merge_pit_fundamentals(fundamentals, prices.index, "net_income")
        eq_pit = merge_pit_fundamentals(fundamentals, prices.index, "total_equity")

        common = prices.columns.intersection(ni_pit.columns).intersection(eq_pit.columns)
        if common.empty:
            return pd.DataFrame(index=prices.index, columns=prices.columns) * np.nan

        eq_safe = eq_pit[common].replace(0, np.nan)
        roe = ni_pit[common] / eq_safe

        result = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
        result[common] = roe

        return result


class GrossProfitability(Factor):
    """Gross Profitability: gross_profit / total_assets.

    Novy-Marx (2013) showed this is a strong predictor of returns,
    capturing "the other side of value."
    """

    def __init__(self) -> None:
        super().__init__(name="gross_profitability", category="quality")

    def compute_raw(
        self,
        prices: pd.DataFrame,
        fundamentals: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        if fundamentals is None or fundamentals.empty:
            return pd.DataFrame(index=prices.index, columns=prices.columns) * np.nan

        gp_pit = merge_pit_fundamentals(fundamentals, prices.index, "gross_profit")
        ta_pit = merge_pit_fundamentals(fundamentals, prices.index, "total_assets")

        common = prices.columns.intersection(gp_pit.columns).intersection(ta_pit.columns)
        if common.empty:
            return pd.DataFrame(index=prices.index, columns=prices.columns) * np.nan

        ta_safe = ta_pit[common].replace(0, np.nan)
        gp_ta = gp_pit[common] / ta_safe

        result = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
        result[common] = gp_ta

        return result


class Accruals(Factor):
    """Accruals: change in non-cash working capital / total assets.

    NEGATIVE signal: high accruals → earnings are less cash-backed
    → lower future returns (Sloan, 1996).

    Proxy: accruals ≈ (net_income - operating_cash_flow) / total_assets
    """

    def __init__(self) -> None:
        super().__init__(name="accruals", category="quality")

    def compute_raw(
        self,
        prices: pd.DataFrame,
        fundamentals: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        if fundamentals is None or fundamentals.empty:
            return pd.DataFrame(index=prices.index, columns=prices.columns) * np.nan

        ni_pit = merge_pit_fundamentals(fundamentals, prices.index, "net_income")
        ocf_pit = merge_pit_fundamentals(fundamentals, prices.index, "operating_cash_flow")
        ta_pit = merge_pit_fundamentals(fundamentals, prices.index, "total_assets")

        common = (
            prices.columns
            .intersection(ni_pit.columns)
            .intersection(ocf_pit.columns)
            .intersection(ta_pit.columns)
        )
        if common.empty:
            return pd.DataFrame(index=prices.index, columns=prices.columns) * np.nan

        ta_safe = ta_pit[common].replace(0, np.nan)
        accruals = (ni_pit[common] - ocf_pit[common]) / ta_safe

        # Negate: lower accruals = better quality
        result = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
        result[common] = -accruals

        return result


class AssetGrowth(Factor):
    """Asset Growth: YoY total asset growth (negative signal).

    Cooper et al. (2008): firms growing assets aggressively tend
    to underperform. We negate so lower growth = higher signal.
    """

    def __init__(self) -> None:
        super().__init__(name="asset_growth", category="quality")

    def compute_raw(
        self,
        prices: pd.DataFrame,
        fundamentals: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        if fundamentals is None or fundamentals.empty:
            return pd.DataFrame(index=prices.index, columns=prices.columns) * np.nan

        ta_pit = merge_pit_fundamentals(fundamentals, prices.index, "total_assets")

        if ta_pit.empty:
            return pd.DataFrame(index=prices.index, columns=prices.columns) * np.nan

        common = prices.columns.intersection(ta_pit.columns)
        if common.empty:
            return pd.DataFrame(index=prices.index, columns=prices.columns) * np.nan

        # YoY growth ≈ compare to value ~252 trading days ago
        ta = ta_pit[common]
        ta_prev = ta.shift(252)
        ta_prev_safe = ta_prev.replace(0, np.nan)
        growth = (ta - ta_prev) / ta_prev_safe

        # Negate: lower growth = higher signal
        result = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
        result[common] = -growth

        return result
