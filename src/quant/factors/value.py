"""Value factors.

3. Earnings Yield (Basu, 1977)
   - E/P ratio using trailing 12-month earnings
   - Higher earnings yield → cheaper → higher expected return

4. Book-to-Market (Fama & French, 1993)
   - Book value / market cap
   - Higher B/M → "value" stock → higher expected return

5. EV/EBITDA (Loughran & Wellman, 2011)
   - Enterprise Value / EBITDA (NEGATIVE signal — lower is cheaper)
   - More robust than P/E: ignores capital structure and tax differences

All fundamental data uses point-in-time (filing_date) to prevent look-ahead bias.
"""

import numpy as np
import pandas as pd

from src.quant.factors.base import Factor, merge_pit_fundamentals


class EarningsYield(Factor):
    """Earnings Yield (E/P): trailing EPS / price.

    Uses point-in-time EPS from fundamentals table.
    Higher value = cheaper = positive signal.
    """

    def __init__(self) -> None:
        super().__init__(name="earnings_yield", category="value")

    def compute_raw(
        self,
        prices: pd.DataFrame,
        fundamentals: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        if fundamentals is None or fundamentals.empty:
            return pd.DataFrame(index=prices.index, columns=prices.columns) * np.nan

        # Get point-in-time EPS
        eps_pit = merge_pit_fundamentals(fundamentals, prices.index, "eps")

        # Align columns
        common = prices.columns.intersection(eps_pit.columns)
        if common.empty:
            return pd.DataFrame(index=prices.index, columns=prices.columns) * np.nan

        # E/P = EPS / Price
        ep = eps_pit[common] / prices[common]

        # Fill missing columns with NaN
        result = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
        result[common] = ep

        return result


class BookToMarket(Factor):
    """Book-to-Market: book value per share / price.

    Uses point-in-time book value from fundamentals table.
    Higher value = cheaper = positive signal.
    """

    def __init__(self) -> None:
        super().__init__(name="book_to_market", category="value")

    def compute_raw(
        self,
        prices: pd.DataFrame,
        fundamentals: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        if fundamentals is None or fundamentals.empty:
            return pd.DataFrame(index=prices.index, columns=prices.columns) * np.nan

        bvps_pit = merge_pit_fundamentals(fundamentals, prices.index, "book_value_per_share")

        common = prices.columns.intersection(bvps_pit.columns)
        if common.empty:
            return pd.DataFrame(index=prices.index, columns=prices.columns) * np.nan

        bm = bvps_pit[common] / prices[common]

        result = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
        result[common] = bm

        return result


class EVToEBITDA(Factor):
    """EV/EBITDA (negative signal): lower EV/EBITDA = cheaper = better.

    Enterprise Value = Market Cap + Debt - Cash
    EBITDA ≈ Operating Income + Depreciation

    Since fundamentals may not have explicit EBITDA, we approximate:
    EBITDA ≈ operating_income (a reasonable proxy for annual data).

    We negate the ratio so higher signal = more attractive.
    """

    def __init__(self) -> None:
        super().__init__(name="ev_ebitda", category="value")

    def compute_raw(
        self,
        prices: pd.DataFrame,
        fundamentals: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        if fundamentals is None or fundamentals.empty:
            return pd.DataFrame(index=prices.index, columns=prices.columns) * np.nan

        # Use operating_income as EBITDA proxy
        oi_pit = merge_pit_fundamentals(fundamentals, prices.index, "operating_income")
        shares_pit = merge_pit_fundamentals(fundamentals, prices.index, "shares_outstanding")
        debt_pit = merge_pit_fundamentals(fundamentals, prices.index, "total_liabilities")

        common = prices.columns.intersection(oi_pit.columns)
        if common.empty:
            return pd.DataFrame(index=prices.index, columns=prices.columns) * np.nan

        # Market cap = price × shares
        shares = shares_pit.reindex(columns=common)
        market_cap = prices[common] * shares

        # Enterprise value = market_cap + debt - (total_assets - total_liabilities) ... simplified
        # A simpler proxy: EV ≈ market_cap + total_liabilities
        debt = debt_pit.reindex(columns=common).fillna(0)
        ev = market_cap + debt

        # EBITDA proxy
        ebitda = oi_pit[common]

        # Avoid division by zero
        ebitda_safe = ebitda.replace(0, np.nan)

        # EV/EBITDA (negate: lower is better)
        ev_ebitda = -(ev / ebitda_safe)

        result = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
        result[common] = ev_ebitda

        return result
