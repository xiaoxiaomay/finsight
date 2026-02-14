"""Composite (multi-factor) signal combiner.

Combines multiple factor z-scores into a single composite signal.
Supports equal-weight and custom-weight combination schemes.
"""

import pandas as pd

from src.config.logging_config import get_logger
from src.quant.factors.base import Factor

logger = get_logger("factor.composite")

# Default factor weights (equal weight across categories)
DEFAULT_WEIGHTS: dict[str, float] = {
    "momentum_12_1": 0.15,
    "short_term_reversal": 0.05,
    "earnings_yield": 0.10,
    "book_to_market": 0.10,
    "ev_ebitda": 0.10,
    "roe": 0.10,
    "gross_profitability": 0.10,
    "accruals": 0.10,
    "asset_growth": 0.10,
    "volatility_60d": 0.10,
}


class CompositeFactor(Factor):
    """Multi-factor composite signal.

    Combines z-scored factor signals using configurable weights.
    The composite score is itself z-scored cross-sectionally.
    """

    def __init__(
        self,
        factor_weights: dict[str, float] | None = None,
    ) -> None:
        super().__init__(name="composite", category="composite")
        self.factor_weights = factor_weights or DEFAULT_WEIGHTS.copy()

    def compute_raw(
        self,
        prices: pd.DataFrame,
        fundamentals: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Not used directly — use combine() instead."""
        raise NotImplementedError("Use combine() with precomputed factor z-scores.")

    def combine(
        self,
        factor_zscores: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """Combine multiple factor z-score DataFrames into a composite.

        Args:
            factor_zscores: Dict mapping factor_name → wide-format z-score DataFrame.
                           Each DataFrame has DatetimeIndex and symbol columns.

        Returns:
            Wide-format composite z-score DataFrame.
        """
        if not factor_zscores:
            return pd.DataFrame()

        # Normalize weights to sum to 1
        active_factors = {
            k: v for k, v in self.factor_weights.items()
            if k in factor_zscores
        }

        if not active_factors:
            logger.warning("no_matching_factors", available=list(factor_zscores.keys()))
            return pd.DataFrame()

        weight_sum = sum(active_factors.values())
        if weight_sum <= 0:
            return pd.DataFrame()

        normalized = {k: v / weight_sum for k, v in active_factors.items()}

        # Weighted sum of z-scores
        first_key = next(iter(factor_zscores))
        result = pd.DataFrame(
            0.0,
            index=factor_zscores[first_key].index,
            columns=factor_zscores[first_key].columns,
        )

        factors_used = 0
        for name, weight in normalized.items():
            df = factor_zscores[name]
            # Align to result shape
            aligned = df.reindex(index=result.index, columns=result.columns)
            result = result.add(aligned * weight, fill_value=0.0)
            factors_used += 1

        logger.info(
            "composite_combined",
            factors_used=factors_used,
            weights=normalized,
        )

        # Re-zscore the composite cross-sectionally
        from src.quant.factors.base import cross_sectional_zscore

        result = result.apply(cross_sectional_zscore, axis=1)

        return result

    def compute(
        self,
        prices: pd.DataFrame,
        fundamentals: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Override base compute — composite needs all factor z-scores."""
        raise NotImplementedError(
            "CompositeFactor.compute() requires precomputed factor z-scores. "
            "Use combine() instead, passing a dict of factor z-score DataFrames."
        )


def get_all_factors() -> list[Factor]:
    """Get instances of all 10 Tier 1 factors."""
    from src.quant.factors.low_volatility import Volatility60D
    from src.quant.factors.momentum import Momentum12M1M, ShortTermReversal
    from src.quant.factors.quality import (
        ROE,
        Accruals,
        AssetGrowth,
        GrossProfitability,
    )
    from src.quant.factors.value import BookToMarket, EarningsYield, EVToEBITDA

    return [
        Momentum12M1M(),
        ShortTermReversal(),
        EarningsYield(),
        BookToMarket(),
        EVToEBITDA(),
        ROE(),
        GrossProfitability(),
        Accruals(),
        AssetGrowth(),
        Volatility60D(),
    ]
