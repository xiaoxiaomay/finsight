"""Barra-style factor neutralization.

Removes systematic sector and size biases from factor signals.
Without neutralization, a "value" signal might really just be a
"small-cap" signal in disguise.

Methodology:
    For each date cross-section:
    1. Regress factor values against sector dummies + log(market_cap)
    2. Take the residual as the "pure" alpha signal
    3. Winsorize at ±3σ to limit outlier influence
    4. Z-score normalize to mean=0, std=1

Interview Talking Point: "I implemented Barra-style factor neutralization
to ensure alpha signals aren't contaminated by sector or size exposure.
This is the same methodology used by MSCI Barra and major quant funds."
"""

import numpy as np
import pandas as pd

from src.config.logging_config import get_logger

logger = get_logger("factor.neutralizer")


class FactorNeutralizer:
    """Cross-sectional factor neutralization (Barra-style)."""

    def neutralize(
        self,
        factor_values: pd.Series,
        sector: pd.Series,
        market_cap: pd.Series,
    ) -> pd.Series:
        """Neutralize a single cross-section of factor values.

        Regresses factor values on sector dummies + log(market_cap),
        returns the residuals (winsorized and z-scored).

        Args:
            factor_values: Factor signal for each stock (index=symbol).
            sector: GICS sector label for each stock (index=symbol).
            market_cap: Market cap for each stock (index=symbol).

        Returns:
            Neutralized, winsorized, z-scored factor signal.
        """
        # Align all inputs to the same index
        common = factor_values.dropna().index
        common = common.intersection(sector.dropna().index)
        common = common.intersection(market_cap[market_cap > 0].dropna().index)

        if len(common) < 10:
            # Not enough data for meaningful regression
            return _winsorize_zscore(factor_values)

        y = factor_values[common].values.astype(float)
        log_mcap = np.log(market_cap[common].values.astype(float))

        # Sector dummies (drop one for identifiability)
        sector_aligned = sector[common]
        sector_dummies = pd.get_dummies(sector_aligned, drop_first=True, dtype=float)

        # Build design matrix: [intercept, log_mcap, sector_dummies]
        design = np.column_stack([
            np.ones(len(common)),
            log_mcap,
            sector_dummies.values,
        ])

        # OLS regression
        try:
            coeffs, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
            predicted = design @ coeffs
            residuals = y - predicted
        except np.linalg.LinAlgError:
            # Fallback: just demean and scale
            residuals = y - np.nanmean(y)

        # Build result Series
        result = pd.Series(np.nan, index=factor_values.index)
        result[common] = residuals

        return _winsorize_zscore(result)

    def neutralize_panel(
        self,
        factor_df: pd.DataFrame,
        sector_map: dict[str, str],
        market_cap_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Neutralize factor values across all dates.

        Args:
            factor_df: Wide-format factor values (dates × symbols).
            sector_map: Dict mapping symbol → sector label.
            market_cap_df: Wide-format market cap (dates × symbols).

        Returns:
            Neutralized factor DataFrame (same shape as input).
        """
        sector_series = pd.Series(sector_map)
        result = pd.DataFrame(index=factor_df.index, columns=factor_df.columns, dtype=float)

        for dt in factor_df.index:
            row_factor = factor_df.loc[dt].dropna()
            if row_factor.empty:
                continue

            row_mcap = market_cap_df.loc[dt] if dt in market_cap_df.index else pd.Series(dtype=float)

            neutralized = self.neutralize(
                factor_values=row_factor,
                sector=sector_series,
                market_cap=row_mcap,
            )

            result.loc[dt, neutralized.index] = neutralized.values

        return result

    @staticmethod
    def compute_factor_exposure(
        weights: pd.Series,
        factor_values: pd.Series,
    ) -> float:
        """Compute portfolio's weighted exposure to a factor.

        Args:
            weights: Portfolio weights (index=symbol, sums to ~1.0).
            factor_values: Factor signal per stock.

        Returns:
            Portfolio-level factor exposure (weighted sum of z-scores).
        """
        common = weights.index.intersection(factor_values.dropna().index)
        if common.empty:
            return 0.0

        return float((weights[common] * factor_values[common]).sum())


def _winsorize_zscore(s: pd.Series, sigma: float = 3.0) -> pd.Series:
    """Winsorize at ±sigma and z-score normalize.

    Args:
        s: Raw values.
        sigma: Winsorization threshold in standard deviations.

    Returns:
        Winsorized, z-scored Series.
    """
    valid = s.dropna()
    if len(valid) < 3:
        return s * 0.0

    mean = valid.mean()
    std = valid.std()

    if std == 0 or np.isnan(std):
        return s * 0.0

    z = (s - mean) / std

    # Winsorize
    z = z.clip(lower=-sigma, upper=sigma)

    # Re-standardize after winsorization
    z_valid = z.dropna()
    if len(z_valid) > 1 and z_valid.std() > 0:
        z = (z - z_valid.mean()) / z_valid.std()

    return z


def compare_before_after(
    raw: pd.Series,
    neutralized: pd.Series,
    sector: pd.Series,
) -> pd.DataFrame:
    """Generate before/after comparison for visualization.

    Shows how neutralization changes factor exposure across sectors.

    Returns:
        DataFrame with columns: sector, raw_mean, neutralized_mean, change.
    """
    common = raw.dropna().index.intersection(neutralized.dropna().index).intersection(sector.index)

    df = pd.DataFrame({
        "sector": sector[common],
        "raw": raw[common],
        "neutralized": neutralized[common],
    })

    summary = df.groupby("sector").agg(
        raw_mean=("raw", "mean"),
        neutralized_mean=("neutralized", "mean"),
        count=("raw", "count"),
    ).reset_index()

    summary["change"] = summary["neutralized_mean"] - summary["raw_mean"]

    return summary
