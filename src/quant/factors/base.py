"""Abstract base class for all factor computations.

Every factor implements this interface. The pipeline:
1. compute_raw() — raw factor values per (symbol, date)
2. cross_sectional_scores() — z-score + percentile per date cross-section
3. Result stored in factor_signals table

All computations are vectorized (pandas/numpy). No row-by-row loops.
"""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from src.config.logging_config import get_logger


class Factor(ABC):
    """Base class for factor signal computation."""

    def __init__(self, name: str, category: str) -> None:
        self.name = name
        self.category = category  # 'momentum', 'value', 'quality', 'low_vol'
        self.logger = get_logger(f"factor.{name}")

    @abstractmethod
    def compute_raw(
        self,
        prices: pd.DataFrame,
        fundamentals: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Compute raw factor values.

        Args:
            prices: Wide-format price data with DatetimeIndex and symbol columns.
                    Each column is a symbol, values are adj_close prices.
            fundamentals: Optional point-in-time fundamental data.
                         Must have columns: symbol, filing_date, and relevant fields.

        Returns:
            DataFrame with DatetimeIndex, symbol columns, factor values as data.
            Same shape as prices (wide format).
        """

    def compute(
        self,
        prices: pd.DataFrame,
        fundamentals: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Full pipeline: raw values → cross-sectional z-scores.

        Returns long-format DataFrame with columns:
            symbol, date, factor_name, raw_value, z_score, percentile
        """
        raw = self.compute_raw(prices, fundamentals)

        if raw.empty:
            return pd.DataFrame(
                columns=["symbol", "date", "factor_name", "raw_value", "z_score", "percentile"]
            )

        # Cross-sectional z-score and percentile per date
        z_scores = raw.apply(cross_sectional_zscore, axis=1)
        percentiles = raw.rank(axis=1, pct=True) * 100

        # Reshape wide → long
        result = self._to_long_format(raw, z_scores, percentiles)

        self.logger.info(
            "factor_computed",
            factor=self.name,
            dates=len(raw),
            symbols=len(raw.columns),
            rows=len(result),
        )

        return result

    def _to_long_format(
        self,
        raw: pd.DataFrame,
        z_scores: pd.DataFrame,
        percentiles: pd.DataFrame,
    ) -> pd.DataFrame:
        """Convert wide-format factor data to long format for DB storage."""
        raw_long = raw.stack().reset_index()
        raw_long.columns = ["date", "symbol", "raw_value"]

        z_long = z_scores.stack().reset_index()
        z_long.columns = ["date", "symbol", "z_score"]

        pct_long = percentiles.stack().reset_index()
        pct_long.columns = ["date", "symbol", "percentile"]

        result = raw_long.merge(z_long, on=["date", "symbol"])
        result = result.merge(pct_long, on=["date", "symbol"])
        result["factor_name"] = self.name

        # Drop rows where raw_value is NaN
        result = result.dropna(subset=["raw_value"])

        return result[["symbol", "date", "factor_name", "raw_value", "z_score", "percentile"]]


def cross_sectional_zscore(row: pd.Series) -> pd.Series:
    """Compute z-score across a cross-section (single date).

    Winsorizes at 3 sigma before computing z-score to limit
    the influence of extreme outliers.
    """
    valid = row.dropna()
    if len(valid) < 3:
        return row * np.nan

    mean = valid.mean()
    std = valid.std()

    if std == 0 or np.isnan(std):
        return row * 0.0

    z = (row - mean) / std

    # Winsorize at ±3σ
    z = z.clip(lower=-3.0, upper=3.0)

    # Re-standardize after winsorizing
    z_valid = z.dropna()
    if len(z_valid) > 1 and z_valid.std() > 0:
        z = (z - z_valid.mean()) / z_valid.std()

    return z


def prepare_price_matrix(
    market_data: pd.DataFrame,
    price_col: str = "adj_close",
) -> pd.DataFrame:
    """Convert long-format market data to wide-format price matrix.

    Args:
        market_data: Long-format with columns [symbol, date, adj_close, ...].
        price_col: Which price column to use.

    Returns:
        Wide-format DataFrame: DatetimeIndex, symbol columns, price values.
    """
    df = market_data[["symbol", "date", price_col]].copy()
    df["date"] = pd.to_datetime(df["date"])
    pivot = df.pivot(index="date", columns="symbol", values=price_col)
    pivot = pivot.sort_index()
    return pivot


def merge_pit_fundamentals(
    fundamentals: pd.DataFrame,
    dates: pd.DatetimeIndex,
    field: str,
) -> pd.DataFrame:
    """Merge point-in-time fundamentals onto a date grid.

    For each (symbol, date), returns the most recent fundamental value
    whose filing_date <= date. This prevents look-ahead bias.

    Args:
        fundamentals: Must have columns [symbol, filing_date, field].
        dates: Target date grid.
        field: Fundamental field to extract.

    Returns:
        Wide-format DataFrame: dates as index, symbols as columns.
    """
    if fundamentals.empty or field not in fundamentals.columns:
        return pd.DataFrame(index=dates)

    df = fundamentals[["symbol", "filing_date", field]].copy()
    df["filing_date"] = pd.to_datetime(df["filing_date"])
    df = df.dropna(subset=[field])
    df = df.sort_values("filing_date")

    # For each symbol, forward-fill from filing dates onto the date grid
    symbols = df["symbol"].unique()
    result = pd.DataFrame(index=dates)

    for symbol in symbols:
        sym_data = df[df["symbol"] == symbol].set_index("filing_date")[field]
        sym_data = sym_data[~sym_data.index.duplicated(keep="last")]
        # Reindex to the target dates with forward fill
        aligned = sym_data.reindex(dates, method="ffill")
        result[symbol] = aligned

    return result
