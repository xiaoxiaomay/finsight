"""FRED API query tool for agents.

Provides access to Federal Reserve Economic Data for macroeconomic analysis.
Wraps the FRED API with caching and trend computation.
"""

from __future__ import annotations

from datetime import date, timedelta

import httpx
import pandas as pd

from src.config.logging_config import get_logger
from src.config.settings import get_settings

logger = get_logger("tools.fred_api")

FRED_BASE = "https://api.stlouisfed.org/fred"

# Key macro series with descriptions
MACRO_SERIES: dict[str, str] = {
    "GDPC1": "Real GDP (quarterly)",
    "CPIAUCSL": "CPI All Items (monthly)",
    "DFF": "Federal Funds Rate (daily)",
    "DGS10": "10-Year Treasury Yield (daily)",
    "DGS2": "2-Year Treasury Yield (daily)",
    "T10Y2Y": "10Y-2Y Yield Spread (daily)",
    "VIXCLS": "VIX Volatility Index (daily)",
    "UNRATE": "Unemployment Rate (monthly)",
    "ICSA": "Initial Jobless Claims (weekly)",
    "UMCSENT": "Consumer Sentiment (monthly)",
    "INDPRO": "Industrial Production (monthly)",
    "BAMLC0A0CM": "Corporate Bond Spread (daily)",
    "HOUST": "Housing Starts (monthly)",
}


def fetch_series(
    series_id: str,
    start_date: date | None = None,
    end_date: date | None = None,
) -> pd.Series:
    """Fetch a single FRED series.

    Args:
        series_id: FRED series ID (e.g., 'DFF', 'GDPC1').
        start_date: Start date for data.
        end_date: End date for data.

    Returns:
        pd.Series with DatetimeIndex and float values.
    """
    settings = get_settings()
    if not settings.fred_api_key:
        logger.warning("no_fred_api_key")
        return pd.Series(dtype=float)

    if start_date is None:
        start_date = date.today() - timedelta(days=365 * 3)
    if end_date is None:
        end_date = date.today()

    params = {
        "series_id": series_id,
        "api_key": settings.fred_api_key,
        "file_type": "json",
        "observation_start": str(start_date),
        "observation_end": str(end_date),
    }

    try:
        resp = httpx.get(
            f"{FRED_BASE}/series/observations",
            params=params,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        logger.warning("fred_fetch_failed", series_id=series_id)
        return pd.Series(dtype=float)

    observations = data.get("observations", [])
    if not observations:
        return pd.Series(dtype=float)

    records = []
    for obs in observations:
        val = obs.get("value", ".")
        if val != ".":
            records.append({
                "date": pd.Timestamp(obs["date"]),
                "value": float(val),
            })

    if not records:
        return pd.Series(dtype=float)

    df = pd.DataFrame(records).set_index("date")
    return df["value"]


def fetch_macro_snapshot(
    series_ids: list[str] | None = None,
) -> dict[str, dict]:
    """Fetch current values and trends for key macro indicators.

    Returns:
        Dict mapping series_id → {value, change, trend, description}.
    """
    if series_ids is None:
        series_ids = list(MACRO_SERIES.keys())

    snapshot = {}
    for sid in series_ids:
        data = fetch_series(sid)
        if data.empty:
            continue

        current = float(data.iloc[-1])
        prev = float(data.iloc[-2]) if len(data) > 1 else current

        # Compute simple trend
        if len(data) >= 20:
            recent_mean = float(data.iloc[-20:].mean())
            older_mean = float(data.iloc[-60:-20].mean()) if len(data) >= 60 else recent_mean
            if recent_mean > older_mean * 1.02:
                trend = "rising"
            elif recent_mean < older_mean * 0.98:
                trend = "falling"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"

        snapshot[sid] = {
            "value": current,
            "previous": prev,
            "change": current - prev,
            "trend": trend,
            "description": MACRO_SERIES.get(sid, sid),
        }

    return snapshot


def compute_yield_curve_status(
    dgs10: pd.Series | None = None,
    dgs2: pd.Series | None = None,
) -> str:
    """Determine yield curve status from Treasury yields.

    Returns:
        One of: 'normal', 'flat', 'inverted'.
    """
    if dgs10 is None:
        dgs10 = fetch_series("DGS10")
    if dgs2 is None:
        dgs2 = fetch_series("DGS2")

    if dgs10.empty or dgs2.empty:
        return "unknown"

    spread = float(dgs10.iloc[-1]) - float(dgs2.iloc[-1])

    if spread > 0.25:
        return "normal"
    if spread > -0.10:
        return "flat"
    return "inverted"


def assess_fed_stance() -> dict[str, str]:
    """Assess Federal Reserve policy stance from data.

    Returns:
        Dict with fed_policy_stance, rate_direction.
    """
    dff = fetch_series("DFF")
    if dff.empty or len(dff) < 60:
        return {"fed_policy_stance": "unknown", "rate_direction": "unknown"}

    current = float(dff.iloc[-1])
    quarter_ago = float(dff.iloc[-63]) if len(dff) >= 63 else current

    # Rate direction
    if current > quarter_ago + 0.25:
        rate_dir = "hiking"
    elif current < quarter_ago - 0.25:
        rate_dir = "cutting"
    else:
        rate_dir = "pausing"

    # Policy stance (relative to neutral ~2.5%)
    if current > 4.0:
        stance = "hawkish"
    elif current < 2.0:
        stance = "dovish"
    else:
        stance = "neutral"

    return {"fed_policy_stance": stance, "rate_direction": rate_dir}
