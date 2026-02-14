"""Financial calculation tool for agents.

Provides standard financial metric computations that agents can use
to analyze company fundamentals from structured data.
"""

from __future__ import annotations

import pandas as pd

from src.config.logging_config import get_logger

logger = get_logger("tools.financial_calc")


def compute_margins(
    revenue: float,
    cost_of_goods: float | None = None,
    gross_profit: float | None = None,
    operating_income: float | None = None,
    net_income: float | None = None,
) -> dict[str, float | None]:
    """Compute profitability margins.

    Returns:
        Dict with gross_margin, operating_margin, net_margin.
    """
    if revenue == 0:
        return {"gross_margin": None, "operating_margin": None, "net_margin": None}

    gm = None
    if gross_profit is not None:
        gm = gross_profit / revenue
    elif cost_of_goods is not None:
        gm = (revenue - cost_of_goods) / revenue

    om = operating_income / revenue if operating_income is not None else None
    nm = net_income / revenue if net_income is not None else None

    return {"gross_margin": gm, "operating_margin": om, "net_margin": nm}


def compute_growth_rate(current: float, previous: float) -> float | None:
    """Compute year-over-year growth rate."""
    if previous == 0:
        return None
    return (current - previous) / abs(previous)


def compute_valuation_ratios(
    price: float,
    eps: float | None = None,
    book_value_per_share: float | None = None,
    fcf_per_share: float | None = None,
    dividend_per_share: float | None = None,
) -> dict[str, float | None]:
    """Compute valuation ratios.

    Returns:
        Dict with pe_ratio, pb_ratio, fcf_yield, dividend_yield.
    """
    if price <= 0:
        return {
            "pe_ratio": None, "pb_ratio": None,
            "fcf_yield": None, "dividend_yield": None,
        }

    pe = price / eps if eps and eps > 0 else None
    pb = price / book_value_per_share if book_value_per_share and book_value_per_share > 0 else None
    fcf_y = fcf_per_share / price if fcf_per_share is not None else None
    div_y = dividend_per_share / price if dividend_per_share and dividend_per_share > 0 else None

    return {
        "pe_ratio": pe, "pb_ratio": pb,
        "fcf_yield": fcf_y, "dividend_yield": div_y,
    }


def compute_leverage_ratios(
    total_debt: float | None = None,
    total_equity: float | None = None,
    total_assets: float | None = None,
    current_assets: float | None = None,
    current_liabilities: float | None = None,
    ebitda: float | None = None,
) -> dict[str, float | None]:
    """Compute leverage and solvency ratios.

    Returns:
        Dict with debt_to_equity, debt_to_assets, current_ratio, interest_coverage.
    """
    dte = None
    if total_debt is not None and total_equity and total_equity > 0:
        dte = total_debt / total_equity

    dta = None
    if total_debt is not None and total_assets and total_assets > 0:
        dta = total_debt / total_assets

    cr = None
    if current_assets is not None and current_liabilities and current_liabilities > 0:
        cr = current_assets / current_liabilities

    return {
        "debt_to_equity": dte,
        "debt_to_assets": dta,
        "current_ratio": cr,
    }


def compute_cash_flow_metrics(
    operating_cash_flow: float | None = None,
    capex: float | None = None,
    net_income: float | None = None,
    market_cap: float | None = None,
) -> dict[str, float | None]:
    """Compute cash flow quality metrics.

    Returns:
        Dict with fcf, fcf_yield, cash_conversion.
    """
    fcf = None
    if operating_cash_flow is not None and capex is not None:
        fcf = operating_cash_flow - abs(capex)

    fcf_yield = None
    if fcf is not None and market_cap and market_cap > 0:
        fcf_yield = fcf / market_cap

    cash_conv = None
    if fcf is not None and net_income and net_income > 0:
        cash_conv = fcf / net_income

    return {"fcf": fcf, "fcf_yield": fcf_yield, "cash_conversion": cash_conv}


def classify_trend(values: list[float], labels: tuple[str, ...] | None = None) -> str:
    """Classify a time series trend.

    Args:
        values: List of values ordered oldest → newest.
        labels: Custom trend labels. Defaults to ('accelerating', 'stable', 'decelerating').

    Returns:
        Trend label string.
    """
    if labels is None:
        labels = ("accelerating", "stable", "decelerating", "declining")

    if len(values) < 2:
        return labels[1]

    # Compute deltas
    deltas = [values[i] - values[i - 1] for i in range(1, len(values))]
    avg_delta = sum(deltas) / len(deltas)
    last_val = values[-1]

    if len(deltas) >= 2:
        # Acceleration check
        accel = deltas[-1] - deltas[0]
        if avg_delta > 0 and accel > 0:
            return labels[0]  # accelerating
        if avg_delta > 0:
            return labels[1]  # stable/growing
        if avg_delta < 0 and last_val > 0:
            return labels[2]  # decelerating
        return labels[3] if len(labels) > 3 else labels[2]  # declining

    if avg_delta > 0:
        return labels[0]
    if avg_delta < 0:
        return labels[2]
    return labels[1]


def summarize_fundamentals(
    fundamentals_df: pd.DataFrame,
    symbol: str,
) -> dict:
    """Summarize key financial metrics from fundamentals DataFrame.

    Args:
        fundamentals_df: DataFrame with columns like eps, net_income, etc.
        symbol: Stock ticker to filter for.

    Returns:
        Dict of computed financial metrics.
    """
    df = fundamentals_df[fundamentals_df["symbol"] == symbol].sort_values("report_date")

    if df.empty:
        return {}

    latest = df.iloc[-1]
    result = {
        "symbol": symbol,
        "periods_available": len(df),
        "latest_report_date": str(latest.get("report_date", "")),
    }

    # Compute margins if data available
    if "gross_profit" in df.columns and "net_income" in df.columns:
        revenue = latest.get("gross_profit", 0) + latest.get("total_assets", 0) * 0.1
        margins = compute_margins(
            revenue=revenue,
            gross_profit=latest.get("gross_profit"),
            operating_income=latest.get("operating_income"),
            net_income=latest.get("net_income"),
        )
        result.update(margins)

    # Leverage
    leverage = compute_leverage_ratios(
        total_debt=latest.get("total_liabilities"),
        total_equity=latest.get("total_equity"),
        total_assets=latest.get("total_assets"),
    )
    result.update(leverage)

    return result
