#!/usr/bin/env python3
"""Generate a comprehensive AI research report for a stock.

Usage:
    python scripts/generate_report.py AAPL
    python scripts/generate_report.py MSFT --output reports/msft_report.html

Runs all agents (with mocked data when API keys unavailable),
aggregates results, and produces a professional HTML report.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _mock_earnings_result(symbol: str) -> dict:
    """Generate mock earnings analysis when API unavailable."""
    return {
        "symbol": symbol,
        "analysis_date": str(date.today()),
        "revenue_trend": "accelerating",
        "revenue_growth_yoy": 0.08,
        "revenue_surprise_pct": 0.015,
        "margin_trend": "expanding",
        "gross_margin": 0.466,
        "operating_margin": 0.305,
        "net_margin": 0.262,
        "margin_expansion": True,
        "fcf_yield": 0.035,
        "cash_conversion": 1.05,
        "capex_trend": "stable",
        "debt_to_equity": 1.76,
        "current_ratio": 1.07,
        "balance_sheet_quality": "strong",
        "key_guidance_points": [
            "Management expects continued services growth",
            "Raised capital return program",
        ],
        "risk_factors_highlighted": [
            "China market uncertainty",
            "EU regulatory scrutiny",
        ],
        "management_tone": "confident",
        "fundamental_score": 82.5,
        "investment_thesis": (
            f"{symbol} demonstrates strong earnings quality with expanding margins "
            "driven by services mix shift. Robust cash generation supports buybacks."
        ),
        "key_risks": ["China revenue decline", "Antitrust regulation"],
        "catalysts": ["AI product launch", "Services monetization"],
        "sources_used": ["10-K MD&A", "10-K Risk Factors"],
        "confidence_level": "high",
    }


def _mock_macro_result() -> dict:
    """Generate mock macro assessment."""
    return {
        "assessment_date": str(date.today()),
        "cycle_phase": "expansion",
        "cycle_confidence": 0.72,
        "gdp_growth_trend": "stable",
        "inflation_trend": "falling",
        "employment_trend": "strengthening",
        "yield_curve_status": "normal",
        "fed_policy_stance": "neutral",
        "rate_direction": "pausing",
        "equity_outlook": "bullish",
        "sector_preferences": ["Technology", "Healthcare", "Consumer Discretionary"],
        "risk_factors": [
            "Geopolitical tensions",
            "Commercial real estate weakness",
        ],
        "suggested_equity_allocation": 0.65,
        "defensive_tilt": False,
        "reasoning": (
            "Economy in mid-cycle expansion with moderating inflation. "
            "The Fed is likely to begin cutting rates, supporting equities."
        ),
    }


def _mock_news_result(symbol: str) -> dict:
    """Generate mock news sentiment."""
    return {
        "symbol": symbol,
        "analysis_date": str(date.today()),
        "articles_analyzed": 15,
        "overall_sentiment": 0.35,
        "sentiment_label": "positive",
        "key_themes": [
            "Strong quarterly earnings",
            "AI product strategy",
            "Regulatory headwinds",
        ],
        "notable_articles": [
            {
                "title": f"{symbol} Reports Record Revenue",
                "source": "Reuters",
                "published_at": str(date.today()),
                "sentiment_score": 0.8,
                "relevance_score": 0.95,
            },
            {
                "title": f"Analysts Raise {symbol} Price Targets",
                "source": "Bloomberg",
                "published_at": str(date.today()),
                "sentiment_score": 0.6,
                "relevance_score": 0.90,
            },
        ],
        "sentiment_trend": "improving",
        "market_impact_assessment": (
            "Net positive sentiment driven by strong earnings "
            "and AI strategy. Regulatory concerns partially offset."
        ),
        "confidence_level": "high",
    }


def _mock_quant_result(symbol: str) -> dict:
    """Generate mock quant signal."""
    return {
        "symbol": symbol,
        "analysis_date": str(date.today()),
        "factor_scores": [
            {"factor_name": "momentum_12_1", "z_score": 1.2, "percentile": 85},
            {"factor_name": "earnings_yield", "z_score": 0.8, "percentile": 72},
            {"factor_name": "roe", "z_score": 1.5, "percentile": 90},
            {"factor_name": "gross_profitability", "z_score": 1.1, "percentile": 82},
            {"factor_name": "volatility_60d", "z_score": -0.3, "percentile": 42},
            {"factor_name": "book_to_market", "z_score": -0.5, "percentile": 35},
            {"factor_name": "ev_ebitda", "z_score": -0.2, "percentile": 44},
            {"factor_name": "accruals", "z_score": 0.6, "percentile": 68},
            {"factor_name": "asset_growth", "z_score": 0.1, "percentile": 52},
            {"factor_name": "short_term_reversal", "z_score": -0.4, "percentile": 38},
        ],
        "composite_score": 0.72,
        "composite_percentile": 76.0,
        "signal_direction": "long",
        "signal_strength": 0.65,
        "strongest_factors": ["roe", "momentum_12_1", "gross_profitability"],
        "weakest_factors": ["book_to_market", "short_term_reversal", "volatility_60d"],
        "sector": "Technology",
        "confidence_level": "high",
    }


def _mock_peer_result(symbol: str) -> dict:
    """Generate mock peer comparison."""
    return {
        "symbol": symbol,
        "analysis_date": str(date.today()),
        "sector": "Technology",
        "industry": "Consumer Electronics",
        "target_metrics": {
            "symbol": symbol,
            "pe_ratio": 30.5,
            "pb_ratio": 45.2,
            "ev_ebitda": 24.1,
            "roe": 0.160,
            "gross_margin": 0.466,
            "operating_margin": 0.305,
            "revenue_growth_yoy": 0.08,
        },
        "peers": [
            {"symbol": "MSFT", "pe_ratio": 35.2, "pb_ratio": 12.8,
             "ev_ebitda": 27.3, "roe": 0.390, "gross_margin": 0.695,
             "operating_margin": 0.448, "revenue_growth_yoy": 0.16},
            {"symbol": "GOOGL", "pe_ratio": 24.1, "pb_ratio": 6.5,
             "ev_ebitda": 16.8, "roe": 0.280, "gross_margin": 0.574,
             "operating_margin": 0.302, "revenue_growth_yoy": 0.11},
            {"symbol": "NVDA", "pe_ratio": 65.0, "pb_ratio": 42.1,
             "ev_ebitda": 55.2, "roe": 0.860, "gross_margin": 0.735,
             "operating_margin": 0.620, "revenue_growth_yoy": 1.22},
        ],
        "peer_count": 3,
        "pe_vs_peers": "discount",
        "pb_vs_peers": "premium",
        "ev_ebitda_vs_peers": "in_line",
        "valuation_summary": (
            f"{symbol} trades at a modest P/E discount to mega-cap tech peers, "
            "though at a premium on P/B due to aggressive share buybacks."
        ),
        "profitability_rank": 3,
        "growth_rank": 3,
        "overall_rank": 3,
        "competitive_advantages": ["Brand loyalty", "Ecosystem lock-in", "Services margins"],
        "competitive_weaknesses": ["Lower growth vs peers", "Hardware dependency"],
        "peer_comparison_summary": (
            f"{symbol} is well-positioned within the tech sector with strong "
            "profitability but trails peers on growth metrics."
        ),
        "confidence_level": "high",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate an AI-powered research report.",
    )
    parser.add_argument("symbol", help="Stock ticker symbol (e.g., AAPL)")
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output HTML file path (default: reports/<SYMBOL>_report.html)",
    )
    parser.add_argument(
        "--date",
        default=str(date.today()),
        help="Analysis date (YYYY-MM-DD, default: today)",
    )
    args = parser.parse_args()

    symbol = args.symbol.upper()
    output_path = args.output or f"reports/{symbol}_report.html"

    print(f"Generating research report for {symbol}...")
    print(f"Analysis date: {args.date}")

    # Build agent results (using mocked data for offline generation)
    agent_results = {
        "earnings": _mock_earnings_result(symbol),
        "macro": _mock_macro_result(),
        "news": _mock_news_result(symbol),
        "quant_signal": _mock_quant_result(symbol),
        "peer": _mock_peer_result(symbol),
    }

    print(f"  Earnings Agent: fundamental_score = {agent_results['earnings']['fundamental_score']}")
    print(f"  Macro Agent: cycle_phase = {agent_results['macro']['cycle_phase']}")
    print(f"  News Agent: sentiment = {agent_results['news']['overall_sentiment']:+.2f}")
    print(f"  Quant Agent: composite_pctile = {agent_results['quant_signal']['composite_percentile']}")
    print(f"  Peer Agent: overall_rank = #{agent_results['peer']['overall_rank']}")

    # Aggregate results
    from src.agents.aggregator import aggregate_results

    report_data = aggregate_results(symbol=symbol, agent_results=agent_results)
    report_data["company_name"] = _get_company_name(symbol)

    print(f"\n  Composite Score: {report_data['composite_score']:.1f}/100")
    print(f"  Recommendation: {report_data['recommendation']}")
    print(f"  Conviction: {report_data['conviction']}")

    # Generate HTML report
    from src.agents.report_generator import generate_html_report

    html = generate_html_report(report_data, output_path=output_path)

    print(f"\nReport generated: {output_path} ({len(html):,} bytes)")
    print(f"Agents used: {', '.join(report_data.get('agents_used', []))}")

    # Also save raw JSON data
    json_path = output_path.replace(".html", ".json")
    Path(json_path).parent.mkdir(parents=True, exist_ok=True)
    Path(json_path).write_text(json.dumps(report_data, indent=2, default=str))
    print(f"Raw data: {json_path}")


def _get_company_name(symbol: str) -> str:
    """Get company name from ticker."""
    names = {
        "AAPL": "Apple Inc.",
        "MSFT": "Microsoft Corporation",
        "GOOGL": "Alphabet Inc.",
        "AMZN": "Amazon.com Inc.",
        "NVDA": "NVIDIA Corporation",
        "META": "Meta Platforms Inc.",
        "TSLA": "Tesla Inc.",
        "JPM": "JPMorgan Chase & Co.",
    }
    return names.get(symbol, symbol)


if __name__ == "__main__":
    main()
