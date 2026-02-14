"""Tests for Research Report Generator."""

import tempfile
from pathlib import Path

MOCK_REPORT_DATA = {
    "symbol": "AAPL",
    "company_name": "Apple Inc.",
    "recommendation": "Buy",
    "conviction": "medium",
    "composite_score": 72.5,
    "fundamental_score": 82.5,
    "quant_score": 76.0,
    "sentiment_score": 67.5,
    "macro_score": 65.0,
    "peer_score": 60.0,
    "earnings_analysis": {
        "revenue_trend": "accelerating",
        "revenue_growth_yoy": 0.08,
        "gross_margin": 0.466,
        "operating_margin": 0.305,
        "net_margin": 0.262,
        "fcf_yield": 0.035,
        "debt_to_equity": 1.76,
        "current_ratio": 1.07,
        "balance_sheet_quality": "strong",
        "fundamental_score": 82.5,
        "management_tone": "confident",
        "investment_thesis": "Strong earnings quality.",
        "key_guidance_points": ["Revenue guidance raised"],
    },
    "macro_assessment": {
        "cycle_phase": "expansion",
        "gdp_growth_trend": "stable",
        "inflation_trend": "falling",
        "fed_policy_stance": "neutral",
        "rate_direction": "pausing",
        "equity_outlook": "bullish",
        "sector_preferences": ["Technology"],
        "reasoning": "Mid-cycle expansion.",
    },
    "news_sentiment": {
        "overall_sentiment": 0.35,
        "sentiment_label": "positive",
        "sentiment_trend": "improving",
        "articles_analyzed": 15,
        "key_themes": ["Strong earnings"],
        "notable_articles": [
            {"title": "AAPL Record Revenue", "source": "Reuters", "sentiment_score": 0.8},
        ],
        "market_impact_assessment": "Net positive.",
    },
    "quant_signals": {
        "signal_direction": "long",
        "signal_strength": 0.65,
        "composite_score": 0.72,
        "composite_percentile": 76.0,
        "factor_scores": [
            {"factor_name": "momentum_12_1", "z_score": 1.2, "percentile": 85},
            {"factor_name": "roe", "z_score": 1.5, "percentile": 90},
        ],
        "strongest_factors": ["roe", "momentum_12_1"],
        "weakest_factors": ["book_to_market"],
    },
    "peer_comparison": {
        "sector": "Technology",
        "peer_count": 3,
        "pe_vs_peers": "discount",
        "profitability_rank": 3,
        "growth_rank": 3,
        "peers": [
            {"symbol": "MSFT", "pe_ratio": 35.2, "pb_ratio": 12.8, "roe": 0.39,
             "revenue_growth_yoy": 0.16},
        ],
        "peer_comparison_summary": "Well-positioned.",
    },
    "conflicts": [],
    "agents_used": ["earnings", "macro", "news", "quant_signal", "peer"],
    "data_sources": ["SEC EDGAR", "FRED", "NewsAPI"],
    "key_risks": ["China revenue decline"],
    "catalysts": ["AI launch"],
    "bull_case": ["Strong fundamentals"],
    "bear_case": ["Regulatory risk"],
    "market_regime": "bull",
    "regime_factor_tilt": "momentum + growth",
}


class TestReportGenerator:
    """Test HTML report generation."""

    def test_generates_html(self) -> None:
        from src.agents.report_generator import generate_html_report

        html = generate_html_report(MOCK_REPORT_DATA)

        assert "<!DOCTYPE html>" in html
        assert "AAPL" in html
        assert "Apple Inc." in html

    def test_contains_recommendation(self) -> None:
        from src.agents.report_generator import generate_html_report

        html = generate_html_report(MOCK_REPORT_DATA)

        assert "Buy" in html
        assert "72" in html  # composite score

    def test_contains_all_sections(self) -> None:
        from src.agents.report_generator import generate_html_report

        html = generate_html_report(MOCK_REPORT_DATA)

        assert "Executive Summary" in html
        assert "Financial Analysis" in html
        assert "Macro Environment" in html
        assert "News" in html and "Sentiment" in html
        assert "Quantitative Signals" in html
        assert "Industry" in html and "Peers" in html
        assert "Risk Factors" in html

    def test_writes_to_file(self) -> None:
        from src.agents.report_generator import generate_html_report

        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "test_report.html")
            html = generate_html_report(MOCK_REPORT_DATA, output_path=path)

            assert Path(path).exists()
            content = Path(path).read_text()
            assert content == html

    def test_contains_score_cards(self) -> None:
        from src.agents.report_generator import generate_html_report

        html = generate_html_report(MOCK_REPORT_DATA)

        assert "Fundamental" in html
        assert "Quant" in html
        assert "Sentiment" in html

    def test_contains_factor_table(self) -> None:
        from src.agents.report_generator import generate_html_report

        html = generate_html_report(MOCK_REPORT_DATA)

        assert "momentum_12_1" in html
        assert "roe" in html

    def test_contains_peer_table(self) -> None:
        from src.agents.report_generator import generate_html_report

        html = generate_html_report(MOCK_REPORT_DATA)

        assert "MSFT" in html
        assert "Technology" in html

    def test_contains_source_citations(self) -> None:
        from src.agents.report_generator import generate_html_report

        html = generate_html_report(MOCK_REPORT_DATA)

        assert "Source Citations" in html
        assert "SEC EDGAR" in html

    def test_regime_section_shown(self) -> None:
        from src.agents.report_generator import generate_html_report

        html = generate_html_report(MOCK_REPORT_DATA)

        assert "Market Regime" in html
        assert "Bull" in html

    def test_handles_missing_data(self) -> None:
        from src.agents.report_generator import generate_html_report

        minimal = {
            "symbol": "TEST",
            "recommendation": "Hold",
            "conviction": "low",
            "composite_score": 50,
            "agents_used": [],
        }
        html = generate_html_report(minimal)

        assert "TEST" in html
        assert "Hold" in html
