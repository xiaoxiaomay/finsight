"""Tests for agent Pydantic schemas."""

from datetime import date

import pytest
from pydantic import ValidationError


class TestEarningsSchema:
    """Test EarningsAnalysisOutput schema validation."""

    def test_valid_earnings_output(self) -> None:
        """Should accept valid earnings analysis data."""
        from src.agents.schemas.earnings import EarningsAnalysisOutput

        output = EarningsAnalysisOutput(
            symbol="AAPL",
            analysis_date=date(2024, 6, 15),
            revenue_trend="accelerating",
            revenue_growth_yoy=0.12,
            revenue_surprise_pct=0.02,
            margin_trend="expanding",
            gross_margin=0.45,
            operating_margin=0.30,
            net_margin=0.25,
            margin_expansion=True,
            fcf_yield=0.04,
            cash_conversion=1.1,
            capex_trend="stable",
            debt_to_equity=1.5,
            current_ratio=1.2,
            balance_sheet_quality="strong",
            key_guidance_points=["Revenue guidance raised"],
            risk_factors_highlighted=["Supply chain risk"],
            management_tone="confident",
            fundamental_score=78.5,
            investment_thesis="Strong growth with expanding margins.",
            key_risks=["Macro slowdown"],
            catalysts=["New product launch"],
            sources_used=["10-K Item 7: MD&A"],
            confidence_level="high",
        )

        assert output.symbol == "AAPL"
        assert output.fundamental_score == 78.5
        assert output.margin_expansion is True

    def test_fundamental_score_bounds(self) -> None:
        """Score must be between 0 and 100."""
        from src.agents.schemas.earnings import EarningsAnalysisOutput

        with pytest.raises(ValidationError):
            EarningsAnalysisOutput(
                symbol="X",
                analysis_date=date(2024, 1, 1),
                revenue_trend="stable",
                revenue_growth_yoy=0.0,
                margin_trend="stable",
                gross_margin=0.4,
                operating_margin=0.2,
                net_margin=0.1,
                margin_expansion=False,
                fcf_yield=0.03,
                cash_conversion=0.9,
                capex_trend="stable",
                debt_to_equity=1.0,
                current_ratio=1.5,
                balance_sheet_quality="adequate",
                management_tone="cautious",
                fundamental_score=150.0,  # Out of bounds
                investment_thesis="Test",
                confidence_level="medium",
            )

    def test_default_lists(self) -> None:
        """Default lists should be empty."""
        from src.agents.schemas.earnings import EarningsAnalysisOutput

        output = EarningsAnalysisOutput(
            symbol="MSFT",
            analysis_date=date(2024, 1, 1),
            revenue_trend="stable",
            revenue_growth_yoy=0.08,
            margin_trend="stable",
            gross_margin=0.68,
            operating_margin=0.42,
            net_margin=0.35,
            margin_expansion=False,
            fcf_yield=0.03,
            cash_conversion=1.2,
            capex_trend="increasing",
            debt_to_equity=0.5,
            current_ratio=2.0,
            balance_sheet_quality="strong",
            management_tone="confident",
            fundamental_score=85.0,
            investment_thesis="Solid fundamentals.",
            confidence_level="high",
        )

        assert output.key_guidance_points == []
        assert output.catalysts == []
        assert output.sources_used == []


class TestMacroSchema:
    """Test MacroAssessmentOutput schema validation."""

    def test_valid_macro_output(self) -> None:
        """Should accept valid macro assessment data."""
        from src.agents.schemas.macro import MacroAssessmentOutput

        output = MacroAssessmentOutput(
            assessment_date=date(2024, 6, 15),
            cycle_phase="expansion",
            cycle_confidence=0.75,
            gdp_growth_trend="stable",
            inflation_trend="falling",
            employment_trend="strengthening",
            yield_curve_status="normal",
            fed_policy_stance="neutral",
            rate_direction="pausing",
            equity_outlook="bullish",
            sector_preferences=["Technology", "Healthcare"],
            risk_factors=["Geopolitical tensions"],
            suggested_equity_allocation=0.65,
            defensive_tilt=False,
            reasoning="Economy expanding with controlled inflation.",
        )

        assert output.cycle_phase == "expansion"
        assert output.suggested_equity_allocation == 0.65

    def test_allocation_bounds(self) -> None:
        """Allocation must be between 0.0 and 1.0."""
        from src.agents.schemas.macro import MacroAssessmentOutput

        with pytest.raises(ValidationError):
            MacroAssessmentOutput(
                assessment_date=date(2024, 1, 1),
                cycle_phase="expansion",
                cycle_confidence=0.5,
                gdp_growth_trend="stable",
                inflation_trend="stable",
                employment_trend="stable",
                yield_curve_status="normal",
                fed_policy_stance="neutral",
                rate_direction="pausing",
                equity_outlook="neutral",
                suggested_equity_allocation=1.5,  # Out of bounds
                defensive_tilt=False,
                reasoning="Test",
            )


class TestNewsSchema:
    """Test NewsSentimentOutput schema validation."""

    def test_valid_news_output(self) -> None:
        """Should accept valid news sentiment data."""
        from src.agents.schemas.news import NewsArticle, NewsSentimentOutput

        article = NewsArticle(
            title="Apple Reports Record Revenue",
            source="Reuters",
            published_at="2024-06-01T12:00:00Z",
            url="https://example.com/article",
            summary="Apple beat expectations.",
            sentiment_score=0.8,
            relevance_score=0.95,
        )

        output = NewsSentimentOutput(
            symbol="AAPL",
            analysis_date=date(2024, 6, 15),
            articles_analyzed=10,
            overall_sentiment=0.45,
            sentiment_label="positive",
            key_themes=["Strong earnings", "iPhone sales"],
            notable_articles=[article],
            sentiment_trend="improving",
            market_impact_assessment="Positive near-term impact expected.",
            confidence_level="high",
        )

        assert output.overall_sentiment == 0.45
        assert len(output.notable_articles) == 1
        assert output.notable_articles[0].title == "Apple Reports Record Revenue"

    def test_sentiment_bounds(self) -> None:
        """Sentiment must be between -1.0 and 1.0."""
        from src.agents.schemas.news import NewsArticle

        with pytest.raises(ValidationError):
            NewsArticle(
                title="Test",
                source="Test",
                published_at="2024-01-01",
                sentiment_score=2.0,  # Out of bounds
                relevance_score=0.5,
            )

    def test_empty_articles(self) -> None:
        """Should handle empty article list."""
        from src.agents.schemas.news import NewsSentimentOutput

        output = NewsSentimentOutput(
            symbol="GOOG",
            analysis_date=date(2024, 1, 1),
            articles_analyzed=0,
            overall_sentiment=0.0,
            sentiment_label="neutral",
            sentiment_trend="stable",
            market_impact_assessment="No articles.",
            confidence_level="low",
        )

        assert output.articles_analyzed == 0
        assert output.notable_articles == []
