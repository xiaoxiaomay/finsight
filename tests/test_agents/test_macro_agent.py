"""Tests for Macro Environment Agent with mocked LLM and FRED data."""

import json
from datetime import date
from unittest.mock import MagicMock

MOCK_MACRO_RESPONSE = json.dumps({
    "assessment_date": "2024-06-15",
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
        "Geopolitical tensions in Middle East",
        "Commercial real estate weakness",
        "Student loan repayment impact on consumer spending",
    ],
    "suggested_equity_allocation": 0.65,
    "defensive_tilt": False,
    "reasoning": "The economy is in a mid-cycle expansion with moderating inflation. "
    "The Fed is likely to begin cutting rates in the coming quarters, "
    "which historically supports equity valuations.",
})

MOCK_MACRO_DATA = {
    "GDPC1": {
        "value": 22225.35,
        "previous": 22012.1,
        "change": 213.25,
        "trend": "rising",
        "description": "Real GDP (quarterly)",
    },
    "CPIAUCSL": {
        "value": 314.175,
        "previous": 313.534,
        "change": 0.641,
        "trend": "rising",
        "description": "CPI All Items (monthly)",
    },
    "DFF": {
        "value": 5.33,
        "previous": 5.33,
        "change": 0.0,
        "trend": "stable",
        "description": "Federal Funds Rate (daily)",
    },
    "DGS10": {
        "value": 4.25,
        "previous": 4.20,
        "change": 0.05,
        "trend": "stable",
        "description": "10-Year Treasury Yield (daily)",
    },
    "UNRATE": {
        "value": 3.7,
        "previous": 3.8,
        "change": -0.1,
        "trend": "falling",
        "description": "Unemployment Rate (monthly)",
    },
    "VIXCLS": {
        "value": 14.2,
        "previous": 15.8,
        "change": -1.6,
        "trend": "falling",
        "description": "VIX Volatility Index (daily)",
    },
}


def _make_mock_client(response_text: str) -> MagicMock:
    """Create a mock Anthropic client."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text=response_text)]
    mock_client.messages.create.return_value = mock_response
    return mock_client


class TestMacroAgent:
    """Test macro agent with mocked LLM and data."""

    def test_assess_from_data_returns_structured_output(self) -> None:
        """Agent should return valid MacroAssessmentOutput."""
        from src.agents.macro_agent import MacroAgent
        from src.agents.schemas.macro import MacroAssessmentOutput

        mock_client = _make_mock_client(MOCK_MACRO_RESPONSE)
        agent = MacroAgent(client=mock_client)

        result = agent.assess_from_data(
            macro_data=MOCK_MACRO_DATA,
            yield_curve_status="normal",
            fed_assessment={"fed_policy_stance": "neutral", "rate_direction": "pausing"},
            assessment_date=date(2024, 6, 15),
        )

        assert isinstance(result, MacroAssessmentOutput)
        assert result.cycle_phase == "expansion"
        assert result.cycle_confidence == 0.72
        assert result.suggested_equity_allocation == 0.65

    def test_sector_preferences_populated(self) -> None:
        """Output should include sector preferences."""
        from src.agents.macro_agent import MacroAgent

        mock_client = _make_mock_client(MOCK_MACRO_RESPONSE)
        agent = MacroAgent(client=mock_client)

        result = agent.assess_from_data(
            macro_data=MOCK_MACRO_DATA,
            assessment_date=date(2024, 6, 15),
        )

        assert len(result.sector_preferences) > 0
        assert "Technology" in result.sector_preferences

    def test_risk_factors_included(self) -> None:
        """Output should include risk factors."""
        from src.agents.macro_agent import MacroAgent

        mock_client = _make_mock_client(MOCK_MACRO_RESPONSE)
        agent = MacroAgent(client=mock_client)

        result = agent.assess_from_data(
            macro_data=MOCK_MACRO_DATA,
            assessment_date=date(2024, 6, 15),
        )

        assert len(result.risk_factors) > 0

    def test_format_macro_data(self) -> None:
        """Should format macro data as readable text."""
        from src.agents.macro_agent import MacroAgent

        text = MacroAgent._format_macro_data(MOCK_MACRO_DATA)

        assert "Real GDP" in text
        assert "Federal Funds Rate" in text
        assert "rising" in text

    def test_llm_receives_macro_data(self) -> None:
        """LLM prompt should contain macro indicator data."""
        from src.agents.macro_agent import MacroAgent

        mock_client = _make_mock_client(MOCK_MACRO_RESPONSE)
        agent = MacroAgent(client=mock_client)

        agent.assess_from_data(
            macro_data=MOCK_MACRO_DATA,
            assessment_date=date(2024, 6, 15),
        )

        call_args = mock_client.messages.create.call_args
        prompt = call_args.kwargs["messages"][0]["content"]
        assert "Real GDP" in prompt
        assert "Federal Funds Rate" in prompt
