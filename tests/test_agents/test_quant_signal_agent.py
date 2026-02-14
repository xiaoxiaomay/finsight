"""Tests for Quant Signal Agent with mocked LLM."""

import json
from datetime import date
from unittest.mock import MagicMock

MOCK_FACTOR_DATA = {
    "momentum_12_1": {"raw_value": 0.25, "z_score": 1.2, "percentile": 85.0},
    "earnings_yield": {"raw_value": 0.05, "z_score": 0.8, "percentile": 72.0},
    "roe": {"raw_value": 0.16, "z_score": 1.5, "percentile": 90.0},
    "gross_profitability": {"raw_value": 0.55, "z_score": 1.1, "percentile": 82.0},
    "volatility_60d": {"raw_value": 0.22, "z_score": -0.3, "percentile": 42.0},
    "book_to_market": {"raw_value": 0.03, "z_score": -0.5, "percentile": 35.0},
}

MOCK_QUANT_RESPONSE = json.dumps({
    "confidence_level": "high",
    "signal_direction": "long",
    "signal_strength": 0.65,
})


def _make_mock_client(response_text: str) -> MagicMock:
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text=response_text)]
    mock_client.messages.create.return_value = mock_response
    return mock_client


class TestQuantSignalAgent:
    """Test quant signal agent with mocked LLM."""

    def test_analyze_returns_structured_output(self) -> None:
        from src.agents.quant_signal_agent import QuantSignalAgent
        from src.agents.schemas.quant_signal import QuantSignalOutput

        mock_client = _make_mock_client(MOCK_QUANT_RESPONSE)
        agent = QuantSignalAgent(client=mock_client)

        result = agent.analyze(
            symbol="AAPL",
            factor_data=MOCK_FACTOR_DATA,
            sector="Technology",
            analysis_date=date(2024, 6, 15),
        )

        assert isinstance(result, QuantSignalOutput)
        assert result.symbol == "AAPL"
        assert result.sector == "Technology"
        assert result.confidence_level == "high"

    def test_composite_score_computed(self) -> None:
        from src.agents.quant_signal_agent import QuantSignalAgent

        mock_client = _make_mock_client(MOCK_QUANT_RESPONSE)
        agent = QuantSignalAgent(client=mock_client)

        result = agent.analyze(
            symbol="AAPL",
            factor_data=MOCK_FACTOR_DATA,
            analysis_date=date(2024, 6, 15),
        )

        assert result.composite_score != 0.0
        assert 0 <= result.composite_percentile <= 100

    def test_factor_scores_populated(self) -> None:
        from src.agents.quant_signal_agent import QuantSignalAgent

        mock_client = _make_mock_client(MOCK_QUANT_RESPONSE)
        agent = QuantSignalAgent(client=mock_client)

        result = agent.analyze(symbol="AAPL", factor_data=MOCK_FACTOR_DATA)

        assert len(result.factor_scores) == 6
        assert all(f.factor_name for f in result.factor_scores)

    def test_strongest_weakest_factors(self) -> None:
        from src.agents.quant_signal_agent import QuantSignalAgent

        mock_client = _make_mock_client(MOCK_QUANT_RESPONSE)
        agent = QuantSignalAgent(client=mock_client)

        result = agent.analyze(symbol="AAPL", factor_data=MOCK_FACTOR_DATA)

        assert len(result.strongest_factors) == 3
        assert len(result.weakest_factors) == 3
        assert "roe" in result.strongest_factors

    def test_empty_factor_data(self) -> None:
        from src.agents.quant_signal_agent import QuantSignalAgent
        from src.agents.schemas.quant_signal import QuantSignalOutput

        mock_client = _make_mock_client("{}")
        agent = QuantSignalAgent(client=mock_client)

        result = agent.analyze(
            symbol="AAPL",
            factor_data=None,
            analysis_date=date(2024, 6, 15),
        )

        assert isinstance(result, QuantSignalOutput)
        assert result.signal_direction == "neutral"
        assert result.confidence_level == "low"

    def test_signal_direction_from_percentile(self) -> None:
        from src.agents.quant_signal_agent import QuantSignalAgent

        assert QuantSignalAgent._determine_signal(90)["direction"] == "strong_long"
        assert QuantSignalAgent._determine_signal(70)["direction"] == "long"
        assert QuantSignalAgent._determine_signal(50)["direction"] == "neutral"
        assert QuantSignalAgent._determine_signal(30)["direction"] == "short"
        assert QuantSignalAgent._determine_signal(10)["direction"] == "strong_short"

    def test_with_backtest_stats(self) -> None:
        from src.agents.quant_signal_agent import QuantSignalAgent

        mock_client = _make_mock_client(MOCK_QUANT_RESPONSE)
        agent = QuantSignalAgent(client=mock_client)

        bt_stats = {
            "total_return": 0.35,
            "annualized_return": 0.12,
            "sharpe_ratio": 1.45,
            "max_drawdown": -0.15,
        }

        result = agent.analyze(
            symbol="AAPL",
            factor_data=MOCK_FACTOR_DATA,
            backtest_stats=bt_stats,
        )

        assert result.backtest_stats is not None
        assert result.backtest_stats.sharpe_ratio == 1.45
