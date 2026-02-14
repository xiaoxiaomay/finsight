"""Tests for Peer Comparison Agent with mocked LLM."""

import json
from datetime import date
from unittest.mock import MagicMock

MOCK_TARGET_DATA = {
    "pe_ratio": 30.5,
    "pb_ratio": 45.2,
    "ev_ebitda": 24.1,
    "roe": 0.160,
    "gross_margin": 0.466,
    "operating_margin": 0.305,
    "revenue_growth_yoy": 0.08,
}

MOCK_PEERS_DATA = [
    {
        "symbol": "MSFT",
        "pe_ratio": 35.2,
        "pb_ratio": 12.8,
        "ev_ebitda": 27.3,
        "roe": 0.39,
        "gross_margin": 0.695,
        "operating_margin": 0.448,
        "revenue_growth_yoy": 0.16,
    },
    {
        "symbol": "GOOGL",
        "pe_ratio": 24.1,
        "pb_ratio": 6.5,
        "ev_ebitda": 16.8,
        "roe": 0.28,
        "gross_margin": 0.574,
        "operating_margin": 0.302,
        "revenue_growth_yoy": 0.11,
    },
]

MOCK_PEER_RESPONSE = json.dumps({
    "valuation_summary": "Trades at discount on P/E but premium on P/B.",
    "competitive_advantages": ["Strong brand", "Ecosystem lock-in"],
    "competitive_weaknesses": ["Lower growth"],
    "peer_comparison_summary": "Well-positioned but trailing on growth.",
    "confidence_level": "high",
})


def _make_mock_client(response_text: str) -> MagicMock:
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text=response_text)]
    mock_client.messages.create.return_value = mock_response
    return mock_client


class TestPeerComparisonAgent:
    """Test peer comparison agent with mocked LLM."""

    def test_analyze_returns_structured_output(self) -> None:
        from src.agents.peer_comparison_agent import PeerComparisonAgent
        from src.agents.schemas.peer_comparison import PeerComparisonOutput

        mock_client = _make_mock_client(MOCK_PEER_RESPONSE)
        agent = PeerComparisonAgent(client=mock_client)

        result = agent.analyze(
            symbol="AAPL",
            sector="Technology",
            target_data=MOCK_TARGET_DATA,
            peers_data=MOCK_PEERS_DATA,
            analysis_date=date(2024, 6, 15),
        )

        assert isinstance(result, PeerComparisonOutput)
        assert result.symbol == "AAPL"
        assert result.sector == "Technology"

    def test_peer_count(self) -> None:
        from src.agents.peer_comparison_agent import PeerComparisonAgent

        mock_client = _make_mock_client(MOCK_PEER_RESPONSE)
        agent = PeerComparisonAgent(client=mock_client)

        result = agent.analyze(
            symbol="AAPL",
            sector="Technology",
            target_data=MOCK_TARGET_DATA,
            peers_data=MOCK_PEERS_DATA,
        )

        assert result.peer_count == 2
        assert len(result.peers) == 2

    def test_relative_valuation(self) -> None:
        from src.agents.peer_comparison_agent import PeerComparisonAgent

        mock_client = _make_mock_client(MOCK_PEER_RESPONSE)
        agent = PeerComparisonAgent(client=mock_client)

        result = agent.analyze(
            symbol="AAPL",
            sector="Technology",
            target_data=MOCK_TARGET_DATA,
            peers_data=MOCK_PEERS_DATA,
        )

        assert result.pe_vs_peers in ("premium", "in_line", "discount")
        assert result.profitability_rank >= 1
        assert result.growth_rank >= 1

    def test_empty_target_data(self) -> None:
        from src.agents.peer_comparison_agent import PeerComparisonAgent
        from src.agents.schemas.peer_comparison import PeerComparisonOutput

        mock_client = _make_mock_client("{}")
        agent = PeerComparisonAgent(client=mock_client)

        result = agent.analyze(
            symbol="AAPL",
            target_data=None,
            analysis_date=date(2024, 6, 15),
        )

        assert isinstance(result, PeerComparisonOutput)
        assert result.confidence_level == "low"

    def test_sector_peers_lookup(self) -> None:
        from src.agents.peer_comparison_agent import PeerComparisonAgent

        peers = PeerComparisonAgent.get_sector_peers("AAPL", "Technology")

        assert len(peers) > 0
        assert "AAPL" not in peers
        assert len(peers) <= 8

    def test_competitive_analysis_extracted(self) -> None:
        from src.agents.peer_comparison_agent import PeerComparisonAgent

        mock_client = _make_mock_client(MOCK_PEER_RESPONSE)
        agent = PeerComparisonAgent(client=mock_client)

        result = agent.analyze(
            symbol="AAPL",
            sector="Technology",
            target_data=MOCK_TARGET_DATA,
            peers_data=MOCK_PEERS_DATA,
        )

        assert len(result.competitive_advantages) > 0
        assert result.peer_comparison_summary != ""
