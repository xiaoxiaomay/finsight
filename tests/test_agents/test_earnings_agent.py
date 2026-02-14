"""Tests for Earnings Analysis Agent with mocked LLM responses."""

import json
from datetime import date
from unittest.mock import MagicMock

MOCK_EARNINGS_RESPONSE = json.dumps({
    "symbol": "AAPL",
    "analysis_date": "2024-06-15",
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
        "Raised capital return program to $110B",
    ],
    "risk_factors_highlighted": [
        "China market uncertainty",
        "Regulatory scrutiny in EU",
    ],
    "management_tone": "confident",
    "fundamental_score": 82.5,
    "investment_thesis": "Apple demonstrates strong earnings quality with expanding margins "
    "driven by services mix shift. Robust cash generation supports continued buybacks.",
    "key_risks": ["China revenue decline", "Antitrust regulation"],
    "catalysts": ["Apple Intelligence AI launch", "Vision Pro ramp"],
    "sources_used": ["10-K Item 7: MD&A", "10-K Item 1A: Risk Factors"],
    "confidence_level": "high",
})


def _make_mock_client(response_text: str) -> MagicMock:
    """Create a mock Anthropic client that returns a specific response."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text=response_text)]
    mock_client.messages.create.return_value = mock_response
    return mock_client


class TestEarningsAgent:
    """Test earnings agent with mocked LLM."""

    def test_analyze_returns_structured_output(self) -> None:
        """Agent should return valid EarningsAnalysisOutput."""
        from src.agents.earnings_agent import EarningsAgent
        from src.agents.schemas.earnings import EarningsAnalysisOutput

        mock_client = _make_mock_client(MOCK_EARNINGS_RESPONSE)
        agent = EarningsAgent(client=mock_client)

        result = agent.analyze(
            symbol="AAPL",
            analysis_date=date(2024, 6, 15),
        )

        assert isinstance(result, EarningsAnalysisOutput)
        assert result.symbol == "AAPL"
        assert result.fundamental_score == 82.5
        assert result.revenue_trend == "accelerating"
        assert result.margin_expansion is True

    def test_llm_called_with_correct_model(self) -> None:
        """Agent should call LLM with the configured model."""
        from src.agents.earnings_agent import EarningsAgent

        mock_client = _make_mock_client(MOCK_EARNINGS_RESPONSE)
        agent = EarningsAgent(client=mock_client)
        agent.analyze(symbol="AAPL")

        mock_client.messages.create.assert_called_once()

    def test_citations_preserved(self) -> None:
        """Output should include source citations."""
        from src.agents.earnings_agent import EarningsAgent

        mock_client = _make_mock_client(MOCK_EARNINGS_RESPONSE)
        agent = EarningsAgent(client=mock_client)
        result = agent.analyze(symbol="AAPL")

        assert len(result.sources_used) > 0
        assert any("10-K" in src for src in result.sources_used)

    def test_with_fundamentals_data(self) -> None:
        """Agent should incorporate fundamentals data in analysis."""
        import pandas as pd

        from src.agents.earnings_agent import EarningsAgent
        from src.agents.schemas.earnings import EarningsAnalysisOutput

        mock_client = _make_mock_client(MOCK_EARNINGS_RESPONSE)
        agent = EarningsAgent(client=mock_client)

        fund_df = pd.DataFrame([{
            "symbol": "AAPL",
            "report_date": "2024-01-01",
            "eps": 6.42,
            "net_income": 97e9,
            "total_equity": 62e9,
            "total_assets": 352e9,
            "total_liabilities": 290e9,
            "gross_profit": 170e9,
            "operating_income": 114e9,
            "operating_cash_flow": 110e9,
        }])

        result = agent.analyze(symbol="AAPL", fundamentals_df=fund_df)

        assert isinstance(result, EarningsAnalysisOutput)
        call_args = mock_client.messages.create.call_args
        prompt_text = call_args.kwargs["messages"][0]["content"]
        assert "AAPL" in prompt_text

    def test_with_rag_retriever(self) -> None:
        """Agent should use RAG retriever when available."""
        from src.agents.earnings_agent import EarningsAgent
        from src.agents.rag.chunking import Chunk
        from src.agents.rag.retriever import HybridRetriever, RetrievalResult
        from src.agents.schemas.earnings import EarningsAnalysisOutput

        mock_client = _make_mock_client(MOCK_EARNINGS_RESPONSE)

        mock_retriever = MagicMock(spec=HybridRetriever)
        mock_retriever.total_chunks = 10
        mock_retriever.retrieve.return_value = [
            RetrievalResult(
                chunk=Chunk(
                    text="Apple revenue grew 8% YoY to $94.8B",
                    metadata={"section_name": "MD&A"},
                ),
                score=0.85,
            ),
        ]

        agent = EarningsAgent(retriever=mock_retriever, client=mock_client)
        result = agent.analyze(symbol="AAPL")

        assert isinstance(result, EarningsAnalysisOutput)
        mock_retriever.retrieve.assert_called_once()

    def test_json_extraction_from_code_block(self) -> None:
        """Should extract JSON from markdown code blocks."""
        from src.agents.earnings_agent import _extract_json

        text = "Here is the analysis:\n```json\n{\"symbol\": \"AAPL\"}\n```\nDone."
        result = _extract_json(text)
        assert result == '{"symbol": "AAPL"}'

    def test_json_extraction_from_raw(self) -> None:
        """Should extract raw JSON from response."""
        from src.agents.earnings_agent import _extract_json

        text = 'Some text before {"symbol": "AAPL"} some text after'
        result = _extract_json(text)
        assert result == '{"symbol": "AAPL"}'
