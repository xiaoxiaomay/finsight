"""Tests for Result Aggregator and Quality Checker."""


MOCK_AGENT_RESULTS = {
    "earnings": {
        "symbol": "AAPL",
        "fundamental_score": 82.5,
        "investment_thesis": "Strong earnings with expanding margins.",
        "key_risks": ["China revenue decline"],
        "catalysts": ["AI product launch"],
        "sources_used": ["10-K MD&A"],
    },
    "quant_signal": {
        "composite_percentile": 76.0,
        "signal_direction": "long",
        "strongest_factors": ["roe", "momentum_12_1"],
        "weakest_factors": ["book_to_market"],
    },
    "news": {
        "overall_sentiment": 0.35,
        "sentiment_label": "positive",
        "key_themes": ["Strong earnings", "AI strategy"],
    },
    "macro": {
        "equity_outlook": "bullish",
        "suggested_equity_allocation": 0.65,
    },
    "peer": {
        "overall_rank": 2,
        "peer_count": 5,
    },
}


class TestAggregator:
    """Test result aggregation logic."""

    def test_aggregate_produces_report(self) -> None:
        from src.agents.aggregator import aggregate_results

        report = aggregate_results(symbol="AAPL", agent_results=MOCK_AGENT_RESULTS)

        assert report["symbol"] == "AAPL"
        assert "recommendation" in report
        assert "composite_score" in report

    def test_recommendation_generated(self) -> None:
        from src.agents.aggregator import aggregate_results

        report = aggregate_results(symbol="AAPL", agent_results=MOCK_AGENT_RESULTS)

        assert report["recommendation"] in (
            "Strong Buy", "Buy", "Hold", "Sell", "Strong Sell",
        )

    def test_composite_score_range(self) -> None:
        from src.agents.aggregator import aggregate_results

        report = aggregate_results(symbol="AAPL", agent_results=MOCK_AGENT_RESULTS)

        assert 0 <= report["composite_score"] <= 100

    def test_agents_used_tracked(self) -> None:
        from src.agents.aggregator import aggregate_results

        report = aggregate_results(symbol="AAPL", agent_results=MOCK_AGENT_RESULTS)

        assert len(report["agents_used"]) == 5
        assert "earnings" in report["agents_used"]

    def test_data_sources_collected(self) -> None:
        from src.agents.aggregator import aggregate_results

        report = aggregate_results(symbol="AAPL", agent_results=MOCK_AGENT_RESULTS)

        assert len(report["data_sources"]) > 0

    def test_bull_case_generated(self) -> None:
        from src.agents.aggregator import aggregate_results

        report = aggregate_results(symbol="AAPL", agent_results=MOCK_AGENT_RESULTS)

        assert len(report["bull_case"]) > 0

    def test_conflict_detection(self) -> None:
        from src.agents.aggregator import aggregate_results

        conflicting = {
            "earnings": {"fundamental_score": 85.0, "investment_thesis": "Strong"},
            "news": {"overall_sentiment": -0.5, "sentiment_label": "negative"},
        }
        report = aggregate_results(symbol="AAPL", agent_results=conflicting)

        assert len(report["conflicts"]) > 0
        assert report["conflicts"][0]["agents"] == ["earnings", "news"]

    def test_no_conflicts_when_aligned(self) -> None:
        from src.agents.aggregator import aggregate_results

        aligned = {
            "earnings": {"fundamental_score": 80.0},
            "news": {"overall_sentiment": 0.5},
        }
        report = aggregate_results(symbol="AAPL", agent_results=aligned)

        assert len(report["conflicts"]) == 0

    def test_conviction_assessment(self) -> None:
        from src.agents.aggregator import aggregate_results

        report = aggregate_results(symbol="AAPL", agent_results=MOCK_AGENT_RESULTS)

        assert report["conviction"] in ("high", "medium", "low")

    def test_empty_results(self) -> None:
        from src.agents.aggregator import aggregate_results

        report = aggregate_results(symbol="AAPL", agent_results={})

        assert report["composite_score"] == 50.0
        assert report["recommendation"] == "Hold"


class TestScoreExtraction:
    """Test score extraction from agent results."""

    def test_earnings_score_extracted(self) -> None:
        from src.agents.aggregator import _extract_scores

        scores = _extract_scores(MOCK_AGENT_RESULTS)
        assert scores["earnings"] == 82.5

    def test_sentiment_mapped_to_0_100(self) -> None:
        from src.agents.aggregator import _extract_scores

        scores = _extract_scores(MOCK_AGENT_RESULTS)
        # 0.35 -> (0.35 + 1) * 50 = 67.5
        assert 67 <= scores["news"] <= 68

    def test_macro_mapped_to_0_100(self) -> None:
        from src.agents.aggregator import _extract_scores

        scores = _extract_scores(MOCK_AGENT_RESULTS)
        # 0.65 * 100 = 65.0
        assert scores["macro"] == 65.0
