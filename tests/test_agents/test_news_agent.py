"""Tests for News Sentiment Agent with mocked LLM and NewsAPI."""

import json
from datetime import date
from unittest.mock import MagicMock

MOCK_ARTICLES = [
    {
        "title": "Apple Reports Record Q4 Revenue of $89.5 Billion",
        "source": "Reuters",
        "published_at": "2024-11-01T18:30:00Z",
        "url": "https://reuters.com/apple-q4",
        "description": "Apple exceeded analyst expectations with record revenue.",
        "content_snippet": "Apple Inc reported fourth-quarter revenue of $89.5 billion...",
    },
    {
        "title": "Apple Faces Antitrust Probe in European Union",
        "source": "Financial Times",
        "published_at": "2024-10-28T10:00:00Z",
        "url": "https://ft.com/apple-eu",
        "description": "EU regulators opened new investigation into Apple's App Store.",
        "content_snippet": "The European Commission has launched a fresh probe...",
    },
    {
        "title": "Apple Intelligence: Analysts See Long-Term Revenue Opportunity",
        "source": "Bloomberg",
        "published_at": "2024-10-25T14:15:00Z",
        "url": "https://bloomberg.com/apple-ai",
        "description": "Wall Street analysts project AI features could drive upgrade cycle.",
        "content_snippet": "Multiple analysts raised price targets citing Apple Intelligence...",
    },
]

MOCK_NEWS_RESPONSE = json.dumps({
    "symbol": "AAPL",
    "analysis_date": "2024-11-05",
    "articles_analyzed": 3,
    "overall_sentiment": 0.35,
    "sentiment_label": "positive",
    "key_themes": [
        "Strong quarterly earnings",
        "EU regulatory headwinds",
        "AI product strategy",
    ],
    "notable_articles": [
        {
            "title": "Apple Reports Record Q4 Revenue of $89.5 Billion",
            "source": "Reuters",
            "published_at": "2024-11-01T18:30:00Z",
            "url": "https://reuters.com/apple-q4",
            "summary": "Record revenue exceeding expectations.",
            "sentiment_score": 0.8,
            "relevance_score": 0.95,
        },
        {
            "title": "Apple Faces Antitrust Probe in European Union",
            "source": "Financial Times",
            "published_at": "2024-10-28T10:00:00Z",
            "url": "https://ft.com/apple-eu",
            "summary": "New EU regulatory investigation.",
            "sentiment_score": -0.4,
            "relevance_score": 0.85,
        },
        {
            "title": "Apple Intelligence: Analysts See Long-Term Revenue Opportunity",
            "source": "Bloomberg",
            "published_at": "2024-10-25T14:15:00Z",
            "url": "https://bloomberg.com/apple-ai",
            "summary": "Positive analyst outlook on AI strategy.",
            "sentiment_score": 0.6,
            "relevance_score": 0.90,
        },
    ],
    "sentiment_trend": "improving",
    "market_impact_assessment": "Net positive sentiment driven by strong earnings, "
    "partially offset by regulatory concerns. AI strategy viewed favorably by market.",
    "confidence_level": "high",
})


def _make_mock_client(response_text: str) -> MagicMock:
    """Create a mock Anthropic client."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text=response_text)]
    mock_client.messages.create.return_value = mock_response
    return mock_client


class TestNewsAgent:
    """Test news agent with mocked LLM."""

    def test_analyze_returns_structured_output(self) -> None:
        """Agent should return valid NewsSentimentOutput."""
        from src.agents.news_agent import NewsAgent
        from src.agents.schemas.news import NewsSentimentOutput

        mock_client = _make_mock_client(MOCK_NEWS_RESPONSE)
        agent = NewsAgent(client=mock_client)

        result = agent.analyze(
            symbol="AAPL",
            analysis_date=date(2024, 11, 5),
            articles=MOCK_ARTICLES,
        )

        assert isinstance(result, NewsSentimentOutput)
        assert result.symbol == "AAPL"
        assert result.overall_sentiment == 0.35
        assert result.sentiment_label == "positive"

    def test_articles_scored_individually(self) -> None:
        """Each notable article should have a sentiment score."""
        from src.agents.news_agent import NewsAgent

        mock_client = _make_mock_client(MOCK_NEWS_RESPONSE)
        agent = NewsAgent(client=mock_client)

        result = agent.analyze(
            symbol="AAPL",
            articles=MOCK_ARTICLES,
        )

        assert len(result.notable_articles) == 3
        for article in result.notable_articles:
            assert -1.0 <= article.sentiment_score <= 1.0
            assert 0.0 <= article.relevance_score <= 1.0

    def test_key_themes_extracted(self) -> None:
        """Should extract key themes from articles."""
        from src.agents.news_agent import NewsAgent

        mock_client = _make_mock_client(MOCK_NEWS_RESPONSE)
        agent = NewsAgent(client=mock_client)

        result = agent.analyze(symbol="AAPL", articles=MOCK_ARTICLES)

        assert len(result.key_themes) >= 2

    def test_empty_articles_returns_neutral(self) -> None:
        """No articles should return neutral sentiment."""
        from src.agents.news_agent import NewsAgent
        from src.agents.schemas.news import NewsSentimentOutput

        mock_client = _make_mock_client("{}")
        agent = NewsAgent(client=mock_client)

        result = agent.analyze(
            symbol="AAPL",
            articles=[],
            analysis_date=date(2024, 11, 5),
        )

        assert isinstance(result, NewsSentimentOutput)
        assert result.articles_analyzed == 0
        assert result.overall_sentiment == 0.0
        assert result.confidence_level == "low"

    def test_format_articles(self) -> None:
        """Articles should be formatted for LLM prompt."""
        from src.agents.news_agent import NewsAgent

        text = NewsAgent._format_articles(MOCK_ARTICLES)

        assert "Apple Reports Record" in text
        assert "Reuters" in text
        assert "Article 1" in text
        assert "Article 3" in text

    def test_sentiment_trend_included(self) -> None:
        """Output should include sentiment trend assessment."""
        from src.agents.news_agent import NewsAgent

        mock_client = _make_mock_client(MOCK_NEWS_RESPONSE)
        agent = NewsAgent(client=mock_client)

        result = agent.analyze(symbol="AAPL", articles=MOCK_ARTICLES)

        assert result.sentiment_trend in ("improving", "stable", "deteriorating")
        assert result.market_impact_assessment != ""

    def test_llm_receives_articles_in_prompt(self) -> None:
        """LLM prompt should contain article data."""
        from src.agents.news_agent import NewsAgent

        mock_client = _make_mock_client(MOCK_NEWS_RESPONSE)
        agent = NewsAgent(client=mock_client)

        agent.analyze(symbol="AAPL", articles=MOCK_ARTICLES)

        call_args = mock_client.messages.create.call_args
        prompt = call_args.kwargs["messages"][0]["content"]
        assert "Apple Reports Record" in prompt
        assert "AAPL" in prompt
