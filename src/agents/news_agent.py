"""News Sentiment Agent.

Searches recent news about a company, scores sentiment using Claude,
and produces a structured NewsSentimentOutput.
"""

from __future__ import annotations

import json
from datetime import date

import anthropic

from src.agents.prompts.news_synthesis import (
    NEWS_SENTIMENT_PROMPT,
    NEWS_SYSTEM_PROMPT,
)
from src.agents.schemas.news import NewsSentimentOutput
from src.agents.tools.news_api import search_company_news
from src.config.logging_config import get_logger
from src.config.settings import get_settings

logger = get_logger("agent.news")


class NewsAgent:
    """Agent for news sentiment analysis.

    Uses:
    - NewsAPI tool for article fetching
    - Claude Sonnet for sentiment scoring and synthesis
    """

    def __init__(
        self,
        client: anthropic.Anthropic | None = None,
    ) -> None:
        settings = get_settings()
        self.client = client or anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self.model = settings.llm_model
        self.max_tokens = settings.llm_max_tokens

    def analyze(
        self,
        symbol: str,
        company_name: str = "",
        days_back: int = 30,
        analysis_date: date | None = None,
        articles: list[dict] | None = None,
    ) -> NewsSentimentOutput:
        """Run news sentiment analysis for a company.

        Args:
            symbol: Stock ticker (e.g., 'AAPL').
            company_name: Company name for broader search.
            days_back: Number of days to search back.
            analysis_date: Date of analysis (defaults to today).
            articles: Optional pre-fetched articles (skips API call).

        Returns:
            NewsSentimentOutput Pydantic model.
        """
        if analysis_date is None:
            analysis_date = date.today()

        # Step 1: Fetch articles
        if articles is None:
            articles = search_company_news(
                symbol=symbol,
                company_name=company_name,
                days_back=days_back,
            )

        if not articles:
            logger.warning("no_articles_found", symbol=symbol)
            return self._empty_result(symbol, analysis_date)

        # Step 2: Format articles for LLM
        articles_text = self._format_articles(articles)

        # Step 3: Build prompt
        prompt = NEWS_SENTIMENT_PROMPT.format(
            symbol=symbol,
            analysis_date=str(analysis_date),
            articles_text=articles_text,
        )

        # Step 4: Call Claude
        response_text = self._call_llm(prompt)

        # Step 5: Parse into Pydantic model
        return self._parse_response(response_text, symbol, analysis_date)

    def _call_llm(self, prompt: str) -> str:
        """Call Claude Sonnet and return the response text."""
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=NEWS_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text
        except Exception:
            logger.warning("llm_call_failed", agent="news")
            raise

    def _parse_response(
        self,
        response_text: str,
        symbol: str,
        analysis_date: date,
    ) -> NewsSentimentOutput:
        """Parse LLM response into structured output."""
        json_str = _extract_json(response_text)

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            logger.warning("json_parse_failed", agent="news")
            data = {}

        data.setdefault("symbol", symbol)
        data.setdefault("analysis_date", str(analysis_date))

        return NewsSentimentOutput(**data)

    @staticmethod
    def _format_articles(articles: list[dict]) -> str:
        """Format articles for LLM prompt."""
        parts = []
        for i, article in enumerate(articles[:20], 1):
            parts.append(
                f"### Article {i}\n"
                f"**Title:** {article.get('title', 'N/A')}\n"
                f"**Source:** {article.get('source', 'N/A')}\n"
                f"**Date:** {article.get('published_at', 'N/A')}\n"
                f"**Summary:** {article.get('description', 'N/A')}\n"
                f"**Content:** {article.get('content_snippet', '')[:300]}"
            )
        return "\n\n".join(parts)

    @staticmethod
    def _empty_result(symbol: str, analysis_date: date) -> NewsSentimentOutput:
        """Return empty result when no articles found."""
        return NewsSentimentOutput(
            symbol=symbol,
            analysis_date=analysis_date,
            articles_analyzed=0,
            overall_sentiment=0.0,
            sentiment_label="neutral",
            key_themes=[],
            notable_articles=[],
            sentiment_trend="stable",
            market_impact_assessment="No recent news articles found for analysis.",
            confidence_level="low",
        )


def _extract_json(text: str) -> str:
    """Extract JSON from LLM response."""
    if "```json" in text:
        start = text.index("```json") + 7
        end = text.index("```", start)
        return text[start:end].strip()
    if "```" in text:
        start = text.index("```") + 3
        end = text.index("```", start)
        return text[start:end].strip()
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start >= 0 and brace_end > brace_start:
        return text[brace_start:brace_end + 1]
    return text
