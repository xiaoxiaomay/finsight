"""Pydantic schemas for News Sentiment Agent output."""

from datetime import date

from pydantic import BaseModel, Field


class NewsArticle(BaseModel):
    """A single news article with sentiment scoring."""

    title: str
    source: str
    published_at: str
    url: str = ""
    summary: str = ""
    sentiment_score: float = Field(
        ge=-1.0, le=1.0, description="Sentiment from -1 (very negative) to +1 (very positive)"
    )
    relevance_score: float = Field(
        ge=0.0, le=1.0, description="Relevance to the queried symbol"
    )


class NewsSentimentOutput(BaseModel):
    """Structured output from news sentiment analysis."""

    symbol: str
    analysis_date: date

    # Aggregated metrics
    articles_analyzed: int = Field(ge=0)
    overall_sentiment: float = Field(
        ge=-1.0, le=1.0, description="Average sentiment across all articles"
    )
    sentiment_label: str = Field(
        description="One of: very_negative, negative, neutral, positive, very_positive"
    )

    # Themes and articles
    key_themes: list[str] = Field(default_factory=list)
    notable_articles: list[NewsArticle] = Field(default_factory=list)

    # Trend and impact
    sentiment_trend: str = Field(
        description="One of: improving, stable, deteriorating"
    )
    market_impact_assessment: str = Field(
        description="Brief assessment of potential market impact"
    )

    # Meta
    confidence_level: str = Field(
        description="One of: high, medium, low"
    )
