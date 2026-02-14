"""Prompt templates for News Sentiment Agent."""

NEWS_SYSTEM_PROMPT = """You are an expert financial news analyst. You analyze recent news \
articles about companies and industries to assess market sentiment and potential impact on \
stock prices.

Your analysis must be:
1. Objective — separate fact from opinion
2. Quantitative — assign specific sentiment scores to each article
3. Thematic — identify recurring themes across articles
4. Impact-focused — assess how news may affect stock price and fundamentals"""

NEWS_SENTIMENT_PROMPT = """Analyze the following recent news articles about {symbol} and \
assess the overall news sentiment.

## Recent News Articles
{articles_text}

## Instructions
For each article, score the sentiment from -1.0 (very negative) to +1.0 (very positive) \
and assess relevance to {symbol} from 0.0 (irrelevant) to 1.0 (highly relevant).

Then provide an overall sentiment assessment:

1. **Overall Sentiment**: Weighted average of individual article sentiments (weight by relevance).

2. **Sentiment Label**: Classify as very_negative (<-0.5), negative (-0.5 to -0.1), \
neutral (-0.1 to 0.1), positive (0.1 to 0.5), or very_positive (>0.5).

3. **Key Themes**: Identify 3-5 recurring themes across the articles.

4. **Notable Articles**: Highlight the 3-5 most impactful articles with their individual \
sentiment and relevance scores.

5. **Sentiment Trend**: Is sentiment improving, stable, or deteriorating compared to \
earlier articles?

6. **Market Impact**: Briefly assess the potential impact on {symbol}'s stock price.

Respond with a JSON object matching this exact schema:
{{
    "symbol": "{symbol}",
    "analysis_date": "{analysis_date}",
    "articles_analyzed": <int>,
    "overall_sentiment": <float -1.0 to 1.0>,
    "sentiment_label": "very_negative|negative|neutral|positive|very_positive",
    "key_themes": [<string>, ...],
    "notable_articles": [
        {{
            "title": "<string>",
            "source": "<string>",
            "published_at": "<string>",
            "url": "<string>",
            "summary": "<string>",
            "sentiment_score": <float -1.0 to 1.0>,
            "relevance_score": <float 0.0 to 1.0>
        }},
        ...
    ],
    "sentiment_trend": "improving|stable|deteriorating",
    "market_impact_assessment": "<string>",
    "confidence_level": "high|medium|low"
}}"""
