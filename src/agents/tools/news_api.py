"""NewsAPI search tool for agents.

Fetches recent news articles for a company or topic using NewsAPI.org.
Provides article summaries and metadata for sentiment analysis.
"""

from __future__ import annotations

from datetime import date, timedelta

import httpx

from src.config.logging_config import get_logger
from src.config.settings import get_settings

logger = get_logger("tools.news_api")

NEWSAPI_BASE = "https://newsapi.org/v2"


def search_news(
    query: str,
    days_back: int = 30,
    page_size: int = 20,
    language: str = "en",
    sort_by: str = "relevancy",
) -> list[dict]:
    """Search for recent news articles using NewsAPI.

    Args:
        query: Search query (e.g., 'Apple AAPL earnings').
        days_back: Number of days to look back.
        page_size: Maximum articles to return.
        language: Language filter.
        sort_by: Sort order ('relevancy', 'publishedAt', 'popularity').

    Returns:
        List of article dicts with title, source, publishedAt, url, description.
    """
    settings = get_settings()
    if not settings.news_api_key:
        logger.warning("no_news_api_key")
        return []

    from_date = date.today() - timedelta(days=days_back)

    params = {
        "q": query,
        "from": str(from_date),
        "language": language,
        "sortBy": sort_by,
        "pageSize": min(page_size, 100),
        "apiKey": settings.news_api_key,
    }

    try:
        resp = httpx.get(
            f"{NEWSAPI_BASE}/everything",
            params=params,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        logger.warning("news_fetch_failed", query=query)
        return []

    articles = data.get("articles", [])
    results = []

    for article in articles:
        results.append({
            "title": article.get("title", ""),
            "source": article.get("source", {}).get("name", ""),
            "published_at": article.get("publishedAt", ""),
            "url": article.get("url", ""),
            "description": article.get("description", "") or "",
            "content_snippet": (article.get("content", "") or "")[:500],
        })

    logger.info("news_search_complete", query=query, results=len(results))
    return results


def search_company_news(
    symbol: str,
    company_name: str = "",
    days_back: int = 30,
    page_size: int = 20,
) -> list[dict]:
    """Search for news about a specific company.

    Builds an optimized query using ticker symbol and company name.
    """
    # Build query: combine ticker and company name for better results
    parts = [symbol]
    if company_name:
        parts.append(f'"{company_name}"')
    query = " OR ".join(parts)

    return search_news(
        query=query,
        days_back=days_back,
        page_size=page_size,
    )


def get_top_headlines(
    category: str = "business",
    country: str = "us",
    page_size: int = 10,
) -> list[dict]:
    """Get top headlines for a category.

    Args:
        category: News category ('business', 'technology', etc.).
        country: Country code.
        page_size: Maximum articles to return.

    Returns:
        List of article dicts.
    """
    settings = get_settings()
    if not settings.news_api_key:
        logger.warning("no_news_api_key")
        return []

    params = {
        "category": category,
        "country": country,
        "pageSize": min(page_size, 100),
        "apiKey": settings.news_api_key,
    }

    try:
        resp = httpx.get(
            f"{NEWSAPI_BASE}/top-headlines",
            params=params,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        logger.warning("headlines_fetch_failed", category=category)
        return []

    return [
        {
            "title": a.get("title", ""),
            "source": a.get("source", {}).get("name", ""),
            "published_at": a.get("publishedAt", ""),
            "url": a.get("url", ""),
            "description": a.get("description", "") or "",
        }
        for a in data.get("articles", [])
    ]
