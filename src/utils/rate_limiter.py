"""Rate limiting utilities for external API calls.

Provides both sync and async rate limiters to stay within
API rate limits (yfinance, FMP, FRED, NewsAPI).
"""

import asyncio
import time
from collections import deque
from threading import Lock


class RateLimiter:
    """Thread-safe rate limiter using sliding window.

    Args:
        calls_per_second: Maximum calls per second.
    """

    def __init__(self, calls_per_second: float) -> None:
        self.min_interval = 1.0 / calls_per_second
        self._last_call = 0.0
        self._lock = Lock()

    def wait(self) -> None:
        """Block until a call is allowed."""
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_call
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            self._last_call = time.monotonic()

    async def async_wait(self) -> None:
        """Async version: yield control while waiting."""
        now = time.monotonic()
        elapsed = now - self._last_call
        if elapsed < self.min_interval:
            await asyncio.sleep(self.min_interval - elapsed)
        self._last_call = time.monotonic()


class DailyQuotaLimiter:
    """Track daily API call quota.

    Args:
        daily_limit: Maximum calls per day.
    """

    def __init__(self, daily_limit: int) -> None:
        self.daily_limit = daily_limit
        self._calls: deque[float] = deque()
        self._lock = Lock()

    @property
    def remaining(self) -> int:
        """Number of calls remaining today."""
        self._prune()
        return max(0, self.daily_limit - len(self._calls))

    def _prune(self) -> None:
        """Remove calls older than 24 hours."""
        cutoff = time.time() - 86400
        while self._calls and self._calls[0] < cutoff:
            self._calls.popleft()

    def acquire(self) -> bool:
        """Try to acquire a call slot. Returns False if quota exhausted."""
        with self._lock:
            self._prune()
            if len(self._calls) >= self.daily_limit:
                return False
            self._calls.append(time.time())
            return True

    def check(self, needed: int = 1) -> bool:
        """Check if enough quota is available without consuming it."""
        self._prune()
        return self.remaining >= needed
