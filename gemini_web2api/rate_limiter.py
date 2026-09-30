"""Rate limiting system with optional random jitter."""
import random
import threading
import time
from typing import Optional

from .config import CONFIG


def calculate_interval(rate: float, jitter: bool = False) -> float:
    """Calculate the required interval in seconds for the given rate (requests/sec).

    If rate <= 0, returns 0.0.
    If jitter is True, applies a random deviation within +/- 20% ([0.8, 1.2]).
    """
    if not rate or rate <= 0:
        return 0.0
    base_interval = 1.0 / rate
    if jitter:
        return base_interval * random.uniform(0.8, 1.2)
    return base_interval


class RateLimiter:
    """Thread-safe rate limiter supporting delay-based pacing and random jitter."""

    def __init__(self):
        self._lock = threading.Lock()
        self._next_allowed_time = 0.0

    def reset(self):
        """Reset the rate limiter state."""
        with self._lock:
            self._next_allowed_time = 0.0

    def acquire(self, rate: Optional[float] = None, jitter: Optional[bool] = None) -> float:
        """Wait if necessary to comply with the rate limit.

        Returns the duration waited in seconds (0.0 if no wait was needed).
        """
        if rate is None:
            raw_rate = CONFIG.get("rate_limit")
            if raw_rate is None:
                return 0.0
            try:
                rate = float(raw_rate)
            except (ValueError, TypeError):
                return 0.0

        if rate <= 0:
            return 0.0

        if jitter is None:
            jitter = bool(CONFIG.get("rate_limit_jitter", False))

        interval = calculate_interval(rate, jitter)

        with self._lock:
            now = time.monotonic()
            target_time = max(now, self._next_allowed_time)
            self._next_allowed_time = target_time + interval
            wait_time = target_time - now

        if wait_time > 0:
            time.sleep(wait_time)
            return wait_time
        return 0.0


rate_limiter = RateLimiter()
