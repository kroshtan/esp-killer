"""
Per-key token-bucket rate limiting, in process memory.

Correct for the single API process this service runs as. With several processes each would enforce its own
limit; move the buckets to Redis or the database before scaling out.
"""

import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class _Bucket:
    tokens: float
    updated: float


class TokenBucketLimiter:
    def __init__(self, rate_per_s: float, burst: int, clock: Callable[[], float] = time.monotonic) -> None:
        self.rate = rate_per_s
        self.burst = burst
        self.clock = clock
        self._buckets: dict[str, _Bucket] = {}
        self._lock = threading.Lock()

    def acquire(self, key: str) -> float:
        """
        Take one token for ``key``.

        :param key: the bucket, e.g. an API key hash
        :return: 0.0 if allowed, otherwise the seconds until a token is available (for ``Retry-After``)
        """
        now = self.clock()
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = self._buckets[key] = _Bucket(tokens=float(self.burst), updated=now)
            bucket.tokens = min(float(self.burst), bucket.tokens + (now - bucket.updated) * self.rate)
            bucket.updated = now
            if bucket.tokens >= 1.0:
                bucket.tokens -= 1.0
                return 0.0
            return (1.0 - bucket.tokens) / self.rate

    @staticmethod
    def retry_after_header(wait_s: float) -> str:
        """
        Format a wait as a ``Retry-After`` value (whole seconds, rounded up).

        :param wait_s: seconds to wait
        :return: the header value
        """
        return str(max(1, math.ceil(wait_s)))
