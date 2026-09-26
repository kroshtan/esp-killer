import random


class Backoff:
    """Exponential backoff with full jitter, reset on success."""

    def __init__(self, base_s: float = 1.0, cap_s: float = 300.0, rng: random.Random | None = None) -> None:
        self.base_s = base_s
        self.cap_s = cap_s
        self.failures = 0
        self._rng = rng or random.Random()

    def next_delay(self) -> float:
        """
        Record a failure and return how long to wait before retrying.

        :return: seconds to wait
        """
        self.failures += 1
        ceiling = min(self.cap_s, self.base_s * 2 ** (self.failures - 1))
        return self._rng.uniform(ceiling / 2, ceiling)

    def reset(self) -> None:
        """Record a success."""
        self.failures = 0
