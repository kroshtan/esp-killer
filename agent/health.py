"""
Tally how RCON responses parse, for reporting to the backend with each upload (see ``shared.models.ParseHealth``).

The poller adds every parse result; the uploader takes the tally when it builds a batch and gives it back if the
upload fails, so nothing is double counted or lost.
"""

import re
import threading

from shared.models import ParseHealth
from shared.playerdata import ParseResult

_FIELD_NAME = re.compile(r"^[A-Za-z]{1,40}$")


class HealthTally:
    def __init__(self) -> None:
        self._health = ParseHealth()
        self._lock = threading.Lock()

    def record(self, result: ParseResult) -> None:
        """
        Count one parsed response.

        :param result: the parse result
        """
        errors: dict[str, int] = {}
        for e in result.errors:
            reason = ("salvaged: " if e.salvaged else "") + e.reason
            errors[reason[:80]] = errors.get(reason[:80], 0) + 1
        salvaged = sum(e.salvaged for e in result.errors)
        one = ParseHealth(
            polls=1,
            lines=len(result.players) + len(result.errors) - salvaged,
            players=len(result.players),
            salvaged=salvaged,
            unparsed=len(result.errors) - salvaged,
            ended=int(result.ended),
            errors=errors,
            unknown_keys=sorted(k for k in result.unknown_keys if _FIELD_NAME.match(k))[:20],
        )
        with self._lock:
            self._health = self._health + one

    def take(self) -> ParseHealth | None:
        """
        Take the tally so far and start a new one.

        :return: the tally, or None if nothing was polled since the last take
        """
        with self._lock:
            health, self._health = self._health, ParseHealth()
        return health if health.polls else None

    def give_back(self, health: ParseHealth | None) -> None:
        """
        Return a tally whose upload failed, so it is sent with the next one.

        :param health: what :meth:`take` returned
        """
        if health is not None:
            with self._lock:
                self._health = health + self._health
