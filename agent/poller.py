"""
Poll the game server for player data and queue one snapshot per poll.

Health, stamina, hunger and thirst are in the RCON response but are dropped here: :class:`PlayerSample` has no
fields for them. Snapshots with no players are queued too, because "the server was up and empty" is information
the backend needs (for rejoin detection and to tell gaps from absences).
"""

import asyncio
import contextlib
import logging
import time
import uuid
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime

from pydantic import ValidationError

from agent.backoff import Backoff
from agent.queue import SnapshotQueue
from agent.rcon import EvrimaRconClient, RconAuthError, RconError
from shared.models import PlayerSample, Snapshot
from shared.playerdata import ParseResult, parse_player_data
from shared.rcon_protocol import ReadOnlyCommand

logger = logging.getLogger(__name__)

# A wrong password will not fix itself; retry slowly so the game server's log is not flooded.
AUTH_RETRY_S = 60.0


def build_snapshot(result: ParseResult, captured_at: datetime) -> tuple[Snapshot, int]:
    """
    Turn a parse result into a snapshot, skipping players that fail validation.

    :param result: the parsed RCON response
    :param captured_at: when the response was received (UTC)
    :return: the snapshot, and how many parsed players were rejected by validation
    """
    samples = []
    invalid = 0
    for p in result.players:
        try:
            samples.append(
                PlayerSample(
                    player_id=p.player_id,
                    player_name=p.player_name,
                    dino_class=p.dino_class,
                    growth=p.growth,
                    x=p.x,
                    y=p.y,
                    z=p.z,
                )
            )
        except ValidationError:
            invalid += 1
    return Snapshot(snapshot_id=uuid.uuid4(), captured_at=captured_at, players=samples), invalid


class Poller:
    def __init__(
        self,
        client: EvrimaRconClient,
        queue: SnapshotQueue,
        interval_s: float,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self.client = client
        self.queue = queue
        self.interval_s = interval_s
        self.clock = clock
        self.polls = 0
        self._backoff = Backoff(base_s=1.0, cap_s=60.0)

    async def poll_once(self) -> Snapshot:
        """
        Fetch player data once and queue the snapshot.

        Raises :class:`~agent.rcon.RconError` if the server cannot be reached.

        :return: the queued snapshot
        """
        if not self.client.connected:
            await self.client.connect()
            logger.info("connected to RCON at %s:%d", self.client.host, self.client.port)
        text = await self.client.request(ReadOnlyCommand.PLAYER_DATA)
        captured_at = self.clock()
        result = parse_player_data(text)
        snapshot, invalid = build_snapshot(result, captured_at)
        if result.errors or invalid:
            # Only reasons and field keys, never names, ids or positions.
            reasons = Counter(e.reason for e in result.errors)
            logger.warning(
                "player data: %d player(s) parsed, %d unparseable line(s) %s, %d invalid",
                len(result.players),
                len(result.errors),
                dict(reasons),
                invalid,
            )
            for error in result.errors[:5]:
                logger.debug("unparseable line %d (%s): keys=%s", error.line_no, error.reason, error.shape)
        self.queue.put(snapshot)
        self.polls += 1
        return snapshot

    async def run(self, stop: asyncio.Event) -> None:
        """
        Poll every ``interval_s`` until ``stop`` is set, reconnecting with backoff on errors.

        :param stop: set to stop polling
        """
        next_tick = time.monotonic()
        while not stop.is_set():
            try:
                await self.poll_once()
                self._backoff.reset()
            except RconAuthError:
                logger.error("RCON password rejected; retrying in %.0fs", AUTH_RETRY_S)
                next_tick = time.monotonic() + AUTH_RETRY_S
            except RconError as e:
                delay = self._backoff.next_delay()
                logger.warning("RCON unavailable (%s); retrying in %.1fs", e, delay)
                next_tick = time.monotonic() + delay
            else:
                next_tick += self.interval_s
                # After a stall (e.g. the machine slept), skip missed ticks rather than polling in a burst.
                now = time.monotonic()
                if next_tick < now:
                    next_tick = now + self.interval_s
            await _sleep_until(next_tick, stop)
        await self.client.close()


async def _sleep_until(deadline: float, stop: asyncio.Event) -> None:
    with contextlib.suppress(TimeoutError):
        await asyncio.wait_for(stop.wait(), timeout=max(0.0, deadline - time.monotonic()))
