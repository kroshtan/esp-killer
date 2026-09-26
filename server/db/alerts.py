"""
Persistence for alerts: the outbox the worker delivers from, and flagged players' presence for rejoin alerts.

The outbox holds one row per alert per channel, so a Discord outage does not hold up email or the other way round.
Delivery destinations are deliberately not stored here; they are secrets in config.yaml.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime

from server.db.database import Database, parse_ts, ts
from server.db.scoring import OPEN, Flag, flag_from_row

PENDING = "pending"
SENT = "sent"
FAILED = "failed"
KINDS = ("flag", "rejoin")
CHANNELS = ("discord", "email")


@dataclass(frozen=True)
class OutboxItem:
    id: int
    org_id: str
    flag_id: int
    player_id: str
    kind: str
    channel: str
    server_id: str | None
    attempts: int
    created_at: datetime


class AlertRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def enqueue(
        self,
        flag: Flag,
        kind: str,
        channels: Iterable[str],
        now: datetime,
        server_id: str | None = None,
    ) -> list[int]:
        """
        Queue an alert about a flag on each channel.

        :param flag: the flag the alert is about
        :param kind: ``flag`` (the score crossed the threshold) or ``rejoin`` (the player showed up again)
        :param channels: ``discord`` and/or ``email``
        :param now: current time; the alert is due immediately
        :param server_id: for rejoin alerts, the server the player joined
        :return: ids of the queued rows
        :raises ValueError: for an unknown kind or channel
        """
        channels = list(channels)
        if kind not in KINDS or not set(channels) <= set(CHANNELS):
            raise ValueError(f"bad alert kind {kind!r} or channels {channels!r}")
        ids = []
        with self.db.transaction() as conn:
            for channel in channels:
                row = conn.execute(
                    "INSERT INTO alerts (org_id, flag_id, player_id, kind, channel, server_id, status, created_at,"
                    " next_attempt_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) RETURNING id",
                    (flag.org_id, flag.id, flag.player_id, kind, channel, server_id, PENDING, ts(now), ts(now)),
                ).fetchone()
                ids.append(int(row[0]))
        return ids

    def last_alert_at(self, org_id: str, player_id: str, kind: str) -> datetime | None:
        """
        When the most recent alert of this kind about this player was queued.

        :param org_id: the org
        :param player_id: the player
        :param kind: alert kind
        :return: the time, or None if there never was one
        """
        with self.db.connect() as conn:
            value = conn.execute(
                "SELECT MAX(created_at) FROM alerts WHERE org_id = ? AND player_id = ? AND kind = ?",
                (org_id, player_id, kind),
            ).fetchone()[0]
        return parse_ts(value) if value is not None else None

    def due(self, now: datetime, limit: int = 50) -> list[OutboxItem]:
        """
        Pending alerts whose next attempt is due, oldest first.

        :param now: current time
        :param limit: maximum number to return
        :return: the items
        """
        with self.db.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM alerts WHERE status = ? AND next_attempt_at <= ? ORDER BY next_attempt_at, id LIMIT ?",
                (PENDING, ts(now), limit),
            ).fetchall()
        return [
            OutboxItem(
                id=row["id"],
                org_id=row["org_id"],
                flag_id=row["flag_id"],
                player_id=row["player_id"],
                kind=row["kind"],
                channel=row["channel"],
                server_id=row["server_id"],
                attempts=row["attempts"],
                created_at=parse_ts(row["created_at"]),
            )
            for row in rows
        ]

    def mark_sent(self, item_id: int, now: datetime) -> None:
        """
        Record a successful delivery.

        :param item_id: outbox row
        :param now: current time
        """
        with self.db.transaction() as conn:
            conn.execute(
                "UPDATE alerts SET status = ?, sent_at = ?, attempts = attempts + 1, last_error = NULL WHERE id = ?",
                (SENT, ts(now), item_id),
            )

    def mark_retry(self, item_id: int, next_attempt_at: datetime, error: str) -> None:
        """
        Record a failed attempt that will be retried.

        :param item_id: outbox row
        :param next_attempt_at: when to try again
        :param error: short description, without secrets
        """
        with self.db.transaction() as conn:
            conn.execute(
                "UPDATE alerts SET attempts = attempts + 1, next_attempt_at = ?, last_error = ? WHERE id = ?",
                (ts(next_attempt_at), error[:500], item_id),
            )

    def mark_failed(self, item_id: int, error: str) -> None:
        """
        Give up on an alert.

        :param item_id: outbox row
        :param error: short description, without secrets
        """
        with self.db.transaction() as conn:
            conn.execute(
                "UPDATE alerts SET status = ?, attempts = attempts + 1, last_error = ? WHERE id = ?",
                (FAILED, error[:500], item_id),
            )

    def status_counts(self) -> dict[str, int]:
        """
        Number of outbox rows per status.

        :return: status -> count
        """
        with self.db.connect() as conn:
            return {row[0]: int(row[1]) for row in conn.execute("SELECT status, COUNT(*) FROM alerts GROUP BY status")}

    # --- flags and presence ---

    def get_flag(self, flag_id: int) -> Flag | None:
        """
        Load one flag.

        :param flag_id: the flag
        :return: the flag, or None if it does not exist
        """
        with self.db.connect() as conn:
            row = conn.execute("SELECT * FROM flags WHERE id = ?", (flag_id,)).fetchone()
        return flag_from_row(row) if row is not None else None

    def open_flags(self, org_id: str) -> list[Flag]:
        """
        The org's open flags.

        :param org_id: the org
        :return: the flags
        """
        with self.db.connect() as conn:
            rows = conn.execute("SELECT * FROM flags WHERE org_id = ? AND status = ?", (org_id, OPEN)).fetchall()
        return [flag_from_row(row) for row in rows]

    def set_last_seen(self, flag_id: int, seen_at: datetime) -> None:
        """
        Record when a flagged player was last seen.

        :param flag_id: the flag
        :param seen_at: time of the latest sighting
        """
        with self.db.transaction() as conn:
            conn.execute("UPDATE flags SET last_seen_at = ? WHERE id = ?", (ts(seen_at), flag_id))

    def latest_sighting(self, org_id: str, player_id: str) -> datetime | None:
        """
        When a player was last seen on any of the org's servers.

        :param org_id: the org
        :param player_id: the player
        :return: the time, or None if never
        """
        with self.db.connect() as conn:
            value = conn.execute(
                "SELECT MAX(server_ts) FROM positions WHERE org_id = ? AND player_id = ?", (org_id, player_id)
            ).fetchone()[0]
        return parse_ts(value) if value is not None else None

    def sightings(self, org_id: str, player_id: str, after: datetime | None) -> list[tuple[datetime, str]]:
        """
        When and where a player was seen, oldest first.

        :param org_id: the org
        :param player_id: the player
        :param after: only sightings strictly after this time; None for all
        :return: (time, server id) pairs
        """
        with self.db.connect() as conn:
            rows = conn.execute(
                "SELECT server_ts, server_id FROM positions WHERE org_id = ? AND player_id = ? AND server_ts > ?"
                " ORDER BY server_ts",
                (org_id, player_id, ts(after) if after is not None else ""),
            ).fetchall()
        return [(parse_ts(row[0]), row[1]) for row in rows]
