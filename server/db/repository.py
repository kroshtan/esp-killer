"""
Repository for ingest: the API talks to these methods, never to SQL directly.

The dedupe check and the inserts run in one ``BEGIN IMMEDIATE`` transaction, so two concurrent uploads of the
same snapshot cannot both store it.
"""

import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from server.db.database import Database, parse_ts, ts
from shared.models import ParseHealth, Snapshot


@dataclass(frozen=True)
class AgentStatus:
    org_id: str
    server_id: str
    agent_version: str
    last_seen: datetime
    latest_health: ParseHealth | None
    total_health: ParseHealth | None


@dataclass(frozen=True, slots=True)
class IngestResult:
    accepted_snapshots: int
    duplicate_snapshots: int
    rows: int


class Repository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def ingest(
        self, org_id: str, server_id: str, snapshots: Sequence[Snapshot], received_ts: datetime
    ) -> IngestResult:
        """
        Store snapshots that have not been stored before, in one transaction.

        :param org_id: owning organisation
        :param server_id: game server the snapshots came from
        :param snapshots: validated snapshots
        :param received_ts: when the batch arrived (UTC)
        :return: counts of new and duplicate snapshots and position rows written
        """
        ids = [str(s.snapshot_id) for s in snapshots]
        received = ts(received_ts)
        with self.db.transaction() as conn:
            existing = {
                row[0]
                for row in conn.execute(
                    "SELECT snapshot_id FROM ingested_snapshots WHERE org_id = ? AND server_id = ?"
                    f" AND snapshot_id IN ({','.join('?' * len(ids))})",
                    [org_id, server_id, *ids],
                )
            }
            # Skip stored ones, and keep the first of any snapshot repeated within the batch.
            new: dict[str, Snapshot] = {}
            for s in snapshots:
                sid = str(s.snapshot_id)
                if sid not in existing and sid not in new:
                    new[sid] = s
            conn.executemany(
                "INSERT INTO ingested_snapshots (org_id, server_id, snapshot_id, captured_at, received_ts)"
                " VALUES (?, ?, ?, ?, ?)",
                [(org_id, server_id, sid, ts(s.captured_at), received) for sid, s in new.items()],
            )
            rows = [
                (org_id, server_id, p.player_id, p.player_name, p.dino_class, p.growth, p.x, p.y, p.z)
                + (ts(s.captured_at), received)
                for s in new.values()
                for p in s.players
            ]
            conn.executemany(
                "INSERT INTO positions (org_id, server_id, player_id, player_name, dino_class, growth, x, y, z,"
                " server_ts, received_ts) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
        return IngestResult(accepted_snapshots=len(new), duplicate_snapshots=len(snapshots) - len(new), rows=len(rows))

    def count_positions(self, org_id: str | None = None) -> int:
        """
        Count stored position rows.

        :param org_id: restrict to one org
        :return: the count
        """
        with self.db.connect() as conn:
            if org_id is None:
                return int(conn.execute("SELECT COUNT(*) FROM positions").fetchone()[0])
            return int(conn.execute("SELECT COUNT(*) FROM positions WHERE org_id = ?", (org_id,)).fetchone()[0])

    def record_agent_status(
        self, org_id: str, server_id: str, agent_version: str, health: ParseHealth | None, now: datetime
    ) -> AgentStatus:
        """
        Record an upload's agent version and parse health, adding the health to the server's running total.

        :param org_id: the org
        :param server_id: the server
        :param agent_version: version the agent reported
        :param health: the batch's parse health, if the agent sent one
        :param now: current time
        :return: the status before this update (for noticing changes), or a blank one for a new server
        """
        with self.db.transaction() as conn:
            row = conn.execute(
                "SELECT * FROM agent_status WHERE org_id = ? AND server_id = ?", (org_id, server_id)
            ).fetchone()
            before = _agent_status(row) if row is not None else AgentStatus(org_id, server_id, "", now, None, None)
            latest, total = before.latest_health, before.total_health
            if health is not None:
                latest = health
                total = health if total is None else total + health
            conn.execute(
                "INSERT INTO agent_status (org_id, server_id, agent_version, last_seen, latest_health, total_health)"
                " VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT (org_id, server_id) DO UPDATE SET"
                " agent_version = excluded.agent_version, last_seen = excluded.last_seen,"
                " latest_health = excluded.latest_health, total_health = excluded.total_health",
                (
                    org_id,
                    server_id,
                    agent_version,
                    ts(now),
                    latest.model_dump_json() if latest else None,
                    total.model_dump_json() if total else None,
                ),
            )
        return before

    def agent_statuses(self) -> list[AgentStatus]:
        """
        Every server's latest agent status.

        :return: statuses ordered by org and server
        """
        with self.db.connect() as conn:
            rows = conn.execute("SELECT * FROM agent_status ORDER BY org_id, server_id").fetchall()
        return [_agent_status(row) for row in rows]

    def ping(self) -> None:
        """Check the database answers."""
        self.db.ping()


def _agent_status(row: sqlite3.Row) -> AgentStatus:
    return AgentStatus(
        org_id=row["org_id"],
        server_id=row["server_id"],
        agent_version=row["agent_version"],
        last_seen=parse_ts(row["last_seen"]),
        latest_health=_health(row["latest_health"]),
        total_health=_health(row["total_health"]),
    )


def _health(value: str | None) -> ParseHealth | None:
    return ParseHealth.model_validate_json(value) if value else None
