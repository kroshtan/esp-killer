"""
Repository for ingest: the API talks to these methods, never to SQL directly.

The dedupe check and the inserts run in one ``BEGIN IMMEDIATE`` transaction, so two concurrent uploads of the
same snapshot cannot both store it.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from server.db.database import Database, ts
from shared.models import Snapshot


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

    def ping(self) -> None:
        """Check the database answers."""
        self.db.ping()
