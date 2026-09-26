"""
Repository layer: the rest of the server talks to these methods, never to SQL directly.

The implementation uses SQLAlchemy Core with dialect-neutral statements, so moving to Postgres means changing
the database URL, not this code.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import Engine, func, insert, select
from sqlalchemy.exc import IntegrityError

from server.db.tables import ingested_snapshots, positions
from shared.models import Snapshot


@dataclass(frozen=True, slots=True)
class IngestResult:
    accepted_snapshots: int
    duplicate_snapshots: int
    rows: int


class Repository:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine

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
        try:
            return self._ingest(org_id, server_id, snapshots, received_ts)
        except IntegrityError:
            # A concurrent request inserted some of the same snapshots between our check and insert. Retrying
            # sees them as existing.
            return self._ingest(org_id, server_id, snapshots, received_ts)

    def _ingest(
        self, org_id: str, server_id: str, snapshots: Sequence[Snapshot], received_ts: datetime
    ) -> IngestResult:
        ids = [str(s.snapshot_id) for s in snapshots]
        with self.engine.begin() as conn:
            existing: set[str] = set(
                conn.scalars(
                    select(ingested_snapshots.c.snapshot_id).where(
                        ingested_snapshots.c.org_id == org_id,
                        ingested_snapshots.c.server_id == server_id,
                        ingested_snapshots.c.snapshot_id.in_(ids),
                    )
                )
            )
            new = [s for s in snapshots if str(s.snapshot_id) not in existing]
            # A batch may repeat a snapshot; keep the first.
            unique = list({str(s.snapshot_id): s for s in reversed(new)}.values())[::-1]
            if unique:
                conn.execute(
                    insert(ingested_snapshots),
                    [
                        {
                            "org_id": org_id,
                            "server_id": server_id,
                            "snapshot_id": str(s.snapshot_id),
                            "captured_at": s.captured_at,
                            "received_ts": received_ts,
                        }
                        for s in unique
                    ],
                )
            rows: list[dict[str, Any]] = [
                {
                    "org_id": org_id,
                    "server_id": server_id,
                    "player_id": p.player_id,
                    "player_name": p.player_name,
                    "dino_class": p.dino_class,
                    "growth": p.growth,
                    "x": p.x,
                    "y": p.y,
                    "z": p.z,
                    "server_ts": s.captured_at,
                    "received_ts": received_ts,
                }
                for s in unique
                for p in s.players
            ]
            if rows:
                conn.execute(insert(positions), rows)
        return IngestResult(
            accepted_snapshots=len(unique), duplicate_snapshots=len(snapshots) - len(unique), rows=len(rows)
        )

    def count_positions(self, org_id: str | None = None) -> int:
        """
        Count stored position rows.

        :param org_id: restrict to one org
        :return: the count
        """
        stmt = select(func.count()).select_from(positions)
        if org_id is not None:
            stmt = stmt.where(positions.c.org_id == org_id)
        with self.engine.connect() as conn:
            return int(conn.scalar(stmt) or 0)

    def ping(self) -> None:
        """Check the database answers."""
        with self.engine.connect() as conn:
            conn.scalar(select(1))
