"""
A bounded on-disk FIFO of snapshots waiting to be uploaded.

It is SQLite from the standard library, so it works in a PyInstaller binary and survives agent restarts. When the
backend is unreachable the queue grows up to ``max_bytes`` and then drops the oldest snapshots: recent data is
more useful than old data, and the game server host's disk must never fill up because of this agent.

Every operation is a small local transaction (well under a millisecond), so they run directly on the event loop.
"""

import logging
import sqlite3
from pathlib import Path

from shared.models import Snapshot

logger = logging.getLogger(__name__)


class SnapshotQueue:
    def __init__(self, path: Path, max_bytes: int) -> None:
        self.path = path
        self.max_bytes = max_bytes
        self.dropped = 0
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, isolation_level=None)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS snapshots ("
            " id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " payload TEXT NOT NULL,"
            " size INTEGER NOT NULL)"
        )
        self._bytes = int(self._db.execute("SELECT COALESCE(SUM(size), 0) FROM snapshots").fetchone()[0])

    def __len__(self) -> int:
        """Number of queued snapshots."""
        return int(self._db.execute("SELECT COUNT(*) FROM snapshots").fetchone()[0])

    @property
    def size_bytes(self) -> int:
        """
        Total payload size currently queued.

        :return: bytes
        """
        return self._bytes

    def put(self, snapshot: Snapshot) -> None:
        """
        Append a snapshot, dropping the oldest ones if the queue is over its size cap.

        :param snapshot: the snapshot to queue
        """
        payload = snapshot.model_dump_json()
        size = len(payload.encode("utf-8"))
        with self._db:
            self._db.execute("BEGIN")
            self._db.execute("INSERT INTO snapshots (payload, size) VALUES (?, ?)", (payload, size))
            self._bytes += size
            dropped = 0
            while self._bytes > self.max_bytes:
                row = self._db.execute("SELECT id, size FROM snapshots ORDER BY id LIMIT 1").fetchone()
                if row is None:
                    break
                self._db.execute("DELETE FROM snapshots WHERE id = ?", (row[0],))
                self._bytes -= int(row[1])
                dropped += 1
        if dropped:
            self.dropped += dropped
            logger.warning("upload queue full (%d bytes): dropped %d oldest snapshot(s)", self.max_bytes, dropped)

    def peek(self, limit: int) -> list[tuple[int, Snapshot]]:
        """
        The oldest ``limit`` snapshots, without removing them.

        :param limit: maximum number to return
        :return: (queue id, snapshot) pairs, oldest first
        """
        rows = self._db.execute("SELECT id, payload FROM snapshots ORDER BY id LIMIT ?", (limit,)).fetchall()
        return [(int(row_id), Snapshot.model_validate_json(payload)) for row_id, payload in rows]

    def ack(self, ids: list[int]) -> None:
        """
        Remove snapshots, after the backend accepted them (or they were rejected for good).

        :param ids: queue ids from :meth:`peek`
        """
        if not ids:
            return
        placeholders = ",".join("?" * len(ids))
        with self._db:
            self._db.execute("BEGIN")
            freed = self._db.execute(
                f"SELECT COALESCE(SUM(size), 0) FROM snapshots WHERE id IN ({placeholders})",  # noqa: S608
                ids,
            ).fetchone()[0]
            self._db.execute(f"DELETE FROM snapshots WHERE id IN ({placeholders})", ids)  # noqa: S608
        self._bytes -= int(freed)

    def close(self) -> None:
        """Close the database."""
        self._db.close()
