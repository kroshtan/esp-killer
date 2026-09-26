"""
The database: SQLite through the standard library, plain SQL, no ORM.

Every operation opens its own short-lived connection, so the API's threadpool, the worker and the CLI can all use
the database at once. WAL mode lets readers proceed while one writer writes. Writes start with ``BEGIN IMMEDIATE``,
which takes the write lock up front: a read-then-write transaction (like ingest's dedupe check) can then never
race another writer or fail halfway with "database is locked".

The schema is a list of migrations applied in order and tracked in ``PRAGMA user_version``. To change the schema,
append a migration; never edit one that has shipped.

Timestamps are stored as fixed-width UTC ISO-8601 text (``2026-01-01T00:00:00.000000Z``), which sorts and compares
correctly as text and stays readable in the ``sqlite3`` shell.

Moving to Postgres later means a second implementation of the two repositories (``?`` placeholders become ``%s``);
the SQL itself sticks to what both databases support.
"""

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

_TS_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"

MIGRATIONS: list[str] = [
    # 1: ingest, and scoring evidence, scores and flags.
    """
    CREATE TABLE positions (
        id          INTEGER PRIMARY KEY,
        org_id      TEXT NOT NULL,
        server_id   TEXT NOT NULL,
        player_id   TEXT NOT NULL,
        player_name TEXT NOT NULL,
        dino_class  TEXT,
        growth      REAL,
        x           REAL NOT NULL,
        y           REAL NOT NULL,
        z           REAL NOT NULL,
        server_ts   TEXT NOT NULL,
        received_ts TEXT NOT NULL
    );
    -- Scoring reads an org's recent window; per-player lookups serve flags and alerts; retention deletes by time.
    CREATE INDEX ix_positions_org_ts ON positions (org_id, server_ts);
    CREATE INDEX ix_positions_org_player_ts ON positions (org_id, player_id, server_ts);
    CREATE INDEX ix_positions_ts ON positions (server_ts);

    -- One row per snapshot ever accepted, so retried uploads are not stored twice.
    CREATE TABLE ingested_snapshots (
        org_id      TEXT NOT NULL,
        server_id   TEXT NOT NULL,
        snapshot_id TEXT NOT NULL,
        captured_at TEXT NOT NULL,
        received_ts TEXT NOT NULL,
        PRIMARY KEY (org_id, server_id, snapshot_id)
    );
    CREATE INDEX ix_ingested_snapshots_captured_at ON ingested_snapshots (captured_at);

    -- Scoring. Evidence is additive counts per player per window; none of these tables contain positions.
    CREATE TABLE scoring_state (
        org_id          TEXT PRIMARY KEY,
        processed_until TEXT NOT NULL
    );

    CREATE TABLE evidence (
        org_id              TEXT NOT NULL,
        player_id           TEXT NOT NULL,
        window_end          TEXT NOT NULL,
        servers             TEXT NOT NULL,  -- comma-separated server ids
        moving_s            REAL NOT NULL,
        beeline_episodes    INTEGER NOT NULL,
        beeline_null        REAL NOT NULL,
        beeline_start_sum_m REAL NOT NULL,
        ambush_waits        INTEGER NOT NULL,
        ambush_hits         INTEGER NOT NULL,
        ambush_null         REAL NOT NULL,
        PRIMARY KEY (org_id, player_id, window_end)
    );
    CREATE INDEX ix_evidence_org_window ON evidence (org_id, window_end);

    CREATE TABLE spawn_episodes (
        id         INTEGER PRIMARY KEY,
        org_id     TEXT NOT NULL,
        player_id  TEXT NOT NULL,
        window_end TEXT NOT NULL,
        dino_class TEXT,
        duration_s REAL NOT NULL,
        censored   INTEGER NOT NULL
    );
    CREATE INDEX ix_spawn_episodes_org_window ON spawn_episodes (org_id, window_end);

    CREATE TABLE player_scores (
        id          INTEGER PRIMARY KEY,
        org_id      TEXT NOT NULL,
        player_id   TEXT NOT NULL,
        computed_at TEXT NOT NULL,
        score       REAL NOT NULL,
        beeline     REAL,
        ttc         REAL,
        ambush      REAL,
        details     TEXT NOT NULL  -- JSON: the key numbers behind the score
    );
    CREATE INDEX ix_player_scores_org_player ON player_scores (org_id, player_id, computed_at);

    CREATE TABLE flags (
        id          INTEGER PRIMARY KEY,
        org_id      TEXT NOT NULL,
        player_id   TEXT NOT NULL,
        player_name TEXT NOT NULL,
        status      TEXT NOT NULL CHECK (status IN ('open', 'false_positive')),
        score       REAL NOT NULL,  -- latest
        max_score   REAL NOT NULL,
        details     TEXT NOT NULL,  -- JSON, latest
        created_at  TEXT NOT NULL,
        updated_at  TEXT NOT NULL,
        resolved_at TEXT,
        note        TEXT
    );
    CREATE INDEX ix_flags_org_player ON flags (org_id, player_id);
    CREATE INDEX ix_flags_status ON flags (status);
    """,
]


def ts(value: datetime) -> str:
    """
    Format an aware datetime for storage.

    :param value: the time; must be timezone-aware
    :return: fixed-width UTC ISO-8601 text
    :raises ValueError: for a naive datetime
    """
    if value.tzinfo is None:
        raise ValueError("naive datetime; use timezone-aware UTC")
    return value.astimezone(UTC).strftime(_TS_FORMAT)


def parse_ts(value: str) -> datetime:
    """
    Parse a stored timestamp.

    :param value: text written by :func:`ts`
    :return: an aware UTC datetime
    """
    return datetime.strptime(value, _TS_FORMAT).replace(tzinfo=UTC)


class Database:
    def __init__(self, path: Path | str) -> None:
        """
        Open (and if needed create and migrate) the database file.

        :param path: path of the SQLite file; its directory is created if missing
        """
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            self._migrate(conn)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        """
        A connection for reading, or for writing with explicit transactions.

        :yield: the connection (autocommit mode, rows as :class:`sqlite3.Row`), closed afterwards
        """
        conn = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
        finally:
            conn.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """
        A write transaction: committed if the block succeeds, rolled back if it raises.

        :yield: the connection, inside ``BEGIN IMMEDIATE``
        """
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            ok = False
            try:
                yield conn
                ok = True
            finally:
                if ok:
                    conn.commit()
                else:
                    conn.rollback()

    def ping(self) -> None:
        """Check the database answers."""
        with self.connect() as conn:
            conn.execute("SELECT 1").fetchone()

    @staticmethod
    def _migrate(conn: sqlite3.Connection) -> None:
        version = int(conn.execute("PRAGMA user_version").fetchone()[0])
        for number, script in enumerate(MIGRATIONS[version:], start=version + 1):
            # executescript commits any open transaction first, so the migration and its version bump are one
            # script: BEGIN ... COMMIT.
            conn.executescript(f"BEGIN IMMEDIATE;\n{script}\nPRAGMA user_version = {number};\nCOMMIT;")
