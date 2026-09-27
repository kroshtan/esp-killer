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
import time
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
    # 2: alert outbox, and presence tracking for rejoin alerts.
    """
    -- One row per alert per channel. Destinations (webhook URLs, addresses) are not stored: they are secrets and
    -- live in config.yaml, read at send time.
    CREATE TABLE alerts (
        id              INTEGER PRIMARY KEY,
        org_id          TEXT NOT NULL,
        flag_id         INTEGER NOT NULL REFERENCES flags (id),
        player_id       TEXT NOT NULL,
        kind            TEXT NOT NULL CHECK (kind IN ('flag', 'rejoin')),
        channel         TEXT NOT NULL CHECK (channel IN ('discord', 'email')),
        server_id       TEXT,  -- for rejoin alerts: where the player showed up
        status          TEXT NOT NULL CHECK (status IN ('pending', 'sent', 'failed')),
        attempts        INTEGER NOT NULL DEFAULT 0,
        created_at      TEXT NOT NULL,
        next_attempt_at TEXT NOT NULL,
        sent_at         TEXT,
        last_error      TEXT
    );
    CREATE INDEX ix_alerts_due ON alerts (status, next_attempt_at);
    CREATE INDEX ix_alerts_org_player ON alerts (org_id, player_id, kind, created_at);

    -- When a flagged player was last seen; a sighting after a long enough gap is a rejoin.
    ALTER TABLE flags ADD COLUMN last_seen_at TEXT;
    """,
    # 3: what each server's agent reports about itself, to spot outdated agents and RCON format changes.
    """
    CREATE TABLE agent_status (
        org_id        TEXT NOT NULL,
        server_id     TEXT NOT NULL,
        agent_version TEXT NOT NULL,
        last_seen     TEXT NOT NULL,
        latest_health TEXT,  -- JSON ParseHealth of the latest upload that carried one
        total_health  TEXT,  -- JSON ParseHealth summed since the first
        PRIMARY KEY (org_id, server_id)
    );
    """,
    # 4: meetups between pairs of players, per scoring window, from which clans are inferred (see teams.py).
    """
    CREATE TABLE pair_evidence (
        org_id     TEXT NOT NULL,
        player_a   TEXT NOT NULL,  -- the smaller id of the pair
        player_b   TEXT NOT NULL,
        window_end TEXT NOT NULL,
        meets      INTEGER NOT NULL,
        null_meets REAL NOT NULL,
        PRIMARY KEY (org_id, player_a, player_b, window_end)
    );
    CREATE INDEX ix_pair_evidence_org_window ON pair_evidence (org_id, window_end);
    """,
    # 5: tips (information flow) as a second clan signal.
    """
    ALTER TABLE pair_evidence ADD COLUMN tips INTEGER NOT NULL DEFAULT 0;
    ALTER TABLE pair_evidence ADD COLUMN null_tips REAL NOT NULL DEFAULT 0;
    """,
    # 6: which scoring windows have been exported to the training dataset.
    """
    CREATE TABLE export_state (
        org_id      TEXT NOT NULL,
        window_end  TEXT NOT NULL,
        exported_at TEXT NOT NULL,
        rows        INTEGER NOT NULL,
        PRIMARY KEY (org_id, window_end)
    );
    """,
    # 7: the leakage model in shadow mode: per-window statistics per model version, and each player's latest score.
    """
    CREATE TABLE leakage_state (
        org_id        TEXT NOT NULL,
        model_version TEXT NOT NULL,
        window_end    TEXT NOT NULL,
        scored_at     TEXT NOT NULL,
        PRIMARY KEY (org_id, model_version, window_end)
    );
    CREATE TABLE leakage_evidence (
        org_id        TEXT NOT NULL,
        model_version TEXT NOT NULL,
        window_end    TEXT NOT NULL,
        player_id     TEXT NOT NULL,
        stats         TEXT NOT NULL,
        PRIMARY KEY (org_id, model_version, window_end, player_id)
    );
    CREATE INDEX leakage_evidence_window ON leakage_evidence (window_end);
    CREATE TABLE leakage_scores (
        org_id        TEXT NOT NULL,
        player_id     TEXT NOT NULL,
        model_version TEXT NOT NULL,
        computed_at   TEXT NOT NULL,
        z             REAL NOT NULL,
        score         REAL NOT NULL,
        moves         INTEGER NOT NULL,
        hours         REAL NOT NULL,
        summary       TEXT NOT NULL,
        PRIMARY KEY (org_id, player_id)
    );
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
            _enable_wal(conn)
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
        if int(conn.execute("PRAGMA user_version").fetchone()[0]) >= len(MIGRATIONS):
            return
        # The API and the worker may both start on a fresh file: take the write lock, then re-read the version, so
        # only one of them migrates. DDL is transactional in SQLite, so a failed migration leaves nothing behind.
        conn.execute("BEGIN IMMEDIATE")
        try:
            version = int(conn.execute("PRAGMA user_version").fetchone()[0])
            for number, script in enumerate(MIGRATIONS[version:], start=version + 1):
                for statement in _statements(script):
                    conn.execute(statement)
                conn.execute(f"PRAGMA user_version = {number}")
        except BaseException:
            conn.rollback()
            raise
        conn.commit()


def _enable_wal(conn: sqlite3.Connection, attempts: int = 100) -> None:
    """
    Switch the file to WAL mode (persistent, so this is a no-op after the first time).

    Switching needs an exclusive lock, and SQLite reports "database is locked" straight away instead of waiting
    for it, so when several processes open a brand-new file at once, retry briefly.

    :param conn: an open connection
    :param attempts: how many times to try, 50 ms apart
    :raises sqlite3.OperationalError: if the mode still cannot be switched
    """
    for attempt in range(attempts):
        try:
            conn.execute("PRAGMA journal_mode=WAL").fetchone()
            return
        except sqlite3.OperationalError as e:
            if "locked" not in str(e) or attempt == attempts - 1:
                raise
            time.sleep(0.05)


def _statements(script: str) -> list[str]:
    """
    Split a migration into statements (``executescript`` would commit the surrounding transaction).

    :param script: SQL statements separated by semicolons
    :return: the statements
    """
    statements, buffer = [], ""
    for line in script.splitlines(keepends=True):
        buffer += line
        if sqlite3.complete_statement(buffer):
            statements.append(buffer.strip())
            buffer = ""
    if buffer.strip():
        statements.append(buffer.strip())
    return statements
