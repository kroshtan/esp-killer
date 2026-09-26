"""
Database schema (SQLAlchemy Core, portable between SQLite and Postgres).

``positions`` holds exactly the fields listed in the spec and nothing else. Timestamps are timezone-aware UTC
in Python; :class:`UTCDateTime` stores them naive-UTC so SQLite (which has no timezone type) round-trips them.
"""

from datetime import UTC, datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Dialect,
    Float,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    TypeDecorator,
)


class UTCDateTime(TypeDecorator[datetime]):
    """Accepts only aware datetimes; returns aware UTC datetimes."""

    impl = DateTime
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Dialect) -> datetime | None:  # noqa: D102
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError("naive datetime; use timezone-aware UTC")
        return value.astimezone(UTC).replace(tzinfo=None)

    def process_result_value(self, value: datetime | None, dialect: Dialect) -> datetime | None:  # noqa: D102
        return value.replace(tzinfo=UTC) if value is not None else None


metadata = MetaData()

positions = Table(
    "positions",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("org_id", String(64), nullable=False),
    Column("server_id", String(64), nullable=False),
    Column("player_id", String(64), nullable=False),
    Column("player_name", String(128), nullable=False),
    Column("dino_class", String(64)),
    Column("growth", Float),
    Column("x", Float, nullable=False),
    Column("y", Float, nullable=False),
    Column("z", Float, nullable=False),
    Column("server_ts", UTCDateTime, nullable=False),
    Column("received_ts", UTCDateTime, nullable=False),
    # Scoring reads an org's recent window; per-player lookups serve alerts and rejoin checks; retention
    # deletes by time.
    Index("ix_positions_org_ts", "org_id", "server_ts"),
    Index("ix_positions_org_player_ts", "org_id", "player_id", "server_ts"),
    Index("ix_positions_server_ts", "server_ts"),
)

# One row per snapshot ever accepted, so retried uploads are not stored twice. Pruned with the positions.
ingested_snapshots = Table(
    "ingested_snapshots",
    metadata,
    Column("org_id", String(64), primary_key=True),
    Column("server_id", String(64), primary_key=True),
    Column("snapshot_id", String(36), primary_key=True),
    Column("captured_at", UTCDateTime, nullable=False),
    Column("received_ts", UTCDateTime, nullable=False),
    Index("ix_ingested_snapshots_captured_at", "captured_at"),
)

# --- scoring ---
# Evidence is stored per player per scoring window, as additive counts; scores are computed from the sum over the
# scoring horizon. None of these tables contain positions.

scoring_state = Table(
    "scoring_state",
    metadata,
    Column("org_id", String(64), primary_key=True),
    Column("processed_until", UTCDateTime, nullable=False),
)

evidence = Table(
    "evidence",
    metadata,
    Column("org_id", String(64), primary_key=True),
    Column("player_id", String(64), primary_key=True),
    Column("window_end", UTCDateTime, primary_key=True),
    Column("servers", String(1024), nullable=False),  # comma-separated server ids
    Column("moving_s", Float, nullable=False),
    Column("beeline_episodes", Integer, nullable=False),
    Column("beeline_null", Float, nullable=False),
    Column("beeline_start_sum_m", Float, nullable=False),
    Column("ambush_waits", Integer, nullable=False),
    Column("ambush_hits", Integer, nullable=False),
    Column("ambush_null", Float, nullable=False),
    Index("ix_evidence_org_window", "org_id", "window_end"),
)

spawn_episodes = Table(
    "spawn_episodes",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("org_id", String(64), nullable=False),
    Column("player_id", String(64), nullable=False),
    Column("window_end", UTCDateTime, nullable=False),
    Column("dino_class", String(64)),
    Column("duration_s", Float, nullable=False),
    Column("censored", Boolean, nullable=False),
    Index("ix_spawn_episodes_org_window", "org_id", "window_end"),
)

player_scores = Table(
    "player_scores",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("org_id", String(64), nullable=False),
    Column("player_id", String(64), nullable=False),
    Column("computed_at", UTCDateTime, nullable=False),
    Column("score", Float, nullable=False),
    Column("beeline", Float),
    Column("ttc", Float),
    Column("ambush", Float),
    Column("details", Text, nullable=False),  # JSON: the key numbers behind the score
    Index("ix_player_scores_org_player", "org_id", "player_id", "computed_at"),
)

flags = Table(
    "flags",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("org_id", String(64), nullable=False),
    Column("player_id", String(64), nullable=False),
    Column("player_name", String(128), nullable=False),
    Column("status", String(16), nullable=False),  # "open" or "false_positive"
    Column("score", Float, nullable=False),  # latest
    Column("max_score", Float, nullable=False),
    Column("details", Text, nullable=False),  # JSON, latest
    Column("created_at", UTCDateTime, nullable=False),
    Column("updated_at", UTCDateTime, nullable=False),
    Column("resolved_at", UTCDateTime),
    Column("note", String(500)),
    Index("ix_flags_org_player", "org_id", "player_id"),
    Index("ix_flags_status", "status"),
)
