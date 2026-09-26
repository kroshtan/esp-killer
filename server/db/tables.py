"""
Database schema (SQLAlchemy Core, portable between SQLite and Postgres).

``positions`` holds exactly the fields listed in the spec and nothing else. Timestamps are timezone-aware UTC
in Python; :class:`UTCDateTime` stores them naive-UTC so SQLite (which has no timezone type) round-trips them.
"""

from datetime import UTC, datetime

from sqlalchemy import (
    Column,
    DateTime,
    Dialect,
    Float,
    Index,
    Integer,
    MetaData,
    String,
    Table,
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
