from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

from server.db.database import MIGRATIONS, Database, parse_ts, ts


def test_timestamps_round_trip_and_sort_as_text() -> None:
    a = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    b = a + timedelta(microseconds=1)
    assert parse_ts(ts(a)) == a
    assert ts(a) < ts(b)
    assert len(ts(a)) == len(ts(b))
    # Other offsets are stored as UTC.
    assert ts(a.astimezone(timezone(timedelta(hours=2)))) == ts(a)
    with pytest.raises(ValueError, match="naive"):
        ts(datetime(2026, 1, 1))  # noqa: DTZ001


def test_migrations_run_once(tmp_path: Path) -> None:
    Database(tmp_path / "x.db")
    db = Database(tmp_path / "x.db")  # reopening must not re-run them
    with db.connect() as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == len(MIGRATIONS)
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"


def test_transaction_rolls_back_on_error(tmp_path: Path) -> None:
    db = Database(tmp_path / "x.db")
    with pytest.raises(RuntimeError), db.transaction() as conn:
        conn.execute("INSERT INTO scoring_state VALUES ('org', '2026-01-01T00:00:00.000000Z')")
        raise RuntimeError
    with db.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM scoring_state").fetchone()[0] == 0
