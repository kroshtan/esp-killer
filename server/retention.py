"""
Data retention: delete what is no longer needed, oldest first, in small batches.

Raw positions (personal data: player ids and where they were) are kept only ``positions_days`` (default 14).
Scoring evidence is counts, not positions, and is kept just long enough to cover the scoring horizon. Flags and
the alerts about them are the review record and are kept longer.

Batches keep each write transaction short, so a big first cleanup never blocks ingest for long.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

from server.db.database import Database, ts

logger = logging.getLogger(__name__)

BATCH = 5000


@dataclass(frozen=True)
class RetentionPolicy:
    positions_days: float = 14.0
    evidence_days: float = 8.0  # must exceed the scoring horizon (7 days by default)
    scores_days: float = 90.0
    alerts_days: float = 90.0
    flags_days: float = 365.0


# (table, timestamp column, policy field). Order matters: alerts reference flags, so alerts go first.
_TABLES = (
    ("positions", "server_ts", "positions_days"),
    ("ingested_snapshots", "captured_at", "positions_days"),
    ("evidence", "window_end", "evidence_days"),
    ("spawn_episodes", "window_end", "evidence_days"),
    ("pair_evidence", "window_end", "evidence_days"),
    ("leakage_state", "window_end", "evidence_days"),
    ("leakage_evidence", "window_end", "evidence_days"),
    ("leakage_scores", "computed_at", "scores_days"),
    ("player_scores", "computed_at", "scores_days"),
    ("alerts", "created_at", "alerts_days"),
    ("flags", "updated_at", "flags_days"),
)


def run_retention(db: Database, policy: RetentionPolicy, now: datetime) -> dict[str, int]:
    """
    Delete rows older than the policy allows.

    Flags are only deleted when no alert still refers to them.

    :param db: the database
    :param policy: how long to keep what
    :param now: current time
    :return: rows deleted per table
    """
    deleted: dict[str, int] = {}
    for table, column, field in _TABLES:
        cutoff = ts(now - timedelta(days=getattr(policy, field)))
        # Table and column names come from the constant above, never from input.
        extra = " AND id NOT IN (SELECT flag_id FROM alerts)" if table == "flags" else ""
        total = 0
        while True:
            with db.transaction() as conn:
                n = conn.execute(
                    f"DELETE FROM {table} WHERE rowid IN"  # noqa: S608
                    f" (SELECT rowid FROM {table} WHERE {column} < ?{extra} LIMIT ?)",
                    (cutoff, BATCH),
                ).rowcount
            total += n
            if n < BATCH:
                break
        deleted[table] = total
    logger.info("retention: deleted %s", ", ".join(f"{t}={n}" for t, n in deleted.items() if n) or "nothing")
    return deleted
