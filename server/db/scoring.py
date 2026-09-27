"""
Persistence for the scoring job: positions in (as DataFrames), evidence, scores and flags out.

Evidence for a window and the job's progress marker are written in one transaction, so a crash can never count a
window twice or skip one.
"""

import json
import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd

from server.db.database import Database, parse_ts, ts
from server.scoring.combine import OrgEvidence, PlayerScore
from server.scoring.features import AmbushEvidence, BeelineEvidence, SpawnEpisode
from server.scoring.teams import Pair, PairEvidence

OPEN = "open"
FALSE_POSITIVE = "false_positive"
FLAG_STATUSES = (OPEN, FALSE_POSITIVE)


@dataclass(frozen=True)
class Flag:
    id: int
    org_id: str
    player_id: str
    player_name: str
    status: str
    score: float
    max_score: float
    details: dict[str, Any]
    created_at: datetime
    updated_at: datetime
    resolved_at: datetime | None
    note: str | None
    last_seen_at: datetime | None = None


def _placeholders(n: int) -> str:
    return ",".join("?" * n)


class ScoringRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    # --- positions ---

    def first_position_time(self, org_id: str) -> datetime | None:
        """
        When the org's oldest stored position was recorded.

        :param org_id: the org
        :return: the time, or None if there are no positions
        """
        with self.db.connect() as conn:
            value = conn.execute("SELECT MIN(server_ts) FROM positions WHERE org_id = ?", (org_id,)).fetchone()[0]
        return parse_ts(value) if value is not None else None

    def load_positions(self, org_id: str, since: datetime, until: datetime) -> pd.DataFrame:
        """
        An org's positions in ``[since, until)``.

        :param org_id: the org
        :param since: inclusive start
        :param until: exclusive end
        :return: columns server_id, player_id, player_name, dino_class, x, y (game units), server_ts (UTC text)
        """
        with self.db.connect() as conn:
            return pd.read_sql_query(
                "SELECT server_id, player_id, player_name, dino_class, x, y, server_ts FROM positions"
                " WHERE org_id = ? AND server_ts >= ? AND server_ts < ?",
                conn,
                params=(org_id, ts(since), ts(until)),
            )

    def latest_names(self, org_id: str, player_ids: Iterable[str]) -> dict[str, str]:
        """
        Each player's most recently seen name.

        :param org_id: the org
        :param player_ids: players to look up
        :return: player id -> name (players without positions are missing)
        """
        ids = list(player_ids)
        if not ids:
            return {}
        # SQLite returns the other columns from the row holding the MAX() (documented "bare column" behaviour).
        with self.db.connect() as conn:
            rows = conn.execute(
                "SELECT player_id, player_name, MAX(server_ts) FROM positions"
                f" WHERE org_id = ? AND player_id IN ({_placeholders(len(ids))}) GROUP BY player_id",
                [org_id, *ids],
            ).fetchall()
        return {row["player_id"]: row["player_name"] for row in rows}

    # --- evidence ---

    def processed_until(self, org_id: str) -> datetime | None:
        """
        The end of the last window whose evidence is stored.

        :param org_id: the org
        :return: the time, or None if nothing has been processed
        """
        with self.db.connect() as conn:
            row = conn.execute("SELECT processed_until FROM scoring_state WHERE org_id = ?", (org_id,)).fetchone()
        return parse_ts(row[0]) if row is not None else None

    def save_window(self, org_id: str, window_end: datetime, ev: OrgEvidence) -> None:
        """
        Store a window's evidence and advance the progress marker, atomically.

        :param org_id: the org
        :param window_end: end of the window (becomes the new ``processed_until``)
        :param ev: the window's evidence
        """
        end = ts(window_end)
        with self.db.transaction() as conn:
            conn.executemany(
                "INSERT INTO evidence (org_id, player_id, window_end, servers, moving_s, beeline_episodes,"
                " beeline_null, beeline_start_sum_m, ambush_waits, ambush_hits, ambush_null)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        org_id,
                        pid,
                        end,
                        ",".join(sorted(ev.servers.get(pid, set()))),
                        ev.beeline[pid].moving_s,
                        ev.beeline[pid].episodes,
                        ev.beeline[pid].null_episodes,
                        ev.beeline[pid].start_distance_sum_m,
                        ev.ambush[pid].waits,
                        ev.ambush[pid].hits,
                        ev.ambush[pid].null_hits,
                    )
                    for pid in ev.player_ids
                ],
            )
            conn.executemany(
                "INSERT INTO spawn_episodes (org_id, player_id, window_end, dino_class, duration_s, censored)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                [(org_id, e.player_id, end, e.dino_class, e.duration_s, e.censored) for e in ev.episodes],
            )
            conn.executemany(
                "INSERT INTO pair_evidence"
                " (org_id, player_a, player_b, window_end, meets, null_meets, tips, null_tips)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (org_id, a, b, end, pe.meets, pe.null_meets, pe.tips, pe.null_tips)
                    for (a, b), pe in ev.pairs.items()
                ],
            )
            conn.execute(
                "INSERT INTO scoring_state (org_id, processed_until) VALUES (?, ?)"
                " ON CONFLICT (org_id) DO UPDATE SET processed_until = excluded.processed_until",
                (org_id, end),
            )

    def load_evidence(self, org_id: str, since: datetime) -> OrgEvidence:
        """
        The org's evidence summed over windows ending after ``since``.

        :param org_id: the org
        :param since: horizon start
        :return: the summed evidence
        """
        ev = OrgEvidence()
        with self.db.connect() as conn:
            for row in conn.execute("SELECT * FROM evidence WHERE org_id = ? AND window_end > ?", (org_id, ts(since))):
                pid = row["player_id"]
                ev.beeline[pid] += BeelineEvidence(
                    row["moving_s"], row["beeline_episodes"], row["beeline_null"], row["beeline_start_sum_m"]
                )
                ev.ambush[pid] += AmbushEvidence(row["ambush_waits"], row["ambush_hits"], row["ambush_null"])
                ev.servers[pid] |= set(filter(None, row["servers"].split(",")))
            for row in conn.execute(
                "SELECT player_id, dino_class, duration_s, censored FROM spawn_episodes"
                " WHERE org_id = ? AND window_end > ?",
                (org_id, ts(since)),
            ):
                ev.episodes.append(
                    SpawnEpisode(row["player_id"], row["dino_class"], row["duration_s"], bool(row["censored"]))
                )
        return ev

    def load_pair_evidence(self, org_id: str, since: datetime) -> dict[Pair, PairEvidence]:
        """
        Pair meetups summed over windows ending after ``since``.

        :param org_id: the org
        :param since: horizon start
        :return: evidence per pair
        """
        with self.db.connect() as conn:
            rows = conn.execute(
                "SELECT player_a, player_b, SUM(meets), SUM(null_meets), SUM(tips), SUM(null_tips) FROM pair_evidence"
                " WHERE org_id = ? AND window_end > ? GROUP BY player_a, player_b",
                (org_id, ts(since)),
            ).fetchall()
        return {
            (a, b): PairEvidence(int(meets), float(null_meets), int(tips), float(null_tips))
            for a, b, meets, null_meets, tips, null_tips in rows
        }

    # --- scores and flags ---

    def save_scores(self, org_id: str, computed_at: datetime, scores: Sequence[PlayerScore]) -> None:
        """
        Append a score history row per player.

        :param org_id: the org
        :param computed_at: when the scores were computed
        :param scores: the scores
        """
        with self.db.transaction() as conn:
            conn.executemany(
                "INSERT INTO player_scores (org_id, player_id, computed_at, score, beeline, ttc, ambush, details)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        org_id,
                        s.player_id,
                        ts(computed_at),
                        s.score,
                        s.beeline.score if s.beeline else None,
                        s.ttc.score if s.ttc else None,
                        s.ambush.score if s.ambush else None,
                        json.dumps(s.details()),
                    )
                    for s in scores
                ],
            )

    def flags_for(self, org_id: str, player_ids: Iterable[str]) -> list[Flag]:
        """
        All flags (any status) of the given players.

        :param org_id: the org
        :param player_ids: the players
        :return: their flags, newest first
        """
        ids = list(player_ids)
        if not ids:
            return []
        with self.db.connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM flags WHERE org_id = ? AND player_id IN ({_placeholders(len(ids))})"
                " ORDER BY created_at DESC, id DESC",
                [org_id, *ids],
            ).fetchall()
        return [flag_from_row(row) for row in rows]

    def create_flag(self, org_id: str, player_name: str, s: PlayerScore, now: datetime) -> int:
        """
        Open a flag for a player.

        :param org_id: the org
        :param player_name: the player's latest name
        :param s: the score that crossed the threshold
        :param now: current time
        :return: the new flag's id
        """
        with self.db.transaction() as conn:
            row = conn.execute(
                "INSERT INTO flags (org_id, player_id, player_name, status, score, max_score, details, created_at,"
                " updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) RETURNING id",
                (org_id, s.player_id, player_name, OPEN, s.score, s.score, json.dumps(s.details()), ts(now), ts(now)),
            ).fetchone()
        return int(row[0])

    def update_flag(self, flag: Flag, player_name: str, s: PlayerScore, now: datetime) -> None:
        """
        Refresh an open flag with the latest score.

        :param flag: the flag
        :param player_name: the player's latest name
        :param s: the latest score
        :param now: current time
        """
        with self.db.transaction() as conn:
            conn.execute(
                "UPDATE flags SET player_name = ?, score = ?, max_score = ?, details = ?, updated_at = ? WHERE id = ?",
                (player_name, s.score, max(flag.max_score, s.score), json.dumps(s.details()), ts(now), flag.id),
            )

    def list_flags(self, org_id: str | None = None, status: str | None = None) -> list[Flag]:
        """
        Flags, newest first.

        :param org_id: restrict to one org
        :param status: restrict to one status
        :return: the flags
        """
        where, params = [], []
        if org_id is not None:
            where.append("org_id = ?")
            params.append(org_id)
        if status is not None:
            where.append("status = ?")
            params.append(status)
        clause = f" WHERE {' AND '.join(where)}" if where else ""
        with self.db.connect() as conn:
            rows = conn.execute(f"SELECT * FROM flags{clause} ORDER BY created_at DESC, id DESC", params).fetchall()
        return [flag_from_row(row) for row in rows]

    def mark_false_positive(self, flag_id: int, now: datetime, note: str | None = None) -> Flag | None:
        """
        Mark a flag as a false positive. The player is then not flagged again for a while (see the scoring config).

        :param flag_id: the flag
        :param now: current time
        :param note: optional reason
        :return: the updated flag, or None if it does not exist
        """
        with self.db.transaction() as conn:
            row = conn.execute(
                "UPDATE flags SET status = ?, resolved_at = ?, updated_at = ?, note = ? WHERE id = ? RETURNING *",
                (FALSE_POSITIVE, ts(now), ts(now), note, flag_id),
            ).fetchone()
        return flag_from_row(row) if row is not None else None


def flag_from_row(row: sqlite3.Row) -> Flag:
    """
    Build a :class:`Flag` from a ``flags`` row.

    :param row: the row
    :return: the flag
    """
    return Flag(
        id=row["id"],
        org_id=row["org_id"],
        player_id=row["player_id"],
        player_name=row["player_name"],
        status=row["status"],
        score=row["score"],
        max_score=row["max_score"],
        details=json.loads(row["details"]),
        created_at=parse_ts(row["created_at"]),
        updated_at=parse_ts(row["updated_at"]),
        resolved_at=parse_ts(row["resolved_at"]) if row["resolved_at"] is not None else None,
        note=row["note"],
        last_seen_at=parse_ts(row["last_seen_at"]) if row["last_seen_at"] is not None else None,
    )
