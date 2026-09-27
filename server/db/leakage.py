"""
Persistence for the leakage model in shadow mode.

Per scoring window and model version, each player's additive :class:`LeakageStats` are stored; a window is marked
scored for a version together with its statistics, so a crash never counts a window twice. Summing a player's
statistics over the evidence horizon and rescoring gives their current score, of which only the latest is kept.
A newly promoted model scores the windows still within the horizon again, so its scores build up within days.
"""

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime

from server.db.database import Database, parse_ts, ts
from server.training.leakage import LeakageScore, LeakageStats


@dataclass(frozen=True)
class ShadowScore:
    """A player's latest leakage score. Shown next to the rule-based score; it never opens a flag."""

    org_id: str
    player_id: str
    model_version: str
    computed_at: datetime
    z: float
    score: float
    moves: int
    hours: float
    summary: str

    def line(self) -> str:
        """
        One line for alerts and the CLI.

        :return: e.g. ``"Model (shadow, not part of the score): leakage z 3.1 over 5.0 h of movement"``
        """
        return (
            f"Model (shadow, not part of the score): leakage z {self.z:.1f} over {self.hours:.1f} h of movement"
            f" (model {self.model_version})"
        )


def _placeholders(n: int) -> str:
    return ",".join("?" * n)


class LeakageRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def pending_windows(self, model_version: str, since: datetime, limit: int) -> list[tuple[str, datetime]]:
        """
        Scoring windows ending after ``since`` that this model version has not scored yet, oldest first.

        :param model_version: the model version
        :param since: only windows ending after this (the evidence horizon)
        :param limit: maximum number to return
        :return: (org id, window end) pairs
        """
        with self.db.connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT e.org_id, e.window_end FROM evidence e"
                " LEFT JOIN leakage_state s"
                " ON s.org_id = e.org_id AND s.window_end = e.window_end AND s.model_version = ?"
                " WHERE s.org_id IS NULL AND e.window_end > ? ORDER BY e.window_end LIMIT ?",
                (model_version, ts(since), limit),
            ).fetchall()
        return [(row[0], parse_ts(row[1])) for row in rows]

    def save_window(
        self,
        org_id: str,
        model_version: str,
        window_end: datetime,
        stats: Mapping[str, LeakageStats],
        now: datetime,
    ) -> None:
        """
        Store one window's statistics and mark it scored, atomically.

        :param org_id: the org
        :param model_version: the model version that computed them
        :param window_end: end of the window
        :param stats: player id -> statistics
        :param now: current time
        """
        end = ts(window_end)
        with self.db.transaction() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO leakage_evidence (org_id, model_version, window_end, player_id, stats)"
                " VALUES (?, ?, ?, ?, ?)",
                [(org_id, model_version, end, pid, json.dumps(s.to_dict())) for pid, s in stats.items()],
            )
            conn.execute(
                "INSERT OR REPLACE INTO leakage_state (org_id, model_version, window_end, scored_at)"
                " VALUES (?, ?, ?, ?)",
                (org_id, model_version, end, ts(now)),
            )

    def totals(
        self, org_id: str, model_version: str, since: datetime, player_ids: Iterable[str]
    ) -> dict[str, LeakageStats]:
        """
        Players' statistics summed over windows ending after ``since``.

        :param org_id: the org
        :param model_version: the model version
        :param since: horizon start
        :param player_ids: the players
        :return: player id -> summed statistics, for players with any
        """
        ids = sorted(set(player_ids))
        total: dict[str, LeakageStats] = {}
        with self.db.connect() as conn:
            for start in range(0, len(ids), 500):
                chunk = ids[start : start + 500]
                rows = conn.execute(
                    "SELECT player_id, stats FROM leakage_evidence"
                    " WHERE org_id = ? AND model_version = ? AND window_end > ?"
                    f" AND player_id IN ({_placeholders(len(chunk))})",
                    (org_id, model_version, ts(since), *chunk),
                ).fetchall()
                for pid, stats in rows:
                    total[pid] = total.get(pid, LeakageStats()) + LeakageStats.from_dict(json.loads(stats))
        return total

    def save_scores(
        self,
        org_id: str,
        model_version: str,
        computed_at: datetime,
        scores: Mapping[str, LeakageScore],
        stride_s: float,
    ) -> None:
        """
        Replace players' latest scores.

        :param org_id: the org
        :param model_version: the model version
        :param computed_at: when they were computed
        :param scores: player id -> score
        :param stride_s: seconds of movement per move (the model's sampling stride)
        """
        with self.db.transaction() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO leakage_scores"
                " (org_id, player_id, model_version, computed_at, z, score, moves, hours, summary)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        org_id,
                        pid,
                        model_version,
                        ts(computed_at),
                        s.z,
                        s.score_0_1,
                        s.n,
                        s.n * stride_s / 3600,
                        s.summary,
                    )
                    for pid, s in scores.items()
                ],
            )

    def latest(self, org_id: str, player_id: str) -> ShadowScore | None:
        """
        A player's latest score.

        :param org_id: the org
        :param player_id: the player
        :return: the score, or None if the model has not scored them
        """
        with self.db.connect() as conn:
            row = conn.execute(
                "SELECT org_id, player_id, model_version, computed_at, z, score, moves, hours, summary"
                " FROM leakage_scores WHERE org_id = ? AND player_id = ?",
                (org_id, player_id),
            ).fetchone()
        if row is None:
            return None
        return ShadowScore(row[0], row[1], row[2], parse_ts(row[3]), row[4], row[5], row[6], row[7], row[8])
