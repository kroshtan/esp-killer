"""
Persistence for the scoring job: positions in (as DataFrames), evidence, scores and flags out.

Evidence for a window and the job's progress marker are written in one transaction, so a crash can never count a
window twice or skip one.
"""

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd
from sqlalchemy import Engine, func, insert, select, update

from server.db.tables import evidence, flags, player_scores, positions, scoring_state, spawn_episodes
from server.scoring.combine import OrgEvidence, PlayerScore
from server.scoring.features import AmbushEvidence, BeelineEvidence, SpawnEpisode

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


class ScoringRepository:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    # --- positions ---

    def first_position_time(self, org_id: str) -> datetime | None:
        """
        When the org's oldest stored position was recorded.

        :param org_id: the org
        :return: the time, or None if there are no positions
        """
        with self.engine.connect() as conn:
            value: datetime | None = conn.scalar(
                select(func.min(positions.c.server_ts)).where(positions.c.org_id == org_id)
            )
        return value

    def load_positions(self, org_id: str, since: datetime, until: datetime) -> pd.DataFrame:
        """
        An org's positions in ``[since, until)``.

        :param org_id: the org
        :param since: inclusive start
        :param until: exclusive end
        :return: columns server_id, player_id, player_name, dino_class, x, y (game units), server_ts (aware UTC)
        """
        stmt = select(
            positions.c.server_id,
            positions.c.player_id,
            positions.c.player_name,
            positions.c.dino_class,
            positions.c.x,
            positions.c.y,
            positions.c.server_ts,
        ).where(positions.c.org_id == org_id, positions.c.server_ts >= since, positions.c.server_ts < until)
        with self.engine.connect() as conn:
            rows = conn.execute(stmt).all()
        return pd.DataFrame(
            rows, columns=["server_id", "player_id", "player_name", "dino_class", "x", "y", "server_ts"]
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
        latest = (
            select(positions.c.player_id, func.max(positions.c.server_ts).label("ts"))
            .where(positions.c.org_id == org_id, positions.c.player_id.in_(ids))
            .group_by(positions.c.player_id)
            .subquery()
        )
        stmt = (
            select(positions.c.player_id, positions.c.player_name)
            .join(
                latest,
                (positions.c.player_id == latest.c.player_id) & (positions.c.server_ts == latest.c.ts),
            )
            .where(positions.c.org_id == org_id)
        )
        with self.engine.connect() as conn:
            return {str(pid): str(name) for pid, name in conn.execute(stmt)}

    # --- evidence ---

    def processed_until(self, org_id: str) -> datetime | None:
        """
        The end of the last window whose evidence is stored.

        :param org_id: the org
        :return: the time, or None if nothing has been processed
        """
        with self.engine.connect() as conn:
            value: datetime | None = conn.scalar(
                select(scoring_state.c.processed_until).where(scoring_state.c.org_id == org_id)
            )
        return value

    def save_window(self, org_id: str, window_end: datetime, ev: OrgEvidence) -> None:
        """
        Store a window's evidence and advance the progress marker, atomically.

        :param org_id: the org
        :param window_end: end of the window (becomes the new ``processed_until``)
        :param ev: the window's evidence
        """
        rows = [
            {
                "org_id": org_id,
                "player_id": player_id,
                "window_end": window_end,
                "servers": ",".join(sorted(ev.servers.get(player_id, set()))),
                "moving_s": ev.beeline[player_id].moving_s,
                "beeline_episodes": ev.beeline[player_id].episodes,
                "beeline_null": ev.beeline[player_id].null_episodes,
                "beeline_start_sum_m": ev.beeline[player_id].start_distance_sum_m,
                "ambush_waits": ev.ambush[player_id].waits,
                "ambush_hits": ev.ambush[player_id].hits,
                "ambush_null": ev.ambush[player_id].null_hits,
            }
            for player_id in ev.player_ids
        ]
        episodes = [
            {
                "org_id": org_id,
                "player_id": e.player_id,
                "window_end": window_end,
                "dino_class": e.dino_class,
                "duration_s": e.duration_s,
                "censored": e.censored,
            }
            for e in ev.episodes
        ]
        with self.engine.begin() as conn:
            if rows:
                conn.execute(insert(evidence), rows)
            if episodes:
                conn.execute(insert(spawn_episodes), episodes)
            updated = conn.execute(
                update(scoring_state).where(scoring_state.c.org_id == org_id).values(processed_until=window_end)
            )
            if updated.rowcount == 0:
                conn.execute(insert(scoring_state).values(org_id=org_id, processed_until=window_end))

    def load_evidence(self, org_id: str, since: datetime) -> OrgEvidence:
        """
        The org's evidence summed over windows ending after ``since``.

        :param org_id: the org
        :param since: horizon start
        :return: the summed evidence
        """
        ev = OrgEvidence()
        with self.engine.connect() as conn:
            for row in conn.execute(
                select(evidence).where(evidence.c.org_id == org_id, evidence.c.window_end > since)
            ).mappings():
                pid = row["player_id"]
                ev.beeline[pid] += BeelineEvidence(
                    row["moving_s"], row["beeline_episodes"], row["beeline_null"], row["beeline_start_sum_m"]
                )
                ev.ambush[pid] += AmbushEvidence(row["ambush_waits"], row["ambush_hits"], row["ambush_null"])
                ev.servers[pid] |= set(filter(None, row["servers"].split(",")))
            for row in conn.execute(
                select(spawn_episodes).where(spawn_episodes.c.org_id == org_id, spawn_episodes.c.window_end > since)
            ).mappings():
                ev.episodes.append(
                    SpawnEpisode(row["player_id"], row["dino_class"], row["duration_s"], bool(row["censored"]))
                )
        return ev

    # --- scores and flags ---

    def save_scores(self, org_id: str, computed_at: datetime, scores: Sequence[PlayerScore]) -> None:
        """
        Append a score history row per player.

        :param org_id: the org
        :param computed_at: when the scores were computed
        :param scores: the scores
        """
        if not scores:
            return
        with self.engine.begin() as conn:
            conn.execute(
                insert(player_scores),
                [
                    {
                        "org_id": org_id,
                        "player_id": s.player_id,
                        "computed_at": computed_at,
                        "score": s.score,
                        "beeline": s.beeline.score if s.beeline else None,
                        "ttc": s.ttc.score if s.ttc else None,
                        "ambush": s.ambush.score if s.ambush else None,
                        "details": json.dumps(s.details()),
                    }
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
        stmt = (
            select(flags)
            .where(flags.c.org_id == org_id, flags.c.player_id.in_(ids))
            .order_by(flags.c.created_at.desc(), flags.c.id.desc())
        )
        with self.engine.connect() as conn:
            return [_flag(row) for row in conn.execute(stmt).mappings()]

    def create_flag(self, org_id: str, player_name: str, s: PlayerScore, now: datetime) -> int:
        """
        Open a flag for a player.

        :param org_id: the org
        :param player_name: the player's latest name
        :param s: the score that crossed the threshold
        :param now: current time
        :return: the new flag's id
        """
        with self.engine.begin() as conn:
            result = conn.execute(
                insert(flags).values(
                    org_id=org_id,
                    player_id=s.player_id,
                    player_name=player_name,
                    status=OPEN,
                    score=s.score,
                    max_score=s.score,
                    details=json.dumps(s.details()),
                    created_at=now,
                    updated_at=now,
                )
            )
            return int(result.inserted_primary_key[0])  # type: ignore[index]

    def update_flag(self, flag: Flag, player_name: str, s: PlayerScore, now: datetime) -> None:
        """
        Refresh an open flag with the latest score.

        :param flag: the flag
        :param player_name: the player's latest name
        :param s: the latest score
        :param now: current time
        """
        with self.engine.begin() as conn:
            conn.execute(
                update(flags)
                .where(flags.c.id == flag.id)
                .values(
                    player_name=player_name,
                    score=s.score,
                    max_score=max(flag.max_score, s.score),
                    details=json.dumps(s.details()),
                    updated_at=now,
                )
            )

    def list_flags(self, org_id: str | None = None, status: str | None = None) -> list[Flag]:
        """
        Flags, newest first.

        :param org_id: restrict to one org
        :param status: restrict to one status
        :return: the flags
        """
        stmt = select(flags).order_by(flags.c.created_at.desc(), flags.c.id.desc())
        if org_id is not None:
            stmt = stmt.where(flags.c.org_id == org_id)
        if status is not None:
            stmt = stmt.where(flags.c.status == status)
        with self.engine.connect() as conn:
            return [_flag(row) for row in conn.execute(stmt).mappings()]

    def mark_false_positive(self, flag_id: int, now: datetime, note: str | None = None) -> Flag | None:
        """
        Mark a flag as a false positive. The player is then not flagged again for a while (see the scoring config).

        :param flag_id: the flag
        :param now: current time
        :param note: optional reason
        :return: the updated flag, or None if it does not exist
        """
        with self.engine.begin() as conn:
            conn.execute(
                update(flags)
                .where(flags.c.id == flag_id)
                .values(status=FALSE_POSITIVE, resolved_at=now, updated_at=now, note=note)
            )
            row = conn.execute(select(flags).where(flags.c.id == flag_id)).mappings().first()
        return _flag(row) if row is not None else None


def _flag(row: Any) -> Flag:
    return Flag(
        id=row["id"],
        org_id=row["org_id"],
        player_id=row["player_id"],
        player_name=row["player_name"],
        status=row["status"],
        score=row["score"],
        max_score=row["max_score"],
        details=json.loads(row["details"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        resolved_at=row["resolved_at"],
        note=row["note"],
    )
