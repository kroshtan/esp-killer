"""
The scoring job: turn newly stored positions into evidence, scores and flags.

For each org, every run:

1. processes each complete window (``window_minutes``) since the last processed one, up to ``now - lag``. It reads
   the window plus ``context_minutes`` before it, extracts evidence counted only inside the window, and stores it
   together with the progress marker;
2. if any window was processed, re-scores the players active in those windows from the evidence summed over the
   last ``horizon_days``;
3. opens a flag for each player at or above ``flag_threshold`` who has no open flag and was not marked a false
   positive within ``false_positive_suppress_days``, and refreshes the scores on open flags.

Scores are evidence for a human. Nothing here kicks, bans or messages anyone.
"""

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import pandas as pd

from server.db.scoring import FALSE_POSITIVE, OPEN, Flag, ScoringRepository
from server.orgconfig import OrgConfig
from server.scoring.combine import OrgEvidence, PlayerScore, extract_evidence, score_evidence
from server.scoring.config import ScoringConfig
from server.scoring.game import GameProfile, load_profile
from server.scoring.trajectories import Trajectories, from_frame

logger = logging.getLogger(__name__)

# Positions can arrive a little late (the agent uploads every ~15 s); windows are only processed once this much
# time has passed after their end.
DEFAULT_LAG = timedelta(minutes=5)
# Catching up after downtime happens over several runs rather than one very long one.
MAX_WINDOWS_PER_RUN = 12


@dataclass
class OrgRunResult:
    org_id: str
    windows: int = 0
    scored: int = 0
    new_flag_ids: list[int] = field(default_factory=list)


def run_scoring(
    repo: ScoringRepository, config: OrgConfig, now: datetime, lag: timedelta = DEFAULT_LAG
) -> list[OrgRunResult]:
    """
    Run the job once for every org.

    :param repo: scoring repository
    :param config: org config (orgs and scoring thresholds)
    :param now: current time (UTC)
    :param lag: how long after a window's end before it is processed
    :return: what happened per org
    """
    cfg = config.scoring_config
    results = []
    for org_id in config.orgs:
        result = score_org_windows(repo, org_id, cfg, now, lag, profiles=config.game_profiles(org_id))
        results.append(result)
        if result.windows:
            logger.info(
                "%s: %d window(s) processed, %d player(s) scored, %d new flag(s)",
                org_id,
                result.windows,
                result.scored,
                len(result.new_flag_ids),
            )
    return results


def score_org_windows(
    repo: ScoringRepository,
    org_id: str,
    cfg: ScoringConfig,
    now: datetime,
    lag: timedelta,
    *,
    profiles: Mapping[str, GameProfile] | None = None,
) -> OrgRunResult:
    """
    Process an org's pending windows, then re-score and flag.

    :param repo: scoring repository
    :param org_id: the org
    :param cfg: scoring config
    :param now: current time
    :param lag: how long after a window's end before it is processed
    :param profiles: game profile per server id; servers not listed use the default profile
    :return: what happened
    """
    result = OrgRunResult(org_id)
    window = timedelta(minutes=cfg.window_minutes)
    start = repo.processed_until(org_id) or repo.first_position_time(org_id)
    if start is None:
        return result

    active: set[str] = set()
    while start + window <= now - lag and result.windows < MAX_WINDOWS_PER_RUN:
        end = start + window
        ev = window_evidence(repo, org_id, cfg, start, end, profiles=profiles)
        repo.save_window(org_id, end, ev)
        active.update(ev.player_ids)
        result.windows += 1
        start = end
    if not active:
        return result

    total = repo.load_evidence(org_id, since=now - timedelta(days=cfg.horizon_days))
    scores = score_evidence(total, cfg, players=sorted(active))
    repo.save_scores(org_id, now, scores)
    result.scored = len(scores)
    result.new_flag_ids = apply_flags(repo, org_id, scores, cfg, now)
    return result


def window_evidence(
    repo: ScoringRepository,
    org_id: str,
    cfg: ScoringConfig,
    start: datetime,
    end: datetime,
    *,
    profiles: Mapping[str, GameProfile] | None = None,
) -> OrgEvidence:
    """
    Extract one window's evidence from the database.

    :param repo: scoring repository
    :param org_id: the org
    :param cfg: scoring config
    :param start: window start
    :param end: window end
    :param profiles: game profile per server id
    :return: evidence counted inside ``[start, end)``
    """
    frame = repo.load_positions(org_id, start - timedelta(minutes=cfg.context_minutes), end)
    return extract_evidence(to_trajectories(frame, cfg, profiles), cfg, count_from=start.timestamp())


def to_trajectories(
    frame: pd.DataFrame, cfg: ScoringConfig, profiles: Mapping[str, GameProfile] | None = None
) -> list[Trajectories]:
    """
    Convert stored positions (game units, datetimes) into one resampled trajectory set per server.

    :param frame: output of :meth:`ScoringRepository.load_positions`
    :param cfg: scoring config (units and grid)
    :param profiles: game profile per server id; servers not listed use the default profile
    :return: one per server with data
    """
    if frame.empty:
        return []
    samples = pd.DataFrame(
        {
            "server_id": frame["server_id"],
            "t": epoch_seconds(frame["server_ts"]),
            "player_id": frame["player_id"],
            "x": frame["x"] / cfg.units_per_metre,
            "y": frame["y"] / cfg.units_per_metre,
            "dino_class": frame["dino_class"],
        }
    )
    profiles = profiles or {}
    return [
        from_frame(group, str(server_id), cfg, profiles.get(str(server_id)) or load_profile())
        for server_id, group in samples.groupby("server_id", sort=True)
    ]


def epoch_seconds(timestamps: pd.Series) -> pd.Series:
    """
    Convert stored UTC timestamps to epoch seconds.

    :param timestamps: timestamps as stored (UTC ISO text) or datetimes
    :return: float seconds since the epoch
    """
    return (pd.to_datetime(timestamps, utc=True) - pd.Timestamp(0, tz="UTC")).dt.total_seconds()


def apply_flags(
    repo: ScoringRepository, org_id: str, scores: Sequence[PlayerScore], cfg: ScoringConfig, now: datetime
) -> list[int]:
    """
    Open or refresh flags for players at or above the threshold.

    :param repo: scoring repository
    :param org_id: the org
    :param scores: fresh scores
    :param cfg: scoring config
    :param now: current time
    :return: ids of newly opened flags
    """
    above = [s for s in scores if s.score >= cfg.flag_threshold]
    if not above:
        return []
    existing: dict[str, list[Flag]] = {}
    for flag in repo.flags_for(org_id, [s.player_id for s in above]):
        existing.setdefault(flag.player_id, []).append(flag)
    names = repo.latest_names(org_id, [s.player_id for s in above])
    suppress_since = now - timedelta(days=cfg.false_positive_suppress_days)

    new_ids = []
    for s in above:
        player_flags = existing.get(s.player_id, [])
        name = names.get(s.player_id, s.player_id)
        open_flag = next((f for f in player_flags if f.status == OPEN), None)
        if open_flag is not None:
            repo.update_flag(open_flag, name, s, now)
            continue
        if any(
            f.status == FALSE_POSITIVE and f.resolved_at is not None and f.resolved_at >= suppress_since
            for f in player_flags
        ):
            continue
        new_ids.append(repo.create_flag(org_id, name, s, now))
    return new_ids
