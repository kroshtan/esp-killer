"""
Turn per-server evidence into one score per player per org.

The same player id on several servers of an org is one player: their evidence is summed across servers, and
across the windows of the scoring horizon, before scoring. Summing is what gives the scores their power: a
cheater's excess over the null grows with playing time, an honest player's does not.

Each behaviour becomes a sub-score in [0, 1], or None when there is not enough evidence to say anything.
Sub-scores are combined with a weighted noisy-OR (see :class:`~server.scoring.config.ScoringConfig`).
"""

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from server.scoring.config import ScoringConfig
from server.scoring.features import (
    AmbushEvidence,
    BeelineEvidence,
    SpawnEpisode,
    ambush_evidence,
    associates,
    beeline_evidence,
    null_shifts,
    spawn_episodes,
)
from server.scoring.teams import Pair, PairEvidence, infer_teams, pair_evidence, team_matrix
from server.scoring.trajectories import Trajectories


@dataclass(frozen=True)
class SubScore:
    score: float
    details: dict[str, float | int]


@dataclass(frozen=True)
class PlayerScore:
    player_id: str
    score: float
    beeline: SubScore | None
    ttc: SubScore | None
    ambush: SubScore | None
    servers: tuple[str, ...] = field(default_factory=tuple)

    def details(self) -> dict[str, Any]:
        """
        The key numbers behind the score, for storage and alerts. Contains no positions.

        :return: JSON-serialisable details
        """
        return {
            "servers": list(self.servers),
            **{
                name: {"score": round(sub.score, 3), **sub.details}
                for name, sub in (("beeline", self.beeline), ("ttc", self.ttc), ("ambush", self.ambush))
                if sub is not None
            },
        }


@dataclass
class OrgEvidence:
    """
    Everything the scores are computed from, for one org over some period. Adding two gives the evidence for both.

    Nothing in here is a position: it is counts and durations per player, plus spawn episodes.
    """

    beeline: defaultdict[str, BeelineEvidence] = field(default_factory=lambda: defaultdict(BeelineEvidence))
    ambush: defaultdict[str, AmbushEvidence] = field(default_factory=lambda: defaultdict(AmbushEvidence))
    episodes: list[SpawnEpisode] = field(default_factory=list)
    servers: defaultdict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    pairs: defaultdict[Pair, PairEvidence] = field(default_factory=lambda: defaultdict(PairEvidence))

    def __add__(self, other: "OrgEvidence") -> "OrgEvidence":
        """Evidence for both periods (or servers); neither operand is modified."""
        result = OrgEvidence()
        for part in (self, other):
            for player_id, bee in part.beeline.items():
                result.beeline[player_id] += bee
            for player_id, amb in part.ambush.items():
                result.ambush[player_id] += amb
            result.episodes.extend(part.episodes)
            for player_id, servers in part.servers.items():
                result.servers[player_id] |= servers
            for pair, pe in part.pairs.items():
                result.pairs[pair] += pe
        return result

    @property
    def player_ids(self) -> list[str]:
        """
        Every player with any evidence.

        :return: player ids, sorted
        """
        return sorted(self.servers)


def extract_evidence(
    trajectories: Sequence[Trajectories],
    config: ScoringConfig,
    count_from: float | None = None,
    prior_pairs: Mapping[Pair, PairEvidence] | None = None,
) -> OrgEvidence:
    """
    Extract evidence from one window of an org's servers.

    Clans are inferred first, from the meetups in this window plus ``prior_pairs`` (the rest of the horizon), so
    that information shared inside a clan is not counted as evidence.

    :param trajectories: one per server, covering the window plus any leading context
    :param config: scoring config
    :param count_from: time (same clock as ``Trajectories.t``) from which evidence is counted; earlier data is
        context only. None counts everything.
    :param prior_pairs: pair evidence from earlier windows of the horizon
    :return: the window's evidence (its pair evidence covers this window only)
    """
    ev = OrgEvidence()
    starts = [0 if count_from is None else int(np.searchsorted(tr.t, count_from)) for tr in trajectories]
    for tr, start in zip(trajectories, starts, strict=True):
        for pair, pe in pair_evidence(tr, config, null_shifts(tr, config), start).items():
            ev.pairs[pair] += pe
    all_pairs: defaultdict[Pair, PairEvidence] = defaultdict(PairEvidence, prior_pairs or {})
    for pair, pe in ev.pairs.items():
        all_pairs[pair] += pe
    teams = infer_teams(all_pairs, config)

    for tr, start in zip(trajectories, starts, strict=True):
        if len(tr.t) < 2:  # noqa: PLR2004
            continue
        team = team_matrix(tr, teams)
        assoc = associates(tr, config) | team
        for player_id, bee in beeline_evidence(tr, config, assoc, start, team=team).items():
            ev.beeline[player_id] += bee
        for player_id, amb in ambush_evidence(tr, config, assoc, start).items():
            ev.ambush[player_id] += amb
        ev.episodes.extend(spawn_episodes(tr, config, assoc, start))
        present = tr.present[start:].any(axis=0)
        for p, player_id in enumerate(tr.player_ids):
            if present[p]:
                ev.servers[player_id].add(tr.server_id)
    return ev


def score_evidence(
    evidence: OrgEvidence, config: ScoringConfig, players: Iterable[str] | None = None
) -> list[PlayerScore]:
    """
    Score players from (accumulated) evidence.

    :param evidence: the org's evidence, e.g. summed over the scoring horizon
    :param config: scoring config
    :param players: score only these players (e.g. those active in the latest window); default all
    :return: one score per player, highest first
    """
    ttc = ttc_scores(evidence.episodes, config)
    weights = (config.weight_beeline, config.weight_ttc, config.weight_ambush)
    scores = []
    for player_id in evidence.player_ids if players is None else players:
        subs = (
            beeline_score(evidence.beeline[player_id], config),
            ttc.get(player_id),
            ambush_score(evidence.ambush[player_id], config),
        )
        combined = 1.0 - math.prod(1.0 - w * s.score for w, s in zip(weights, subs, strict=True) if s is not None)
        servers = tuple(sorted(evidence.servers.get(player_id, set())))
        scores.append(PlayerScore(player_id, combined, subs[0], subs[1], subs[2], servers=servers))
    return sorted(scores, key=lambda s: s.score, reverse=True)


def score_org(trajectories: Sequence[Trajectories], config: ScoringConfig) -> list[PlayerScore]:
    """
    Score every player seen on any of an org's servers, from one window of data.

    :param trajectories: one per server
    :param config: scoring config
    :return: one score per player, highest first
    """
    return score_evidence(extract_evidence(trajectories, config), config)


def beeline_score(ev: BeelineEvidence, config: ScoringConfig) -> SubScore | None:
    """
    Sub-score from beeline evidence: how many more beelines than the time-shifted null predicts.

    :param ev: summed evidence
    :param config: scoring config
    :return: the sub-score, or None below the evidence gate
    """
    if ev.moving_s < config.beeline_min_moving_s:
        return None
    z = count_z(ev.episodes, ev.null_episodes)
    return SubScore(
        _ramp(z, config.beeline_z_floor, config.beeline_z_full),
        {
            "episodes": ev.episodes,
            "null_episodes": round(ev.null_episodes, 2),
            "z": round(z, 2),
            "moving_minutes": round(ev.moving_s / 60, 1),
            "mean_start_distance_m": round(ev.start_distance_sum_m / ev.episodes) if ev.episodes else 0,
        },
    )


def ambush_score(ev: AmbushEvidence, config: ScoringConfig) -> SubScore | None:
    """
    Sub-score from ambush evidence: how many more successful waits than the time-shifted null predicts.

    :param ev: summed evidence
    :param config: scoring config
    :return: the sub-score, or None below the evidence gate
    """
    if ev.waits < config.ambush_min_waits:
        return None
    z = count_z(ev.hits, ev.null_hits)
    return SubScore(
        _ramp(z, config.ambush_z_floor, config.ambush_z_full),
        {"waits": ev.waits, "hits": ev.hits, "null_hits": round(ev.null_hits, 2), "z": round(z, 2)},
    )


def count_z(observed: float, expected: float) -> float:
    """
    How far an observed count exceeds the null's expectation, in (Poisson) standard deviations.

    The +1 keeps a null of zero from turning one or two chance events into a huge z.

    :param observed: observed count
    :param expected: expected count under the null
    :return: the z-score (negative if fewer than expected)
    """
    return (observed - expected) / math.sqrt(expected + 1.0)


def _ramp(z: float, floor: float, full: float) -> float:
    """0 at or below ``floor``, 1 at or above ``full``, linear in between."""
    if full <= floor:
        return 1.0 if z >= full else 0.0
    return min(1.0, max(0.0, (z - floor) / (full - floor)))


def ttc_scores(episodes: Iterable[SpawnEpisode], config: ScoringConfig) -> dict[str, SubScore]:
    """
    Sub-scores from time to contact, relative to everyone else's spawns of the same class.

    Each of a player's episodes gets a rank in [0, 1] against the baseline of *other* players' episodes (the
    fraction that reached contact faster; censored episodes are handled conservatively). Under the null
    hypothesis, that the player is no faster than anyone else, the ranks are uniform, so their mean has
    expectation 0.5 and standard deviation ``1 / sqrt(12 n)``. The z-score of the mean rank is the evidence, which
    automatically discounts players with few episodes.

    Vectorised with sorted baselines and leave-one-out counts, because a week of an org's spawns is tens of
    thousands of episodes.

    :param episodes: all spawn episodes in the org's scoring horizon
    :param config: scoring config
    :return: sub-score per player with at least ``ttc_min_episodes`` episodes
    """
    episodes = list(episodes)
    if not episodes:
        return {}
    duration = np.array([e.duration_s for e in episodes], dtype=float)
    censored = np.array([e.censored for e in episodes], dtype=bool)
    classes = np.array([e.dino_class or "" for e in episodes], dtype=object)
    players = np.array([e.player_id for e in episodes], dtype=object)
    everyone = _Baseline(duration, censored)
    by_class = {c: _Baseline(duration[classes == c], censored[classes == c]) for c in set(classes.tolist())}
    server_median = float(np.median(duration[~censored])) if (~censored).any() else -1.0

    result = {}
    for player_id in sorted(set(players.tolist())):
        mine = players == player_id
        n_own = int(mine.sum())
        if n_own < config.ttc_min_episodes or everyone.n - n_own < config.ttc_min_baseline:
            continue
        own_d, own_c, own_cls = duration[mine], censored[mine], classes[mine]
        ranks = np.empty(n_own)
        for cls in set(own_cls.tolist()):
            sel = own_cls == cls
            baseline, own_in_baseline = by_class[cls], sel
            if baseline.n - int(sel.sum()) < config.ttc_min_baseline:
                baseline, own_in_baseline = everyone, np.ones(n_own, dtype=bool)
            t = own_d[sel]
            below = baseline.weight_below(t) - _Baseline(own_d[own_in_baseline], own_c[own_in_baseline]).weight_below(
                t
            )
            frac = below / (baseline.n - int(own_in_baseline.sum()))
            # A censored episode made no contact within its observed time: somewhere between there and slowest.
            ranks[sel] = np.where(own_c[sel], (frac + 1.0) / 2.0, frac)
        mean_rank = float(ranks.mean())
        z = (0.5 - mean_rank) * math.sqrt(12 * n_own)
        contacts = own_d[~own_c]
        result[player_id] = SubScore(
            _ramp(z, config.ttc_z_floor, config.ttc_z_full),
            {
                "episodes": n_own,
                "contacts": len(contacts),
                "mean_rank": round(mean_rank, 3),
                "z": round(z, 2),
                "median_ttc_s": round(float(np.median(contacts))) if len(contacts) else -1,
                "server_median_ttc_s": round(server_median),
            },
        )
    return result


class _Baseline:
    """Spawn-episode durations, sorted, for counting how many reached contact before a given time."""

    def __init__(self, duration: np.ndarray, censored: np.ndarray) -> None:
        self.contact = np.sort(duration[~censored])
        self.left = np.sort(duration[censored])
        self.n = len(duration)

    def weight_below(self, t: np.ndarray) -> np.ndarray:
        """Episodes that made contact before each ``t``, plus half of those that left before it without contact."""
        return np.searchsorted(self.contact, t, side="left") + 0.5 * np.searchsorted(self.left, t, side="left")
