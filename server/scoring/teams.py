"""
Clans from behaviour: which players coordinate with each other, inferred from movement alone.

A clan here is any set of players who play together: an in-game group, a mixed-species team, or a large clan whose
members spread over the map and share sightings over voice chat. Nothing about that is visible to the server, so
it is inferred from what persistent teammates do and strangers do not: **meet up again and again**. Members of a
dispersed clan split up to hunt or scout, but keep coming back together (to regroup, at a base, to travel), day
after day.

Like the rest of the detector, meetups are compared with a time-shifted null: the other player's trajectory
shifted by 10-30 minutes. Two strangers who both frequent the same waterhole also meet in the shifted version;
teammates meet far more often than that.

Pair evidence is additive, so it is summed over the scoring horizon (a week) before pairs are linked. Linking is
deliberately lenient: a wrong link can only excuse an approach (a cheater caught later), while a missed link can
flag a fair player, which is worse.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np

from server.scoring.config import ScoringConfig
from server.scoring.trajectories import Trajectories

Pair = tuple[str, str]  # player ids, sorted


@dataclass
class PairEvidence:
    meets: int = 0  # separate meetups (together, after having been apart)
    null_meets: float = 0.0  # the same against time-shifted trajectories (mean over shifts)

    def __add__(self, other: "PairEvidence") -> "PairEvidence":
        """Evidence for both periods (or servers)."""
        return PairEvidence(self.meets + other.meets, self.null_meets + other.null_meets)

    def z(self) -> float:
        """
        How far meetups exceed what chance explains, in (Poisson) standard deviations.

        :return: the z-score
        """
        return (self.meets - self.null_meets) / float(np.sqrt(self.null_meets + 1.0))


def pair_key(a: str, b: str) -> Pair:
    """
    The canonical key for a pair of players.

    :param a: a player id
    :param b: another player id
    :return: the ids, sorted
    """
    return (a, b) if a < b else (b, a)


def pair_evidence(
    tr: Trajectories, config: ScoringConfig, shifts: Iterable[int], start: int = 0
) -> dict[Pair, PairEvidence]:
    """
    Count meetups for every pair of players on one server, and the null expectation.

    A meetup is a spell of at least ``team_meet_min_s`` within ``team_meet_radius_m``, starting at or after
    ``start``, that follows a period apart (beyond ``team_apart_m``) or is the pair's first. A meeting after which
    either player dies (disappears or respawns) within ``team_fight_grace_s`` was a fight and does not count.
    Pairs that never met are left out.

    :param tr: trajectories
    :param config: scoring config
    :param shifts: time shifts (grid steps) for the null
    :param start: count only meetups beginning at this grid index or later
    :return: evidence per pair
    """
    shifts = list(shifts)
    n = len(tr.player_ids)
    result: dict[Pair, PairEvidence] = {}
    if len(tr.t) < 2:  # noqa: PLR2004
        return result
    died = _deaths(tr, config)
    for i in range(n - 1):
        others = tr.pos[:, i + 1 :, :]
        observed = _meetups(
            tr, config, tr.pos[:, i, :], others, start=start, my_death=died[:, i], their_death=died[:, i + 1 :]
        )
        if not observed.any():
            continue
        null = np.zeros(observed.shape)
        for s in shifts:
            shifted = np.roll(others, s, axis=0)
            null += _meetups(
                tr,
                config,
                tr.pos[:, i, :],
                shifted,
                start=start,
                my_death=died[:, i],
                their_death=np.roll(died[:, i + 1 :], s, axis=0),
            )
        null /= max(1, len(shifts))
        for k in np.flatnonzero(observed):
            j = i + 1 + int(k)
            result[pair_key(tr.player_ids[i], tr.player_ids[j])] = PairEvidence(int(observed[k]), float(null[k]))
    return result


def _deaths(tr: Trajectories, config: ScoringConfig) -> np.ndarray:
    """(T, P): whether each player dies (disappears or jumps, i.e. respawns) within the fight grace period."""
    present = tr.present
    step = np.zeros(present.shape)
    step[1:] = np.nan_to_num(np.hypot(*(tr.pos[1:] - tr.pos[:-1]).transpose(2, 0, 1)))
    ends = present & ~np.roll(present, -1, axis=0)
    ends[-1] = False  # the window ending is not a death
    jumps = np.roll(step > config.respawn_jump_m, -1, axis=0)
    jumps[-1] = False
    event = ends | jumps
    # Look ahead: an event anywhere in the next grace steps.
    grace = tr.steps(config.team_fight_grace_s)
    padded = np.vstack([event, np.zeros((grace, event.shape[1]), dtype=bool)])
    c = np.cumsum(padded, axis=0)
    ahead = c[grace:] - np.vstack([np.zeros((1, event.shape[1])), c[: len(event) - 1]])[: len(event)]
    return np.asarray(ahead > 0)


def _meetups(
    tr: Trajectories,
    config: ScoringConfig,
    mine: np.ndarray,
    others: np.ndarray,
    *,
    start: int,
    my_death: np.ndarray,
    their_death: np.ndarray,
) -> np.ndarray:
    """Meetups between one player (T, 2) and each of several others (T, K, 2), fights excluded."""
    rel = others - mine[:, None, :]
    dist = np.hypot(rel[..., 0], rel[..., 1])
    with np.errstate(invalid="ignore"):
        close = dist <= config.team_meet_radius_m
        apart = dist > config.team_apart_m
    min_steps = tr.steps(config.team_meet_min_s)
    counts = np.zeros(dist.shape[1], dtype=int)
    for k in np.flatnonzero(close.any(axis=0)):
        previous_end: int | None = None
        for run_start, run_end in _runs(close[:, k]):
            if run_end - run_start + 1 < min_steps:
                continue
            separated = previous_end is None or bool(apart[previous_end:run_start, k].any())
            fight = bool(my_death[run_end] or their_death[run_end, k])
            if separated and run_start >= start and not fight:
                counts[k] += 1
            previous_end = run_end
    return counts


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.concatenate(([False], mask, [False])).astype(np.int8)
    edges = np.flatnonzero(np.diff(padded))
    return list(zip(edges[0::2].tolist(), (edges[1::2] - 1).tolist(), strict=True))


def infer_teams(pairs: Mapping[Pair, PairEvidence], config: ScoringConfig) -> dict[str, int]:
    """
    Group players into clans: link pairs whose meetups clearly exceed chance, then take connected groups.

    :param pairs: pair evidence, summed over the horizon
    :param config: scoring config (``team_min_meets``, ``team_link_z``)
    :return: player id -> clan number, for players in a clan of two or more
    """
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for (a, b), ev in pairs.items():
        if ev.meets >= config.team_min_meets and ev.z() >= config.team_link_z:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
    roots = sorted({find(p) for p in parent})
    number = {root: n for n, root in enumerate(roots)}
    return {p: number[find(p)] for p in parent}


def team_matrix(tr: Trajectories, teams: Mapping[str, int] | None) -> np.ndarray:
    """
    Which players on a server are clanmates.

    :param tr: trajectories
    :param teams: player id -> clan number
    :return: (P, P) bool, False on the diagonal
    """
    n = len(tr.player_ids)
    if not teams:
        return np.zeros((n, n), dtype=bool)
    # Players without a clan get a unique negative number, so they match nobody.
    clan = np.array([teams.get(p, -1 - k) for k, p in enumerate(tr.player_ids)])
    result: np.ndarray = clan[:, None] == clan[None, :]
    np.fill_diagonal(result, False)
    return result
