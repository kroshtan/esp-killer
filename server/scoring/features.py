"""
Behavioural evidence from one server's trajectories. Pure functions of numpy arrays: no I/O, no clock.

Each feature returns *sufficient statistics* per player (times, counts), not scores, so evidence from several
servers, and from successive time windows, can simply be added up before it is turned into a score (see
``combine.py``). Every feature takes ``start``: the grid index from which evidence is counted. Data before it is
context only (for look-backs and the null), so consecutive windows with overlapping context never count the
same episode twice.

Beeline and ambush are both measured twice: once against where the other players really were, and once against
a **null** in which every other trajectory is shifted in time by 10-30 minutes. The null keeps everything about
*where* people go (waterholes, trails, popular spots) and destroys only *where they are right now*, which is the
one thing an honest player cannot know about someone out of sight. The evidence is the excess over the null.
"""

from dataclasses import dataclass

import numpy as np

from server.scoring.config import ScoringConfig
from server.scoring.teams import Pair, PairEvidence, deaths, pair_key
from server.scoring.trajectories import Trajectories


@dataclass
class BeelineEvidence:
    moving_s: float = 0.0
    episodes: int = 0  # beelines from out of range that ended at the target
    null_episodes: float = 0.0  # the same, against time-shifted players (mean over shifts)
    start_distance_sum_m: float = 0.0  # for the mean start distance in alert details

    def __add__(self, other: "BeelineEvidence") -> "BeelineEvidence":
        """Evidence for both periods (or servers)."""
        return BeelineEvidence(
            self.moving_s + other.moving_s,
            self.episodes + other.episodes,
            self.null_episodes + other.null_episodes,
            self.start_distance_sum_m + other.start_distance_sum_m,
        )


@dataclass
class AmbushEvidence:
    waits: int = 0
    hits: int = 0  # waits where a player who was far away at the start arrived
    null_hits: float = 0.0  # the same, against time-shifted players (mean over shifts)

    def __add__(self, other: "AmbushEvidence") -> "AmbushEvidence":
        """Evidence for both periods (or servers)."""
        return AmbushEvidence(self.waits + other.waits, self.hits + other.hits, self.null_hits + other.null_hits)


@dataclass(frozen=True)
class BeelineEpisode:
    start: int  # grid index where the player lined up on the target
    arrival: int  # grid index of closest approach
    target: int  # player index of the target
    start_distance_m: float


@dataclass(frozen=True)
class SpawnEpisode:
    player_id: str
    dino_class: str | None
    duration_s: float  # time to first contact, or observed time without contact if censored
    censored: bool  # True if no contact happened before the player left, respawned or the window ended


def pairwise_distances(pos: np.ndarray, i: int, others: np.ndarray | None = None) -> np.ndarray:
    """
    Distance from player ``i`` to every player, at every time.

    :param pos: (T, P, 2) positions
    :param i: player index
    :param others: (T, P, 2) positions to measure to; defaults to ``pos``
    :return: (T, P) distances, NaN where either player is absent
    """
    others = pos if others is None else others
    rel = others - pos[:, i : i + 1, :]
    return np.asarray(np.hypot(rel[..., 0], rel[..., 1]))


def associates(tr: Trajectories, config: ScoringConfig) -> np.ndarray:
    """
    Which pairs of players spent enough time together to be considered a group.

    :param tr: trajectories
    :param config: scoring config
    :return: (P, P) symmetric bool matrix, False on the diagonal
    """
    n = len(tr.player_ids)
    close_s = np.zeros((n, n))
    for i in range(n):
        d = pairwise_distances(tr.pos, i)
        close_s[i] = np.sum(d <= config.associate_radius_m, axis=0) * tr.dt
    result = close_s >= config.associate_min_s
    np.fill_diagonal(result, False)
    return np.asarray(result | result.T)


def null_shifts(tr: Trajectories, config: ScoringConfig) -> list[int]:
    """
    The time shifts (in grid steps) usable for the null on this window.

    A shift is only used if it is at most half the window, so shifted and unshifted data mostly overlap.

    :param tr: trajectories
    :param config: scoring config
    :return: shifts in steps, possibly empty
    """
    return [tr.steps(s) for s in config.null_shifts_s if tr.steps(s) <= len(tr.t) // 2]


def _shifted(pos: np.ndarray, steps: int) -> np.ndarray:
    return np.roll(pos, steps, axis=0)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """(start, end) index pairs, end inclusive, of consecutive True values."""
    padded = np.concatenate(([False], mask, [False])).astype(np.int8)
    edges = np.flatnonzero(np.diff(padded))
    return list(zip(edges[0::2].tolist(), (edges[1::2] - 1).tolist(), strict=True))


def _bridge(mask: np.ndarray, steps: int) -> np.ndarray:
    """Fill False gaps of at most ``steps`` between True values in a 1-D mask."""
    out = mask.copy()
    runs = _runs(mask)
    for (_, prev_end), (next_start, _) in zip(runs, runs[1:], strict=False):
        if next_start - prev_end - 1 <= steps:
            out[prev_end + 1 : next_start] = True
    return out


def _recently(mask: np.ndarray, steps: int) -> np.ndarray:
    """Per column, whether ``mask`` was True at any of the last ``steps`` + 1 rows (including this one)."""
    c = np.cumsum(np.vstack([np.zeros((1, mask.shape[1])), mask.astype(float)]), axis=0)
    lagged = np.vstack([np.zeros((steps, mask.shape[1])), c[:-steps]]) if steps < len(c) else np.zeros_like(c)
    return np.asarray((c[1:] - lagged[: len(mask)]) > 0)


def _headings(tr: Trajectories, config: ScoringConfig) -> tuple[np.ndarray, np.ndarray]:
    """Look-ahead displacement over ``heading_window_s`` and whether the player counts as moving."""
    k = tr.steps(config.heading_window_s)
    disp = np.full_like(tr.pos, np.nan)
    if len(tr.t) > k:
        disp[:-k] = tr.pos[k:] - tr.pos[:-k]
    speed = np.hypot(disp[..., 0], disp[..., 1]) / (k * tr.dt)
    with np.errstate(invalid="ignore"):
        moving = speed >= config.min_speed_mps
    return disp, moving


def beeline_evidence(
    tr: Trajectories,
    config: ScoringConfig,
    assoc: np.ndarray,
    start: int = 0,
    team: np.ndarray | None = None,
    *,
    anyone_sighting: bool = False,
) -> dict[str, BeelineEvidence]:
    """
    Count each player's beelines: heading persistently for a player who was out of range, and arriving.

    An episode is a run of grid steps in which the player moves and heads within ``beeline_cos`` of the bearing to
    the same target, for at least ``beeline_min_duration_s`` while the target is out of range, where:

    * the target was beyond the player's awareness range (from their class, see ``game/``) at the start and had
      not been within it for ``near_lookback_s``;
    * the player stays lined up until the target comes within ``beeline_arrive_m`` (normally the awareness range;
      the run may end up to ``beeline_arrive_grace_s`` before that). What happens after sighting is not evidence;
    * the player *turned* onto the target: ``beeline_turn_lookback_s`` earlier they were not already heading for
      the meeting place;
    * the target did not come to the player (``beeline_max_target_approach``);
    * it did not end in a peaceful reunion (together a while soon after, nobody dying): meeting up is not hunting;
    * no clanmate (``team``) had the target within their own range shortly before (``team_shared_awareness_s``):
      information shared inside a clan is legitimate. Clanmates are never targets.

    Chasing someone you saw recently, walking into an ambush or up to someone resting on your route, meeting
    someone head-on, being hunted, and heading for a group mate (``assoc``) therefore never count. Overlapping
    episodes (walking up to several people at once) count as one.

    :param tr: trajectories
    :param config: scoring config
    :param assoc: (P, P) associate matrix from :func:`associates`
    :param start: count only moving time and episodes from this grid index on
    :param team: (P, P) clanmates: what they had in range, or were already heading for, counts as known. Not
        excluded as targets by this; pass clanmates in ``assoc`` for that.
    :param anyone_sighting: also count what *any* other player had in range as known
    :return: evidence per player id
    """
    if len(tr.t) < 2:  # noqa: PLR2004
        return {}
    team = team if team is not None else np.zeros_like(assoc)
    disp, moving = _headings(tr, config)
    shifts = null_shifts(tr, config)
    knowledge = _Knowledge(tr, config, disp, moving, assoc) if team.any() or anyone_sighting else None
    died = deaths(tr, config)
    result = {}
    for i, player_id in enumerate(tr.player_ids):
        ev = BeelineEvidence(moving_s=float(moving[start:, i].sum() * tr.dt))
        if ev.moving_s == 0:
            result[player_id] = ev
            continue
        shared = knowledge.shared(i, team[i], anyone_sighting=anyone_sighting) if knowledge is not None else None
        common = {"heading": disp[:, i], "moving": moving[:, i], "assoc_row": assoc[i], "count_from": start}
        found = _beeline_episodes(tr, config, i, others=tr.pos, shared_known=shared, died=died, **common)
        ev.episodes, ev.start_distance_sum_m = len(found), float(sum(e.start_distance_m for e in found))
        if shifts:
            null = [
                len(
                    _beeline_episodes(
                        tr,
                        config,
                        i,
                        others=_shifted(tr.pos, s),
                        # Shift the others' knowledge with them: who knew where the phantom target was.
                        shared_known=_shifted(shared, s) if shared is not None else None,
                        died=_shifted(died, s),
                        **common,
                    )
                )
                for s in shifts
            ]
            ev.null_episodes = float(np.mean(null))
        result[player_id] = ev
    return result


class _Knowledge:
    """
    Who had whom in sight, and who was heading for whom, at every moment; built once per window.

    ``shared(i, clanmates)`` answers "did a clanmate (or, optionally, an independent spotter) have the target in range,
    or was a clanmate already heading for it, shortly before?" for every target and moment.

    An independent spotter is someone who had the target in their own range from a distance (further than
    ``team_meet_radius_m``) and is not the target's companion or clanmate (``assoc``). Without that, a target's own
    friends, who always "see" it, would excuse anyone raiding the group.
    """

    def __init__(
        self, tr: Trajectories, config: ScoringConfig, disp: np.ndarray, moving: np.ndarray, assoc: np.ndarray
    ) -> None:
        n = len(tr.player_ids)
        self.tr = tr
        self.share = tr.steps(config.team_shared_awareness_s)
        self.sight = np.zeros((n, len(tr.t), n), dtype=bool)  # sight[m, t, k]: m had k in range
        self.spot = np.zeros((n, len(tr.t), n), dtype=bool)  # spot[m, t, k]: independently, from a distance
        self.pursue = np.zeros((n, len(tr.t), n), dtype=bool)  # pursue[m, t, k]: m was heading straight for k
        for m in range(n):
            rel = tr.pos - tr.pos[:, m : m + 1, :]
            dist = np.hypot(rel[..., 0], rel[..., 1])
            h = disp[:, m]
            with np.errstate(invalid="ignore", divide="ignore"):
                self.sight[m] = dist <= tr.awareness[:, m][:, None]
                self.spot[m] = self.sight[m] & (dist > config.team_meet_radius_m) & ~assoc[m][None, :]
                cos = (rel[..., 0] * h[:, None, 0] + rel[..., 1] * h[:, None, 1]) / (dist * np.hypot(*h.T)[:, None])
                self.pursue[m] = moving[:, m][:, None] & (cos >= config.beeline_cos)
            self.sight[m, :, m] = False
            self.spot[m, :, m] = False
            self.pursue[m, :, m] = False
        self.spot_count = self.spot.sum(axis=0)

    def shared(self, i: int, clanmates: np.ndarray, *, anyone_sighting: bool) -> np.ndarray:
        """
        (T, P): whether others knew where each player was, now or shortly before.

        :param i: the approaching player (their own knowledge is handled separately)
        :param clanmates: (P,) the player's clanmates: their sightings and their pursuits count
        :param anyone_sighting: also count independent spotters' sightings. Not their pursuits: among strangers,
            "somebody was heading that way" means nothing (with enough players someone always is), so joining a hunt
            is a relay only between clanmates.
        :return: the shared-knowledge mask
        """
        clanmates = clanmates.copy()
        clanmates[i] = False
        known = self.sight[clanmates].any(axis=0) | self.pursue[clanmates].any(axis=0)
        if anyone_sighting:
            known |= self.spot_count - self.spot[i] > 0
        return _recently(known, self.share)


def beeline_episodes(tr: Trajectories, config: ScoringConfig, player_id: str) -> list[BeelineEpisode]:
    """
    One player's beeline episodes, for showing to a human (the alert image), not for scoring.

    :param tr: trajectories
    :param config: scoring config
    :param player_id: the player
    :return: the episodes (grid indices into ``tr.t``; ``target`` indexes ``tr.player_ids``)
    :raises KeyError: if the player is not in ``tr``
    """
    if player_id not in tr.player_ids:
        raise KeyError(player_id)
    if len(tr.t) < 2:  # noqa: PLR2004
        return []
    i = tr.player_ids.index(player_id)
    disp, moving = _headings(tr, config)
    assoc = associates(tr, config)
    return _beeline_episodes(
        tr, config, i, others=tr.pos, heading=disp[:, i], moving=moving[:, i], assoc_row=assoc[i], count_from=0
    )


def _beeline_episodes(
    tr: Trajectories,
    config: ScoringConfig,
    i: int,
    *,
    others: np.ndarray,
    heading: np.ndarray,
    moving: np.ndarray,
    assoc_row: np.ndarray,
    count_from: int,
    shared_known: np.ndarray | None = None,
    died: np.ndarray | None = None,
) -> list[BeelineEpisode]:
    """The beeline episodes of player ``i`` towards ``others``."""
    rel = others - tr.pos[:, i : i + 1, :]
    dist = np.hypot(rel[..., 0], rel[..., 1])
    heading_len = np.hypot(heading[:, 0], heading[:, 1])
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = (rel[..., 0] * heading[:, None, 0] + rel[..., 1] * heading[:, None, 1]) / (dist * heading_len[:, None])
        aligned = moving[:, None] & (cos >= config.beeline_cos)
    # The player's own range, as it was at each moment (it changes when they respawn as another class).
    aware = tr.awareness[:, i]
    aligned[:, i] = False
    aligned[:, assoc_row] = False
    with np.errstate(invalid="ignore"):
        near_recent = _recently(dist <= aware[:, None], tr.steps(config.near_lookback_s))
    if shared_known is not None:
        near_recent |= shared_known
    min_steps = tr.steps(config.beeline_min_duration_s)
    tail = tr.steps(config.beeline_arrive_grace_s)
    bridge = tr.steps(config.beeline_bridge_s)

    episodes: list[BeelineEpisode] = []
    for j in np.flatnonzero(aligned.any(axis=0)):
        for start, end in _runs(_bridge(aligned[:, j], bridge)):
            if start < count_from or near_recent[start, j]:
                continue
            with np.errstate(invalid="ignore"):
                far_steps = int(np.sum(dist[start : end + 1, j] > aware[start : end + 1]))
            if far_steps < min_steps:
                continue
            d0 = dist[start, j]
            if not (aware[start] < d0 <= config.beeline_max_start_m):
                continue
            with np.errstate(invalid="ignore"):
                reached = np.flatnonzero(dist[start : end + tail + 1, j] <= config.beeline_arrive_m)
            if len(reached) == 0:
                continue
            episode = BeelineEpisode(start, start + int(reached[0]), int(j), float(d0))
            if not _explained_otherwise(
                tr, config, i, episode, others=others, heading=heading, moving=moving, died=died
            ):
                episodes.append(episode)
    return _merge(episodes)


def tip_evidence(
    tr: Trajectories, config: ScoringConfig, assoc: np.ndarray, shifts: list[int], start: int = 0
) -> dict[Pair, PairEvidence]:
    """
    Count tips: far approaches by one player to a target that another player had in sight just before.

    For every approach that the player's own senses do not explain, each other player who *spotted* the target in
    the preceding ``team_shared_awareness_s`` scores a tip for the pair: had it within their own range, but was not
    with it (further than ``team_meet_radius_m``, and not a travelling companion of it). Without that condition,
    anyone approaching a player would score tips with that player's own friends. The null does the same with the
    other player's trajectory shifted in time. Teammates tip each other far more often than chance.

    :param tr: trajectories
    :param config: scoring config
    :param assoc: (P, P) associates (never targets)
    :param shifts: time shifts (grid steps) for the null
    :param start: count only approaches starting at this grid index or later
    :return: tip counts per pair (meetup fields zero)
    """
    n = len(tr.player_ids)
    if len(tr.t) < 2:  # noqa: PLR2004
        return {}
    disp, moving = _headings(tr, config)
    share = tr.steps(config.team_shared_awareness_s)
    observed = np.zeros((n, n))
    null = np.zeros((n, n))
    for i in range(n):
        episodes = _beeline_episodes(
            tr, config, i, others=tr.pos, heading=disp[:, i], moving=moving[:, i], assoc_row=assoc[i], count_from=start
        )
        for e in episodes:
            lo = max(0, e.start - share)
            observed[i] += _spotted(tr, config, tr.pos, target=e.target, lo=lo, hi=e.start)
            for s in shifts:
                shifted = np.roll(tr.pos, s, axis=0)
                null[i] += _spotted(tr, config, shifted, target=e.target, lo=lo, hi=e.start) / len(shifts)
            # Not tips: the approacher itself, the target, and the target's travelling companions.
            excluded = assoc[e.target].copy()
            excluded[[i, e.target]] = True
            observed[i, excluded] = 0
            null[i, excluded] = 0
    result = {}
    for i in range(n):
        for j in range(i + 1, n):
            tips, expected = observed[i, j] + observed[j, i], null[i, j] + null[j, i]
            if tips or expected:
                pe = PairEvidence(tips=int(tips), null_tips=float(expected))
                result[pair_key(tr.player_ids[i], tr.player_ids[j])] = pe
    return result


def _explained_otherwise(
    tr: Trajectories,
    config: ScoringConfig,
    i: int,
    e: BeelineEpisode,
    *,
    others: np.ndarray,
    heading: np.ndarray,
    moving: np.ndarray,
    died: np.ndarray | None,
) -> bool:
    """Whether something other than knowing where the target was explains the approach."""
    before = tr.steps(config.beeline_turn_lookback_s)
    then = e.start - before
    if then >= 0 and moving[then] and _heading_for(tr, config, i, heading, then=then, place=e.arrival):
        return True  # already on the way there before lining up on the target
    # The target came to the player: it moved towards where the player was when the episode started.
    towards_player = (tr.pos[e.start, i] - others[e.start, e.target]) / e.start_distance_m
    approach = float(np.dot(others[e.arrival, e.target] - others[e.start, e.target], towards_player))
    if approach > config.beeline_max_target_approach * e.start_distance_m:
        return True
    # Went to meet up with them, not to hunt them.
    return died is not None and _reunion(tr, config, i, e.target, others=others, died=died, arrival=e.arrival)


def _reunion(
    tr: Trajectories, config: ScoringConfig, i: int, j: int, *, others: np.ndarray, died: np.ndarray, arrival: int
) -> bool:
    """Whether, soon after arriving, the two were peacefully together for a while: a reunion, not a hunt."""
    end = min(len(tr.t), arrival + tr.steps(config.reunion_window_s) + 1)
    rel = others[arrival:end, j] - tr.pos[arrival:end, i]
    with np.errstate(invalid="ignore"):
        together = np.hypot(rel[:, 0], rel[:, 1]) <= config.team_meet_radius_m
    peaceful = together & ~died[arrival:end, i] & ~died[arrival:end, j]
    return any(e - s + 1 >= tr.steps(config.team_meet_min_s) for s, e in _runs(peaceful))


def _heading_for(
    tr: Trajectories, config: ScoringConfig, i: int, heading: np.ndarray, *, then: int, place: int
) -> bool:
    """Whether player ``i`` was, at ``then``, already heading for where they are at ``place``."""
    earlier = heading[then]
    to_place = tr.pos[place, i] - tr.pos[then, i]
    norm = float(np.hypot(*earlier) * np.hypot(*to_place))
    return norm > 0 and float(np.dot(earlier, to_place)) / norm >= config.beeline_cos


def _spotted(
    tr: Trajectories, config: ScoringConfig, positions: np.ndarray, *, target: int, lo: int, hi: int
) -> np.ndarray:
    """(P,): whether each player (at ``positions``) spotted ``target`` during ``[lo, hi]``: in range, not with it."""
    rel = positions[lo : hi + 1] - tr.pos[lo : hi + 1, target : target + 1, :]
    dist = np.hypot(rel[..., 0], rel[..., 1])
    with np.errstate(invalid="ignore"):
        spotted = (dist <= tr.awareness[lo : hi + 1]) & (dist > config.team_meet_radius_m)
    return np.asarray(spotted.any(axis=0), dtype=float)


def _merge(episodes: list[BeelineEpisode]) -> list[BeelineEpisode]:
    """Merge episodes that overlap in time (walking up to a group is one beeline), keeping the first target."""
    merged: list[BeelineEpisode] = []
    for e in sorted(episodes, key=lambda e: (e.start, e.arrival)):
        if merged and e.start <= merged[-1].arrival:
            prev = merged[-1]
            merged[-1] = BeelineEpisode(prev.start, max(prev.arrival, e.arrival), prev.target, prev.start_distance_m)
        else:
            merged.append(e)
    return merged


def spawn_episodes(tr: Trajectories, config: ScoringConfig, assoc: np.ndarray, start: int = 0) -> list[SpawnEpisode]:
    """
    Time from each spawn to first contact with a non-group player.

    A spawn is a player appearing after the start of the window, a jump of more than ``respawn_jump_m`` in one
    grid step, or a class change. Appearances at the very start of the window are left out: that player may
    have spawned long before. An episode ends at contact (within ``contact_m``), or is censored when the player
    leaves, respawns or the window ends. Episodes observed for less than ``ttc_min_observed_s`` are dropped.

    :param tr: trajectories
    :param config: scoring config
    :param assoc: (P, P) associate matrix
    :param start: count only spawns from this grid index on
    :return: all episodes on this server
    """
    if len(tr.t) < 2:  # noqa: PLR2004
        return []
    present = tr.present
    step = np.full(present.shape, np.nan)
    step[1:] = np.hypot(*(tr.pos[1:] - tr.pos[:-1]).transpose(2, 0, 1))
    min_steps = tr.steps(config.ttc_min_observed_s)
    episodes = []
    for i, player_id in enumerate(tr.player_ids):
        d = pairwise_distances(tr.pos, i)
        d[:, i] = np.nan
        d[:, assoc[i]] = np.nan
        with np.errstate(invalid="ignore"):
            contact = np.nanmin(np.where(np.isnan(d), np.inf, d), axis=1) <= config.contact_m
        classes = tr.classes[:, i] if tr.classes.size else np.full(len(tr.t), None, dtype=object)
        with np.errstate(invalid="ignore"):
            jumped = step[:, i] > config.respawn_jump_m
        spawned = present[:, i] & (~np.roll(present[:, i], 1) | jumped | (classes != np.roll(classes, 1)))
        spawned[0] = False
        for run_start, run_end in _runs(present[:, i]):
            # A presence run may contain several lives (respawn by teleport or class change).
            cuts = (np.flatnonzero(spawned[run_start : run_end + 1]) + run_start).tolist()
            bounds = [*cuts, run_end + 1]
            for s, nxt in zip(cuts, bounds[1:], strict=False):
                if s < start:
                    continue
                hits = np.flatnonzero(contact[s:nxt])
                if len(hits):
                    episodes.append(SpawnEpisode(player_id, classes[s], float(hits[0] * tr.dt), censored=False))
                elif nxt - s >= min_steps:
                    episodes.append(SpawnEpisode(player_id, classes[s], float((nxt - s) * tr.dt), censored=True))
    return episodes


def ambush_evidence(
    tr: Trajectories, config: ScoringConfig, assoc: np.ndarray, start: int = 0
) -> dict[str, AmbushEvidence]:
    """
    How often a player's waits end with the arrival of someone who was out of range when the wait began.

    A wait is a run of at least ``ambush_min_wait_s`` below ``stationary_speed_mps``. It is a hit if a non-group
    player who was beyond the waiting player's awareness range of the waiting spot at the start comes within
    ``ambush_radius_m`` of it before the wait ends (plus ``ambush_grace_s``) **and dies shortly after**
    (``team_fight_grace_s``). An ambush is an attack: an arrival not followed by a kill (a regroup, clanmates
    joining at a carcass, someone passing by) says nothing either way, so it is not counted.

    :param tr: trajectories
    :param config: scoring config
    :param assoc: (P, P) associate matrix
    :param start: count only waits that begin at or after this grid index
    :return: evidence per player id
    """
    if len(tr.t) < 2:  # noqa: PLR2004
        return {}
    step = np.full(tr.pos.shape[:2], np.nan)
    step[:-1] = np.hypot(*(tr.pos[1:] - tr.pos[:-1]).transpose(2, 0, 1)) / tr.dt
    # Smooth over three steps so one jittery sample does not split a wait.
    kernel = np.ones(3) / 3
    speed = np.apply_along_axis(lambda s: np.convolve(s, kernel, mode="same"), 0, step)
    with np.errstate(invalid="ignore"):
        stationary = speed < config.stationary_speed_mps
    min_steps = tr.steps(config.ambush_min_wait_s)
    grace = tr.steps(config.ambush_grace_s)
    shifts = null_shifts(tr, config)
    died = deaths(tr, config)

    result = {}
    for i, player_id in enumerate(tr.player_ids):
        waits = [(s, e) for s, e in _runs(stationary[:, i]) if e - s + 1 >= min_steps and s >= start]
        ev = AmbushEvidence(waits=len(waits))
        if waits:
            common = {"waits": waits, "grace": grace, "assoc_row": assoc[i]}
            ev.hits = _ambush_hits(tr, config, i, others=tr.pos, died=died, **common)
            if shifts:
                null = [
                    _ambush_hits(tr, config, i, others=_shifted(tr.pos, s), died=_shifted(died, s), **common)
                    for s in shifts
                ]
                ev.null_hits = float(np.mean(null))
        result[player_id] = ev
    return result


def _ambush_hits(
    tr: Trajectories,
    config: ScoringConfig,
    i: int,
    *,
    others: np.ndarray,
    waits: list[tuple[int, int]],
    grace: int,
    assoc_row: np.ndarray,
    died: np.ndarray,
) -> int:
    hits = 0
    for start, end in waits:
        spot = np.nanmean(tr.pos[start : end + 1, i], axis=0)
        stop = min(end + grace, len(tr.t) - 1)
        rel = others[start : stop + 1] - spot
        d = np.hypot(rel[..., 0], rel[..., 1])
        d[:, i] = np.nan
        d[:, assoc_row] = np.nan
        with np.errstate(invalid="ignore"):
            far_at_start = d[0] > tr.awareness[start, i]
            arrived = np.any(d <= config.ambush_radius_m, axis=0)
        for j in np.flatnonzero(far_at_start & arrived):
            at = start + int(np.flatnonzero(d[:, j] <= config.ambush_radius_m)[0])
            if died[at, j]:
                hits += 1
                break
    return hits
