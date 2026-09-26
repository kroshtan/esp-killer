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
    tr: Trajectories, config: ScoringConfig, assoc: np.ndarray, start: int = 0
) -> dict[str, BeelineEvidence]:
    """
    Count each player's beelines: heading persistently for a player who was out of range, and arriving.

    An episode is a run of grid steps in which the player moves and heads within ``beeline_cos`` of the bearing to
    the same target, for at least ``beeline_min_duration_s`` while the target is out of range, where:

    * the target was beyond ``awareness_m`` at the start and had not been within it for ``near_lookback_s``;
    * the player stays lined up until the target comes within ``beeline_arrive_m`` (normally the awareness range;
      the run may end up to ``beeline_arrive_grace_s`` before that). What happens after sighting is not evidence;
    * the player *turned* onto the target: ``beeline_turn_lookback_s`` earlier they were not already heading for
      the meeting place;
    * the target did not come to the player (``beeline_max_target_approach``).

    Chasing someone you saw recently, walking into an ambush or up to someone resting on your route, meeting
    someone head-on, being hunted, and heading for a group mate (``assoc``) therefore never count. Overlapping
    episodes (walking up to several people at once) count as one.

    :param tr: trajectories
    :param config: scoring config
    :param assoc: (P, P) associate matrix from :func:`associates`
    :param start: count only moving time and episodes from this grid index on
    :return: evidence per player id
    """
    if len(tr.t) < 2:  # noqa: PLR2004
        return {}
    disp, moving = _headings(tr, config)
    shifts = null_shifts(tr, config)
    result = {}
    for i, player_id in enumerate(tr.player_ids):
        ev = BeelineEvidence(moving_s=float(moving[start:, i].sum() * tr.dt))
        if ev.moving_s == 0:
            result[player_id] = ev
            continue
        common = {"heading": disp[:, i], "moving": moving[:, i], "assoc_row": assoc[i], "count_from": start}
        found = _beeline_episodes(tr, config, i, others=tr.pos, **common)
        ev.episodes, ev.start_distance_sum_m = len(found), float(sum(e.start_distance_m for e in found))
        if shifts:
            null = [len(_beeline_episodes(tr, config, i, others=_shifted(tr.pos, s), **common)) for s in shifts]
            ev.null_episodes = float(np.mean(null))
        result[player_id] = ev
    return result


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
) -> list[BeelineEpisode]:
    """The beeline episodes of player ``i`` towards ``others``."""
    rel = others - tr.pos[:, i : i + 1, :]
    dist = np.hypot(rel[..., 0], rel[..., 1])
    heading_len = np.hypot(heading[:, 0], heading[:, 1])
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = (rel[..., 0] * heading[:, None, 0] + rel[..., 1] * heading[:, None, 1]) / (dist * heading_len[:, None])
        aligned = moving[:, None] & (cos >= config.beeline_cos)
        near = dist <= config.awareness_m
    aligned[:, i] = False
    aligned[:, assoc_row] = False
    near_recent = _recently(near, tr.steps(config.near_lookback_s))
    min_steps = tr.steps(config.beeline_min_duration_s)
    tail = tr.steps(config.beeline_arrive_grace_s)
    bridge = tr.steps(config.beeline_bridge_s)
    before = tr.steps(config.beeline_turn_lookback_s)

    episodes: list[BeelineEpisode] = []
    for j in np.flatnonzero(aligned.any(axis=0)):
        for start, end in _runs(_bridge(aligned[:, j], bridge)):
            if start < count_from or near_recent[start, j]:
                continue
            with np.errstate(invalid="ignore"):
                far_steps = int(np.sum(dist[start : end + 1, j] > config.awareness_m))
            if far_steps < min_steps:
                continue
            d0 = dist[start, j]
            if not (config.awareness_m < d0 <= config.beeline_max_start_m):
                continue
            with np.errstate(invalid="ignore"):
                reached = np.flatnonzero(dist[start : end + tail + 1, j] <= config.beeline_arrive_m)
            if len(reached) == 0:
                continue
            arrival = start + int(reached[0])
            if start >= before and moving[start - before]:
                earlier = heading[start - before]
                to_meeting = tr.pos[arrival, i] - tr.pos[start - before, i]
                norm = float(np.hypot(*earlier) * np.hypot(*to_meeting))
                if norm > 0 and float(np.dot(earlier, to_meeting)) / norm >= config.beeline_cos:
                    continue  # already on the way there before lining up on the target
            # How far the target moved towards where the player was when the episode started.
            approach = float(np.dot(others[arrival, j] - others[start, j], -rel[start, j] / d0))
            if approach > config.beeline_max_target_approach * d0:
                continue
            episodes.append(BeelineEpisode(start, arrival, int(j), float(d0)))
    return _merge(episodes)


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
    player who was beyond ``awareness_m`` of the waiting spot at the start comes within ``ambush_radius_m`` of it
    before the wait ends (plus ``ambush_grace_s``).

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

    result = {}
    for i, player_id in enumerate(tr.player_ids):
        waits = [(s, e) for s, e in _runs(stationary[:, i]) if e - s + 1 >= min_steps and s >= start]
        ev = AmbushEvidence(waits=len(waits))
        if waits:
            common = {"waits": waits, "grace": grace, "assoc_row": assoc[i]}
            ev.hits = _ambush_hits(tr, config, i, others=tr.pos, **common)
            if shifts:
                ev.null_hits = float(
                    np.mean([_ambush_hits(tr, config, i, others=_shifted(tr.pos, s), **common) for s in shifts])
                )
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
            far_at_start = d[0] > config.awareness_m
            arrived = np.any(d <= config.ambush_radius_m, axis=0)
        if np.any(far_at_start & arrived):
            hits += 1
    return hits
