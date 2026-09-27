"""
Information leakage: how much a player's moves follow players nobody could have told them about.

The rules in ``server/scoring/features.py`` look for specific shapes of ESP use (beelines, ambushes). This is the
general, label-free version of the same idea. Every 15 s, each player's next move (the direction of their next 20 s,
relative to the heading they last moved in, in 12 bins, or "stays put") is scored by two models:

* **model A** knows what the player could legitimately know: their own movement, where they are on the map
  (waterholes, trails), time since spawn, class, the players within their awareness range, friends (players they
  spend time with, and inferred clanmates), and players out of range that are **known** by the same rule the beeline
  scoring uses: seen by the player in the last ``near_lookback_s``, or within range of a clanmate or of an independent
  spotter (not the target's companion, from further than ``team_meet_radius_m``) in the last
  ``team_shared_awareness_s``, right next to a player the player can see or knows about, or about to come into view
  before the move is over (the move may be a reaction to them);
* **model B** is A plus a residual that sees only the remaining, **hidden** players: the nearest few within
  ``far_m``, how many there are, and how steadily the player has been heading for the nearest of them.

Both are step-selection models (as in animal-movement ecology): they score every candidate move, so "the move towards
a hidden player is the one taken" is one pattern shared by all directions, learnable from few examples. A's columns
are a strict prefix of B's per-sample columns and A's candidate rows are built from them alone, so A cannot use
anything about hidden players; that is the property the method rests on (tests/unit/test_leakage.py).

A player's evidence is not the raw gain ``log p_B(y) - log p_A(y)`` but the gain over the **time-shifted null**,
the same idea as in the rule-based scoring: the gain with the hidden players replaced by the same players 10, 20 and
30 minutes earlier, averaged. For a player whose moves do not depend on where hidden players are *now*, real and
phantom players are interchangeable, so the terms average zero whatever B learned (where people go, B being more or
less confident than A, map structure: all appear in both and cancel). For an ESP user, the real players predict their
moves and the phantoms do not. Per player, the terms are summed and divided by their standard error (from blocks of
consecutive samples), giving a z that grows with playing time for an ESP user and not for anyone else.

Everything is measured relative to the player's own heading, so nothing depends on the map's orientation except the
map position and absolute direction columns (map structure).

This module is shared by the trainer (``trainer/``) and the backend, which calls :meth:`LeakageModel.player_stats`
once per scoring window, stores the additive :class:`LeakageStats`, and turns their sum over the evidence horizon into
a :class:`LeakageScore` with :meth:`LeakageModel.rescore` (or calls :meth:`LeakageModel.score` for one window).
"""

import hashlib
import io
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import lightgbm as lgb
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from server.scoring.config import ScoringConfig
from server.scoring.features import _headings, _Knowledge, _recently, associates, null_shifts
from server.scoring.teams import team_matrix
from server.scoring.trajectories import Trajectories
from server.training.schema import CURRENT_MODEL, MODELS
from server.training.store import ObjectStore

# Probabilities are floored before taking logs, so one confident miss cannot dominate a player's mean.
_P_FLOOR = 1e-4
_SLOT_FEATURES = ("dist", "bearing", "speed", "heading_to_me")
MODEL_A_FILE = "model_a.txt"
MODEL_B_FILE = "model_b.txt"
METADATA_FILE = "metadata.json"


class FeatureSpec(BaseModel):
    """How samples and features are built. Stored with the model: training and scoring must agree exactly."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    heading_s: float = Field(15.0, gt=0)  # velocity and heading: displacement over the past this long
    horizon_s: float = Field(20.0, gt=0)  # target: displacement over the next this long
    stride_s: float = Field(15.0, gt=0)  # one sample per player this often
    # Also sample players who are standing still (relative to the heading they last moved in): where someone goes
    # when they set off again, after a rest or a fight, is one of the most telling decisions a player makes.
    include_still: bool = True
    bins: int = Field(12, ge=2)  # direction classes of the target; class ``bins`` means "stops"
    min_speed_mps: float = Field(1.5, gt=0)  # below this over the horizon counts as not moving
    respawn_jump_m: float = Field(300.0, gt=0)  # a jump this far in one grid step is a respawn
    far_m: float = Field(3000.0, gt=0)  # unseen players further away than this are ignored
    # Players up to this far beyond the awareness range count as visible. Ranges are approximate (terrain, sound,
    # polling every few seconds), and a hunter who turns towards someone at 320 m is not cheating; without the margin
    # that edge alone is enough to give honest hunters a leakage score.
    sight_margin_m: float = Field(50.0, ge=0)
    n_visible: int = Field(3, ge=1)
    n_known: int = Field(2, ge=1)
    n_hidden: int = Field(3, ge=1)
    spawn_cap_s: float = Field(600.0, gt=0)  # "time since spawn" saturates here (a window's context is longer)
    classes: tuple[str, ...] = ()  # class vocabulary (categorical codes), fixed at training time

    @property
    def n_classes(self) -> int:
        """
        Number of target classes: the direction bins plus "stops".

        :return: the count
        """
        return self.bins + 1


class Calibration(BaseModel):
    """Maps a player's z to a 0-1 sub-score: 0 up to ``z_floor``, 1 from ``z_full``, linear in between."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    z_floor: float = 3.0
    z_full: float = 7.0
    min_samples: int = Field(120, ge=1)  # fewer samples than this (about 30 minutes moving) always score 0
    block_samples: int = Field(8, ge=1)  # consecutive samples per block for the standard error (2 minutes)

    def score(self, z: float, n: int) -> float:
        """
        The sub-score for a statistic.

        :param z: the player's z
        :param n: the player's number of samples
        :return: 0-1
        """
        if n < self.min_samples or not np.isfinite(z):
            return 0.0
        return float(np.clip((z - self.z_floor) / max(self.z_full - self.z_floor, 1e-9), 0.0, 1.0))


def feature_names(spec: FeatureSpec) -> tuple[list[str], list[str]]:
    """
    The feature columns: model A's, and the columns only model B has (appended after A's).

    :param spec: feature spec
    :return: (A's columns, B's extra columns)
    """
    own = ["x", "y", "heading_sin", "heading_cos", "speed", "speed_lag1", "speed_lag2", "turn", "turn_lag1"]
    own += ["still_s", "since_spawn", "awareness", "dino_class"]

    def slots(prefix: str, n: int) -> list[str]:
        return [f"{prefix}{i}_{f}" for i in range(n) for f in _SLOT_FEATURES]

    a = [*own, *slots("vis", spec.n_visible), "n_visible", *slots("friend", 1), *slots("known", spec.n_known)]
    a += ["n_known", "track_known_60", "track_known_300"]
    b = [*slots("hidden", spec.n_hidden), "n_hidden", "track_hidden_60", "track_hidden_300"]
    return a, b


# Columns that depend on the map's orientation (map structure); every other column is rotation-invariant.
ABSOLUTE_COLUMNS = ("x", "y", "heading_sin", "heading_cos")


@dataclass
class Samples:
    """
    Feature rows. ``x[:, :n_a]`` are model A's columns; model B uses all of them.

    ``x_null[i]`` holds B's extra columns (``x[:, n_a:]``) recomputed against the ``i``-th time-shifted null.
    """

    x: np.ndarray  # (n, features) float32
    y: np.ndarray  # (n,) int: direction bin, or ``bins`` for "stops"
    player: np.ndarray  # (n,) object: player id
    t: np.ndarray  # (n,) float: sample time
    n_a: int
    x_null: np.ndarray  # (shifts, n, features - n_a) float32

    def __len__(self) -> int:
        """Number of samples."""
        return len(self.y)

    def subset(self, mask: np.ndarray) -> "Samples":
        """
        The rows where ``mask`` is true (or at the given indices).

        :param mask: bool mask or index array
        :return: the subset
        """
        return Samples(self.x[mask], self.y[mask], self.player[mask], self.t[mask], self.n_a, self.x_null[:, mask])

    def null(self, i: int) -> "Samples":
        """
        The same samples with B's columns taken from the ``i``-th null.

        :param i: null index
        :return: the samples
        """
        x = np.hstack([self.x[:, : self.n_a], self.x_null[i]])
        return Samples(x, self.y, self.player, self.t, self.n_a, self.x_null[:0])

    @staticmethod
    def empty(n_a: int, n_b: int, shifts: int) -> "Samples":
        """
        No samples.

        :param n_a: number of model A columns
        :param n_b: number of B's extra columns
        :param shifts: number of nulls
        :return: the empty set
        """
        return Samples(
            np.empty((0, n_a + n_b), np.float32),
            np.empty(0, int),
            np.empty(0, object),
            np.empty(0),
            n_a,
            np.empty((shifts, 0, n_b), np.float32),
        )

    @staticmethod
    def concat(parts: Sequence["Samples"], n_a: int, n_b: int, shifts: int) -> "Samples":
        """
        Stack sample sets.

        :param parts: the sets (may be empty)
        :param n_a: number of model A columns
        :param n_b: number of B's extra columns
        :param shifts: number of nulls
        :return: one set
        """
        if not parts:
            return Samples.empty(n_a, n_b, shifts)
        return Samples(
            np.concatenate([p.x for p in parts]),
            np.concatenate([p.y for p in parts]),
            np.concatenate([p.player for p in parts]),
            np.concatenate([p.t for p in parts]),
            n_a,
            np.concatenate([p.x_null for p in parts], axis=1),
        )


def _wrap(angle: np.ndarray) -> np.ndarray:
    return np.asarray((angle + np.pi) % (2 * np.pi) - np.pi)


@dataclass
class _Context:
    """Per-trajectory arrays shared by every player's features."""

    k: int  # heading steps
    horizon: int  # target steps
    life_start: np.ndarray  # (T, P) grid index where the current life began; -1 when absent
    vel: np.ndarray  # (T, P, 2) velocity over the past ``k`` steps, NaN if not continuous
    # (T, P) the direction the player last moved in during this life (their reference heading), NaN if they have not
    # moved yet; and the grid index when that was.
    ref: np.ndarray
    last_moved: np.ndarray
    knowledge: _Knowledge  # who had whom in sight (the rule-based scoring's definition of legitimate knowledge)
    team: np.ndarray  # (P, P) bool: clanmates
    friends: np.ndarray  # (P, P) bool: associates (time spent together in this window) or clanmates
    # shift -> (T, P, P) float32: [t, k, j] = 1 if player k (now) is within ``team_meet_radius_m`` of player j
    # (shifted by ``shift``); built on first use
    companions: dict[int, np.ndarray] = field(default_factory=dict)


def _context(tr: Trajectories, spec: FeatureSpec, config: ScoringConfig, team: np.ndarray) -> _Context:
    n_t, n_p = tr.present.shape
    k = tr.steps(spec.heading_s)
    pos, present = tr.pos, tr.present
    step = np.full((n_t, n_p), np.inf)
    if n_t > 1:
        step[1:] = np.hypot(*np.moveaxis(pos[1:] - pos[:-1], -1, 0))
    prev_present = np.vstack([np.zeros((1, n_p), bool), present[:-1]])
    with np.errstate(invalid="ignore"):
        new_life = present & (~prev_present | (step > spec.respawn_jump_m))
    life_start = np.maximum.accumulate(np.where(new_life, np.arange(n_t)[:, None], -1), axis=0)
    life_start = np.where(present, life_start, -1)

    vel = np.full_like(pos, np.nan)
    if n_t > k:
        vel[k:] = (pos[k:] - pos[:-k]) / (k * tr.dt)
        broken = (life_start[k:] > np.arange(k, n_t)[:, None] - k) | ~present[k:]
        vel[k:][broken] = np.nan
    with np.errstate(invalid="ignore"):
        moving = np.hypot(vel[..., 0], vel[..., 1]) >= spec.min_speed_mps
    last_moved = np.maximum.accumulate(np.where(moving, np.arange(n_t)[:, None], -1), axis=0)
    has_ref = (last_moved >= 0) & (last_moved >= life_start) & present
    angle = np.arctan2(vel[..., 1], vel[..., 0])
    ref = np.where(has_ref, np.take_along_axis(angle, np.maximum(last_moved, 0), axis=0), np.nan)

    friends = associates(tr, config) | team
    disp, moving_ahead = _headings(tr, config)
    return _Context(
        k=k,
        horizon=tr.steps(spec.horizon_s),
        life_start=life_start,
        vel=vel,
        ref=ref,
        last_moved=np.where(has_ref, last_moved, -1),
        knowledge=_Knowledge(tr, config, disp, moving_ahead, friends),
        team=team,
        friends=friends,
    )


def build_samples(
    tr: Trajectories,
    spec: FeatureSpec,
    config: ScoringConfig,
    *,
    count_from: float | None = None,
    players: Iterable[int] | None = None,
    teams: Mapping[str, int] | None = None,
) -> Samples:
    """
    Feature rows for every present player every ``stride_s``.

    A sample needs the player continuously present (no respawn) from three headings back to one horizon ahead, and
    to have moved at some point in this life (for a reference heading). Its target is the direction of the next
    ``horizon_s`` of movement relative to that heading, in ``bins`` classes, or ``bins`` if the player (nearly)
    stays put.

    Every sample also carries B's columns computed against the **null**: the other players shifted in time by each
    of ``config.null_shifts_s`` (as for the beeline rule), with what others knew about them shifted along. Phantoms
    are where people go, but not where they are now.

    :param tr: one server's trajectories
    :param spec: feature spec
    :param config: scoring config: who counts as a friend (``associate_*``), and what counts as legitimately known
        (``near_lookback_s``, ``team_shared_awareness_s``, ``team_meet_radius_m``), exactly as for the beeline rule;
        and the null shifts
    :param count_from: only samples at or after this time (earlier data is context)
    :param players: player indices to build samples for; all by default
    :param teams: inferred clans (player id -> clan number): clanmates' sightings count as known
    :return: the samples
    """
    names_a, names_b = feature_names(spec)
    n_a, n_b = len(names_a), len(names_b)
    n_t = len(tr.t)
    shifts = null_shifts(tr, config) if n_t else []
    if n_t == 0 or not tr.player_ids or not shifts:
        return Samples.empty(n_a, n_b, len(config.null_shifts_s))
    ctx = _context(tr, spec, config, team_matrix(tr, teams))
    first = 3 * ctx.k
    if count_from is not None:
        first = max(first, int(np.searchsorted(tr.t, count_from - 1e-6)))
    candidates = np.arange(first, n_t - ctx.horizon)
    candidates = candidates[candidates % tr.steps(spec.stride_s) == 0]
    class_codes = {c: float(i) for i, c in enumerate(spec.classes)}
    # Pad to the configured number of shifts (a short window uses fewer): repeat the ones that fit.
    shifts = [shifts[i % len(shifts)] for i in range(len(config.null_shifts_s))]
    parts = [
        _player_samples(tr, spec, config, ctx, p, candidates=candidates, class_codes=class_codes, shifts=shifts)
        for p in (range(len(tr.player_ids)) if players is None else players)
    ]
    return Samples.concat([s for s in parts if s is not None], n_a, n_b, len(shifts))


def _player_samples(
    tr: Trajectories,
    spec: FeatureSpec,
    config: ScoringConfig,
    ctx: _Context,
    p: int,
    *,
    candidates: np.ndarray,
    class_codes: Mapping[str, float],
    shifts: Sequence[int],
) -> Samples | None:
    k, h, dt, pos = ctx.k, ctx.horizon, tr.dt, tr.pos
    ls = ctx.life_start[:, p]
    ok = (ls[candidates] >= 0) & (ls[candidates] <= candidates - 3 * k)
    ok &= (ls[candidates + h] >= 0) & (ls[candidates + h] <= candidates)
    ok &= ~np.isnan(ctx.ref[candidates, p])
    ts = candidates[ok]
    speed0 = np.hypot(*ctx.vel[ts, p].T)
    if not spec.include_still:
        ts, speed0 = ts[speed0 >= spec.min_speed_mps], speed0[speed0 >= spec.min_speed_mps]
    if len(ts) == 0:
        return None
    heading, h1, h2 = ctx.ref[ts, p], ctx.ref[ts - k, p], ctx.ref[ts - 2 * k, p]
    speed1, speed2 = np.hypot(*ctx.vel[ts - k, p].T), np.hypot(*ctx.vel[ts - 2 * k, p].T)

    future = pos[ts + h, p] - pos[ts, p]
    future_speed = np.hypot(future[:, 0], future[:, 1]) / (h * dt)
    rel_dir = _wrap(np.arctan2(future[:, 1], future[:, 0]) - heading)
    width = 2 * np.pi / spec.bins
    y = np.floor((rel_dir + np.pi + width / 2) / width).astype(int) % spec.bins
    y = np.where(future_speed < spec.min_speed_mps, spec.bins, y)

    cls = tr.classes[ts, p]
    own = [
        pos[ts, p, 0],
        pos[ts, p, 1],
        np.sin(heading),
        np.cos(heading),
        speed0,
        speed1,
        speed2,
        _wrap(heading - h1),
        _wrap(h1 - h2),
        np.minimum((ts - ctx.last_moved[ts, p]) * dt, spec.spawn_cap_s),
        np.minimum((ts - ls[ts]) * dt, spec.spawn_cap_s),
        tr.awareness[ts, p],
        np.array([class_codes.get(c, np.nan) if c is not None else np.nan for c in cls]),
    ]
    view = _view(tr, config, ctx, p, spec=spec)
    legit = view.visible | view.friend | view.known
    track_known, _ = _tracking(view, spec, ctx, p)
    windows = [tr.steps(60.0), tr.steps(300.0)]
    a_part = [
        *own,
        _slots(view, ctx.vel, view.visible, spec.n_visible, heading=heading, ts=ts),
        view.visible[ts].sum(axis=1),
        _slots(view, ctx.vel, view.friend, 1, heading=heading, ts=ts),
        _slots(view, ctx.vel, view.known, spec.n_known, heading=heading, ts=ts),
        view.known[ts].sum(axis=1),
        *(_trailing_mean(track_known, w)[ts] for w in windows),
    ]
    x_a = np.column_stack(a_part).astype(np.float32)
    x_b = _hidden_columns(view, spec, ctx, p, heading=heading, ts=ts, windows=windows)
    phantoms = (_view(tr, config, ctx, p, spec=spec, shift=s, legit=legit) for s in shifts)
    x_null = np.stack([_hidden_columns(v, spec, ctx, p, heading=heading, ts=ts, windows=windows) for v in phantoms])
    return Samples(
        np.hstack([x_a, x_b]),
        y,
        np.full(len(ts), tr.player_ids[p], dtype=object),
        tr.t[ts].astype(float),
        x_a.shape[1],
        x_null,
    )


def _hidden_columns(
    view: "_View",
    spec: FeatureSpec,
    ctx: _Context,
    p: int,
    *,
    heading: np.ndarray,
    ts: np.ndarray,
    windows: Sequence[int],
) -> np.ndarray:
    """Model B's extra columns: the nearest hidden players, how many there are, and how the player tracks them."""
    _, track_hidden = _tracking(view, spec, ctx, p)
    return np.column_stack(
        [
            _slots(view, view.vel, view.hidden, spec.n_hidden, heading=heading, ts=ts),
            view.hidden[ts].sum(axis=1),
            *(_trailing_mean(track_hidden, w)[ts] for w in windows),
        ]
    ).astype(np.float32)


def _slots(
    view: "_View", vel: np.ndarray, mask: np.ndarray, n: int, *, heading: np.ndarray, ts: np.ndarray
) -> np.ndarray:
    """
    The nearest ``n`` players of a set at the sample times ``ts``.

    Per player: distance, bearing relative to the heading, speed, and the angle between their heading and the
    direction towards the player (0: coming straight at them). NaN for empty slots.
    """
    d = view.dist[ts]
    masked = np.where(mask[ts], d, np.inf)
    order = np.argsort(masked, axis=1)[:, :n]
    rel = np.take_along_axis(view.rel[ts], order[:, :, None], axis=1)  # (m, n, 2)
    vj = np.take_along_axis(vel[ts], order[:, :, None], axis=1)
    speed_j = np.hypot(vj[..., 0], vj[..., 1])
    to_me = np.arctan2(-rel[..., 1], -rel[..., 0])
    with np.errstate(invalid="ignore"):
        heading_to_me = np.where(speed_j > 0.3, _wrap(np.arctan2(vj[..., 1], vj[..., 0]) - to_me), np.nan)  # noqa: PLR2004
    chosen = np.stack(
        [
            np.take_along_axis(d, order, axis=1),
            _wrap(np.arctan2(rel[..., 1], rel[..., 0]) - heading[:, None]),
            speed_j,
            heading_to_me,
        ],
        axis=-1,
    )
    chosen[~np.isfinite(np.take_along_axis(masked, order, axis=1))] = np.nan
    if chosen.shape[1] < n:  # fewer players on the server than slots
        chosen = np.concatenate([chosen, np.full((len(ts), n - chosen.shape[1], 4), np.nan)], axis=1)
    return chosen.reshape(len(ts), -1)


@dataclass
class _View:
    """One player's view of everyone else at every grid step. Each present player is in at most one set."""

    rel: np.ndarray  # (T, P, 2) position of each player relative to this one
    vel: np.ndarray  # (T, P, 2) their velocities
    dist: np.ndarray  # (T, P), NaN where either is absent
    visible: np.ndarray  # (T, P) within the player's awareness range
    friend: np.ndarray  # (T, P) an associate or clanmate, not visible
    known: np.ndarray  # (T, P) out of range but legitimately known: seen recently, or spotted by someone independent
    hidden: np.ndarray  # (T, P) within ``far_m``, and nobody could have told the player where they were


def _companions(tr: Trajectories, ctx: _Context, shift: int, radius: float) -> np.ndarray:
    if shift not in ctx.companions:
        others = (np.roll(tr.pos, shift, axis=0) if shift else tr.pos).astype(np.float32)
        pos = tr.pos.astype(np.float32)
        d = np.hypot(pos[:, :, None, 0] - others[:, None, :, 0], pos[:, :, None, 1] - others[:, None, :, 1])
        with np.errstate(invalid="ignore"):
            ctx.companions[shift] = (d <= radius).astype(np.float32)
    return ctx.companions[shift]


def _view(
    tr: Trajectories,
    config: ScoringConfig,
    ctx: _Context,
    p: int,
    *,
    spec: FeatureSpec,
    shift: int = 0,
    legit: np.ndarray | None = None,
) -> _View:
    """
    Split everyone else into what player ``p`` could see, could legitimately know, and could not have known.

    "Known" is exactly what the beeline rule treats as known: the player's own sightings within ``near_lookback_s``,
    plus what a clanmate or an independent spotter had in range within ``team_shared_awareness_s``. Only ``hidden``
    carries information nobody could legitimately have had.

    A player out of range but next to someone the player can see or knows about (within ``team_meet_radius_m``)
    counts as known too: chasing one member of a group moves you towards the rest of it. So does a player who comes
    into view within the next ``horizon_s``: the move being predicted may be a reaction to seeing them.

    With ``shift``, everyone else is replaced by their time-shifted phantom, and what others knew about them is
    shifted along (the null); ``legit`` is then the real view's visible, friend and known players, whose companions
    count as known.
    """
    others = np.roll(tr.pos, shift, axis=0) if shift else tr.pos
    rel = others - tr.pos[:, p][:, None, :]
    dist = np.hypot(rel[..., 0], rel[..., 1])
    present = ~np.isnan(dist)
    present[:, p] = False
    with np.errstate(invalid="ignore"):
        sight = (dist <= tr.awareness[:, p][:, None] + spec.sight_margin_m) & present
    friend = ctx.friends[p][None, :] & present & ~sight
    with np.errstate(invalid="ignore"):
        within = present & (dist <= spec.far_m) & ~sight & ~friend
    # Seen recently, or about to be seen before the move being predicted is over.
    own = _recently(sight, tr.steps(config.near_lookback_s)) | _recently(sight[::-1], tr.steps(spec.horizon_s))[::-1]
    shared = ctx.knowledge.shared(p, ctx.team[p], anyone_sighting=True)
    if shift:
        shared = np.roll(shared, shift, axis=0)
    known = within & (own | shared)
    if legit is None:
        legit = sight | friend | known
    near = np.matmul(legit.astype(np.float32)[:, None, :], _companions(tr, ctx, shift, config.team_meet_radius_m))
    known |= within & (near[:, 0, :] > 0)
    vel = np.roll(ctx.vel, shift, axis=0) if shift else ctx.vel
    return _View(rel, vel, dist, sight, friend, known, within & ~known)


def _tracking(view: _View, spec: FeatureSpec, ctx: _Context, p: int) -> tuple[np.ndarray, np.ndarray]:
    """
    How the player heads for others, at every grid step.

    :param view: the player's view of the others
    :param spec: feature spec
    :param ctx: shared arrays
    :param p: the player
    :return: the cosine between the player's heading and the bearing to the nearest player they could see or know
        about, and to the nearest hidden one; NaN when not moving or nobody is in the set
    """
    rows = np.arange(len(view.dist))
    bearing = np.arctan2(view.rel[..., 1], view.rel[..., 0])
    with np.errstate(invalid="ignore"):
        moving = np.hypot(*ctx.vel[:, p].T) >= spec.min_speed_mps
    out = []
    for mask in (view.visible | view.friend | view.known, view.hidden):
        masked = np.where(mask, view.dist, np.inf)
        nearest = np.argmin(masked, axis=1)
        ok = moving & np.isfinite(masked[rows, nearest])
        out.append(np.where(ok, np.cos(bearing[rows, nearest] - ctx.ref[:, p]), np.nan))
    return out[0], out[1]


def _trailing_mean(values: np.ndarray, steps: int) -> np.ndarray:
    """Mean of the non-NaN values over the last ``steps`` + 1 rows (NaN if there are none)."""
    ok = ~np.isnan(values)
    total = np.cumsum(np.where(ok, values, 0.0))
    count = np.cumsum(ok)
    total[steps + 1 :] -= total[: -(steps + 1)].copy()
    count[steps + 1 :] -= count[: -(steps + 1)].copy()
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.asarray(np.where(count > 0, total / count, np.nan))


def known_to_others(
    tr: Trajectories,
    spec: FeatureSpec,
    config: ScoringConfig,
    players: Sequence[int],
    teams: Mapping[str, int] | None = None,
) -> dict[int, np.ndarray]:
    """
    For each of ``players``: at every grid step, which others someone could have told them about.

    That is a friend or clanmate, or a player a clanmate or independent spotter had in range recently (the same
    rule as for "known" players in the features). What the player saw themselves is not included: it depends on
    where they are.

    :param tr: trajectories
    :param spec: feature spec
    :param config: scoring config
    :param players: player indices
    :param teams: inferred clans (player id -> clan number)
    :return: player index -> (T, P) bool
    """
    if len(tr.t) == 0:
        return {p: np.zeros((0, len(tr.player_ids)), bool) for p in players}
    ctx = _context(tr, spec, config, team_matrix(tr, teams))
    return {p: ctx.knowledge.shared(p, ctx.team[p], anyone_sighting=True) | ctx.friends[p][None, :] for p in players}


@dataclass(frozen=True)
class LeakageStats:
    """
    Sufficient statistics of one player's leakage terms, additive over windows and servers.

    The statistic is the sum of the per-sample terms over its standard deviation. Consecutive samples are correlated
    (a chase lasts minutes), so the variance is estimated from blocks of consecutive samples, and never taken smaller
    than for independent samples, so a short streak cannot look significant.
    """

    n: int = 0
    blocks: int = 0
    sum_gain: float = 0.0  # sum of per-sample terms (see :func:`leakage_terms`)
    sum_sq: float = 0.0  # sum of their squares
    block_gg: float = 0.0  # sum over blocks of (block sum)^2
    block_gm: float = 0.0  # ... of block sum * block size
    block_mm: float = 0.0  # ... of block size^2

    def __add__(self, other: "LeakageStats") -> "LeakageStats":
        """Statistics of both sets of samples."""
        return LeakageStats(
            self.n + other.n,
            self.blocks + other.blocks,
            self.sum_gain + other.sum_gain,
            self.sum_sq + other.sum_sq,
            self.block_gg + other.block_gg,
            self.block_gm + other.block_gm,
            self.block_mm + other.block_mm,
        )

    @staticmethod
    def of(gains: np.ndarray, block_samples: int) -> "LeakageStats":
        """
        Statistics of one player's terms, in time order.

        :param gains: per-sample terms, in time order
        :param block_samples: consecutive samples per block
        :return: the statistics
        """
        g = np.asarray(gains, float)
        if len(g) == 0:
            return LeakageStats()
        starts = np.arange(0, len(g), block_samples)
        sums = np.add.reduceat(g, starts)
        sizes = np.diff(np.append(starts, len(g))).astype(float)
        return LeakageStats(
            len(g),
            len(sums),
            float(g.sum()),
            float(np.square(g).sum()),
            float(np.square(sums).sum()),
            float((sums * sizes).sum()),
            float(np.square(sizes).sum()),
        )

    @property
    def gain(self) -> float:
        """
        Mean term per sample, in nats: how much better the real hidden players predict the moves taken than phantoms.

        :return: the mean, 0 without samples
        """
        return self.sum_gain / self.n if self.n else 0.0

    @property
    def z(self) -> float:
        """
        The sum of the terms over its standard deviation.

        :return: the z, 0 with fewer than two blocks
        """
        if self.blocks < 2:  # noqa: PLR2004
            return 0.0
        mean = self.gain
        resid = max(self.block_gg - 2 * mean * self.block_gm + mean**2 * self.block_mm, 0.0)
        var_blocks = self.blocks / (self.blocks - 1) * resid
        var_iid = max(self.sum_sq - self.n * mean**2, 0.0) * self.n / max(self.n - 1, 1)
        sd = float(np.sqrt(max(var_iid, var_blocks)))
        return self.sum_gain / sd if sd > 0 else 0.0

    def to_dict(self) -> dict[str, float]:
        """
        As a JSON-able mapping (for storing per-window evidence).

        :return: field name -> value
        """
        return {k: float(v) for k, v in self.__dict__.items()}

    @staticmethod
    def from_dict(data: Mapping[str, float]) -> "LeakageStats":
        """
        Inverse of :meth:`to_dict`.

        :param data: the mapping
        :return: the statistics
        """
        return LeakageStats(
            int(data["n"]),
            int(data["blocks"]),
            *(float(data[k]) for k in ("sum_gain", "sum_sq", "block_gg", "block_gm", "block_mm")),
        )


@dataclass(frozen=True)
class LeakageScore:
    gain: float  # mean term per sample, nats
    n: int  # samples
    z: float
    score_0_1: float
    summary: str  # for people: what the numbers mean
    stats: LeakageStats = field(repr=False)


def aggregate(players: np.ndarray, t: np.ndarray, gains: np.ndarray, block_samples: int) -> dict[str, LeakageStats]:
    """
    Per-player statistics of per-sample terms.

    :param players: (n,) player id per sample
    :param t: (n,) sample times (to order each player's samples)
    :param gains: (n,) terms
    :param block_samples: consecutive samples per block
    :return: player id -> statistics
    """
    out: dict[str, LeakageStats] = {}
    if len(players) == 0:
        return out
    order = np.lexsort((t, players))
    ps, gs = players[order], gains[order]
    bounds = np.flatnonzero(ps[1:] != ps[:-1]) + 1
    for chunk_p, chunk_g in zip(np.split(ps, bounds), np.split(gs, bounds), strict=True):
        out[str(chunk_p[0])] = LeakageStats.of(chunk_g, block_samples)
    return out


def stable_fold(player_id: str, folds: int, salt: str = "") -> int:
    """
    A player's cross-fitting fold, the same on every run and every server.

    :param player_id: pseudonymised player id
    :param folds: number of folds
    :param salt: distinguishes independent splits (e.g. the validation split inside a fold)
    :return: 0 .. folds - 1
    """
    return int.from_bytes(hashlib.sha256(f"{salt}:{player_id}".encode()).digest()[:8], "big") % folds


# Candidate columns: model A sees the direction (relative and absolute: map structure); B's residual only whether the
# candidate is "stop", so that it cannot re-learn anything about directions that does not involve hidden players.
A_CANDIDATE_COLUMNS = ("cand_angle", "cand_stop", "cand_sin", "cand_cos")
B_CANDIDATE_COLUMNS = ("cand_stop",)


def _off(name: str) -> str:
    return name.removesuffix("_bearing") + "_off" if name.endswith("_bearing") else name


def row_names(spec: FeatureSpec) -> tuple[list[str], list[str]]:
    """
    Columns of the candidate rows of model A and of B's residual (see :func:`candidate_rows`).

    :param spec: feature spec
    :return: (A's row columns, B's residual row columns); every ``*_bearing`` becomes ``*_off``, the angle between
        that player and the candidate direction
    """
    names_a, names_b = feature_names(spec)
    return [*A_CANDIDATE_COLUMNS, *map(_off, names_a)], [*B_CANDIDATE_COLUMNS, *map(_off, names_b)]


def candidate_angles(spec: FeatureSpec) -> np.ndarray:
    """
    Centre of each direction bin, relative to the heading (bin ``bins // 2`` is straight on).

    :param spec: feature spec
    :return: (bins,) radians
    """
    return np.asarray(np.arange(spec.bins) * (2 * np.pi / spec.bins) - np.pi)


def _expand(x: np.ndarray, names: Sequence[str], spec: FeatureSpec, candidate: Sequence[str]) -> np.ndarray:
    n, c = len(x), spec.n_classes
    cand_angle = np.tile(np.append(candidate_angles(spec), np.nan), n)  # the stop candidate has no direction
    columns: dict[str, np.ndarray] = {"cand_angle": cand_angle, "cand_stop": np.tile(np.eye(c)[-1], n)}
    if "cand_sin" in candidate:
        absolute = np.repeat(np.arctan2(x[:, names.index("heading_sin")], x[:, names.index("heading_cos")]), c)
        columns["cand_sin"], columns["cand_cos"] = np.sin(absolute + cand_angle), np.cos(absolute + cand_angle)
    out = np.empty((n * c, len(candidate) + len(names)), np.float32)
    for i, name in enumerate(candidate):
        out[:, i] = columns[name]
    out[:, len(candidate) :] = np.repeat(x, c, axis=0)
    for i, name in enumerate(names):
        if name.endswith("_bearing"):
            col = len(candidate) + i
            out[:, col] = _wrap(out[:, col] - cand_angle)
    return out


def candidate_rows(samples: Samples, spec: FeatureSpec) -> tuple[np.ndarray, np.ndarray]:
    """
    One row per sample and candidate move (every direction bin, then "stop"), for the step-selection models.

    The models score each candidate move a player could have made, as in step-selection analysis of animal
    movement, instead of predicting a class: "the candidate pointing at a player nobody could see is the one taken"
    is then one pattern shared by every direction, learnable from a handful of events, instead of one per class.

    Model B is model A plus a residual that sees *only* the hidden players' columns (and whether the candidate is
    "stop"): whatever B predicts differently from A is, by construction, due to players nobody could have seen.

    :param samples: per-sample features
    :param spec: feature spec
    :return: (A's rows from A's columns only, B's residual rows from B's extra columns only), each
        ``(n * n_classes, columns)`` float32, sample-major
    """
    names_a, names_b = feature_names(spec)
    n_a = samples.n_a
    return (
        _expand(samples.x[:, :n_a], names_a, spec, A_CANDIDATE_COLUMNS),
        _expand(samples.x[:, n_a:], names_b, spec, B_CANDIDATE_COLUMNS),
    )


def _log_softmax(scores: np.ndarray) -> np.ndarray:
    scores = scores - scores.max(axis=1, keepdims=True)
    log_p = scores - np.log(np.exp(scores).sum(axis=1, keepdims=True))
    return np.asarray(np.maximum(log_p, np.log(_P_FLOOR)))


def leakage_terms(
    log_pa: np.ndarray, log_pb: np.ndarray, log_pb_null: Sequence[np.ndarray], y: np.ndarray
) -> np.ndarray:
    """
    Per-sample leakage terms: the gain of the move taken, over the same gain against the time-shifted null.

    The gain is ``log p_B(y) - log p_A(y)``: how much better the move actually taken is predicted once the hidden
    players are known. The null gain is the same with the hidden players replaced by their time-shifted phantoms,
    averaged over the shifts. For a player whose moves do not depend on where hidden players are *now*, real and
    phantom players are interchangeable, so the terms average zero whatever B learned: generic effects (where people
    go, B being more or less confident than A) appear in both and cancel. Only following real, current, unknowable
    positions remains.

    :param log_pa: (n, C) model A log-probabilities
    :param log_pb: (n, C) model B log-probabilities
    :param log_pb_null: per shift, (n, C) model B log-probabilities against the null
    :param y: (n,) moves taken
    :return: (n,) terms, in nats
    """
    if len(y) == 0:
        return np.empty(0)
    rows = np.arange(len(y))
    null = np.mean([lp[rows, y] for lp in log_pb_null], axis=0) if log_pb_null else log_pa[rows, y]
    return np.asarray(log_pb[rows, y] - null)


def _scores(booster: lgb.Booster, rows: np.ndarray, n: int, spec: FeatureSpec) -> np.ndarray:
    return np.asarray(booster.predict(rows, raw_score=True, num_threads=0)).reshape(n, spec.n_classes)


def predict_terms(booster_a: lgb.Booster, booster_b: lgb.Booster, samples: Samples, spec: FeatureSpec) -> np.ndarray:
    """
    Leakage terms of samples under a pair of models (see :func:`leakage_terms`).

    :param booster_a: model A
    :param booster_b: B's residual
    :param samples: feature rows
    :param spec: feature spec
    :return: (n,) terms
    """
    if len(samples) == 0:
        return np.empty(0)
    n = len(samples)
    rows_a, rows_b = candidate_rows(samples, spec)
    score_a = _scores(booster_a, rows_a, n, spec)
    log_pb = _log_softmax(score_a + _scores(booster_b, rows_b, n, spec))
    log_pb_null = [
        _log_softmax(score_a + _scores(booster_b, candidate_rows(samples.null(i), spec)[1], n, spec))
        for i in range(len(samples.x_null))
    ]
    return leakage_terms(_log_softmax(score_a), log_pb, log_pb_null, samples.y)


@dataclass
class LeakageModel:
    spec: FeatureSpec
    calibration: Calibration
    booster_a: lgb.Booster
    booster_b: lgb.Booster
    metadata: dict[str, Any] = field(default_factory=dict)

    def terms(self, samples: Samples) -> np.ndarray:
        """
        Per-sample leakage terms (see :func:`leakage_terms`).

        :param samples: feature rows
        :return: (n,) terms
        """
        return predict_terms(self.booster_a, self.booster_b, samples, self.spec)

    def player_stats(
        self,
        trajectories: Sequence[Trajectories],
        config: ScoringConfig,
        count_from: float | None = None,
        *,
        teams: Mapping[str, int] | None = None,
    ) -> dict[str, LeakageStats]:
        """
        Additive per-player statistics for one scoring window (the part the backend stores per window).

        :param trajectories: one per server, as built by :func:`server.scoring.trajectories.from_frame` (with the
            server's game profile, for awareness ranges)
        :param config: scoring config: friends, what counts as known, and the null shifts (as for the beeline rule)
        :param count_from: only samples at or after this time count (earlier data is context)
        :param teams: inferred clans (player id -> clan number): clanmates' sightings count as known
        :return: player id -> statistics, for every player with at least one sample
        """
        total: dict[str, LeakageStats] = {}
        for tr in trajectories:
            samples = build_samples(tr, self.spec, config, count_from=count_from, teams=teams)
            for pid, stats in aggregate(samples.player, samples.t, self.terms(samples), self.block_samples).items():
                total[pid] = total.get(pid, LeakageStats()) + stats
        return total

    @property
    def block_samples(self) -> int:
        """
        Samples per block for the standard error.

        :return: the block size
        """
        return self.calibration.block_samples

    def rescore(self, stats: LeakageStats) -> LeakageScore:
        """
        Turn (accumulated) statistics into a score with this model's calibration.

        :param stats: the statistics, e.g. summed over the evidence horizon
        :return: the score
        """
        z = stats.z
        score = self.calibration.score(z, stats.n)
        hours = stats.n * self.spec.stride_s / 3600
        summary = (
            f"leakage z {z:.1f} (sub-score {score:.2f}): players nobody could have seen predict this player's moves"
            f" {stats.gain:+.3f} nats per move better than the same players at other times, over {stats.n} moves"
            f" ({hours:.1f} h observed)"
        )
        if stats.n < self.calibration.min_samples:
            summary += f"; too few moves to score (needs {self.calibration.min_samples})"
        return LeakageScore(stats.gain, stats.n, z, score, summary, stats)

    def score(
        self,
        trajectories: Sequence[Trajectories],
        config: ScoringConfig,
        count_from: float | None = None,
        *,
        teams: Mapping[str, int] | None = None,
    ) -> dict[str, LeakageScore]:
        """
        Score every player in one window (for evidence summed over several windows, see :meth:`rescore`).

        :param trajectories: one per server, as built by :func:`server.scoring.trajectories.from_frame`
        :param config: scoring config
        :param count_from: only samples at or after this time count (earlier data is context)
        :param teams: inferred clans (player id -> clan number): clanmates' sightings count as known
        :return: player id -> score
        """
        stats = self.player_stats(trajectories, config, count_from, teams=teams)
        return {pid: self.rescore(s) for pid, s in stats.items()}

    def save(self, store: ObjectStore, version: str) -> None:
        """
        Write the model to ``models/<version>/``. Does not promote it (see :func:`promote`).

        :param store: the object store
        :param version: version name
        """
        prefix = f"{MODELS}/{version}"
        store.put(f"{prefix}/{MODEL_A_FILE}", self.booster_a.model_to_string().encode())
        store.put(f"{prefix}/{MODEL_B_FILE}", self.booster_b.model_to_string().encode())
        meta = {
            **self.metadata,
            "version": version,
            "spec": self.spec.model_dump(mode="json"),
            "calibration": self.calibration.model_dump(mode="json"),
        }
        store.put(f"{prefix}/{METADATA_FILE}", json.dumps(meta, indent=2, sort_keys=True).encode())

    @staticmethod
    def load(store: ObjectStore, version: str) -> "LeakageModel":
        """
        Read a saved model.

        :param store: the object store
        :param version: version name
        :return: the model
        """
        prefix = f"{MODELS}/{version}"
        meta = json.loads(store.get(f"{prefix}/{METADATA_FILE}"))
        spec = FeatureSpec.model_validate(meta.pop("spec"))
        calibration = Calibration.model_validate(meta.pop("calibration"))
        a = lgb.Booster(model_str=store.get(f"{prefix}/{MODEL_A_FILE}").decode())
        b = lgb.Booster(model_str=store.get(f"{prefix}/{MODEL_B_FILE}").decode())
        return LeakageModel(spec, calibration, a, b, meta)

    @staticmethod
    def load_current(store: ObjectStore) -> "LeakageModel | None":
        """
        The promoted model, if any.

        :param store: the object store
        :return: the model, or None if none has been promoted
        """
        version = current_version(store)
        return LeakageModel.load(store, version) if version else None


def current_version(store: ObjectStore) -> str | None:
    """
    The version of the promoted model.

    :param store: the object store
    :return: the version, or None if none has been promoted
    """
    try:
        return str(json.loads(store.get(CURRENT_MODEL))["version"])
    except KeyError:
        return None


def promote(store: ObjectStore, version: str) -> None:
    """
    Make a saved model the current one. Written last, so readers never see a half-written model as current.

    :param store: the object store
    :param version: a version already written with :meth:`LeakageModel.save`
    """
    store.put(CURRENT_MODEL, json.dumps({"version": version}).encode())


def model_size_bytes(model: LeakageModel) -> int:
    """
    Size of both boosters as saved.

    :param model: the model
    :return: bytes
    """
    buf = io.StringIO()
    buf.write(model.booster_a.model_to_string())
    buf.write(model.booster_b.model_to_string())
    return len(buf.getvalue().encode())
