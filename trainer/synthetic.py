"""
Synthetic ESP for measuring recall without labels.

There are no labelled cheaters, so recall is measured on made-up ones: a real player's track in which some segments
are replaced by a pursuit of the nearest player nobody could have told them about, at a realistic speed, the way
``tools.sim.behaviours.BeelineCheater`` plays. Everything else (the other players, the map, the rest of the track)
stays real, so the injected player is as hard to tell apart from honest players as real data makes it.

A second benchmark runs the simulator's worlds (``tools/sim``), whose players have known honest and cheating
archetypes, including honest behaviours that come close to the line (hunters, clans answering calls).
"""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from server.scoring.config import ScoringConfig
from server.scoring.game import GameProfile, load_profile
from server.scoring.trajectories import Trajectories, from_frame
from server.training.leakage import FeatureSpec, known_to_others
from tools.sim.scenarios import archetypes, clan_world, mixed_world, record

ESP_SUFFIX = "#esp"


@dataclass(frozen=True)
class Injection:
    """How a synthetic ESP user plays."""

    active_fraction: float = 0.5  # share of 10-minute phases spent hunting with ESP
    phase_s: float = 600.0
    speed_range_mps: tuple[float, float] = (4.0, 8.0)  # clipped from the player's own median moving speed
    contact_m: float = 30.0
    engage_s: float = 45.0  # standing still at the target (the fight)
    give_up_s: float = 300.0
    heading_noise: float = 0.1  # radians per grid step
    gap_s: float = 60.0  # after a hunting phase the player is absent this long, then their real track resumes


def inject_pursuits(
    tr: Trajectories,
    players: Sequence[int],
    rng: np.random.Generator,
    spec: FeatureSpec,
    config: ScoringConfig,
    *,
    injection: Injection | None = None,
) -> Trajectories:
    """
    Replace hunting phases of some players' tracks with ESP pursuits of unseen players.

    Targets are the nearest players nobody could have told the hunter about (beyond their awareness range plus
    ``spec.sight_margin_m``, within ``spec.far_m``, not a friend, not recently spotted by anyone independent: see
    :func:`server.training.leakage.known_to_others`); the other players' tracks are untouched. After a hunting phase
    the player drops out for ``gap_s`` (as after a death) and their real track resumes, so there is no teleport
    inside a life.

    :param tr: real trajectories
    :param players: indices of the players to turn into ESP users
    :param rng: random generator
    :param spec: feature spec
    :param config: scoring config (what counts as known)
    :param injection: how the synthetic cheater plays
    :return: trajectories with the modified tracks
    """
    inj = injection or Injection()
    pos = tr.pos.copy()
    known = known_to_others(tr, spec, config, players)
    for p in players:
        _inject_one(tr, pos, p, rng=rng, spec=spec, inj=inj, known=known[p])
    return replace(tr, pos=pos)


def _inject_one(
    tr: Trajectories,
    pos: np.ndarray,
    p: int,
    *,
    rng: np.random.Generator,
    spec: FeatureSpec,
    inj: Injection,
    known: np.ndarray,
) -> None:
    dt, n_t = tr.dt, len(tr.t)
    real = tr.pos[:, p]
    step_speed = np.hypot(*np.diff(real, axis=0).T) / dt if n_t > 1 else np.zeros(0)
    moving = step_speed[step_speed > 1.5]  # noqa: PLR2004
    speed = float(np.clip(np.median(moving) if len(moving) else 6.0, *inj.speed_range_mps))
    phase = tr.steps(inj.phase_s)
    gap = tr.steps(inj.gap_s)
    t = 0
    while t < n_t:
        end = min(t + phase, n_t)
        if rng.random() >= inj.active_fraction or np.isnan(real[t, 0]):
            t = end
            continue
        stop = _hunt(tr, pos, p, span=(t, end), speed=speed, rng=rng, spec=spec, inj=inj, known=known)
        pos[stop : min(stop + gap, n_t), p] = np.nan
        t = stop + gap


def _hunt(
    tr: Trajectories,
    pos: np.ndarray,
    p: int,
    *,
    span: tuple[int, int],
    speed: float,
    rng: np.random.Generator,
    spec: FeatureSpec,
    inj: Injection,
    known: np.ndarray,
) -> int:
    """Pursue unseen players during ``span`` (or until the player leaves); return where the phase ended."""
    dt = tr.dt
    start, end = span
    here = pos[start, p].copy()
    target: int | None = None
    chase = engage = 0
    t = start
    while t < end and not np.isnan(tr.pos[t, p, 0]):
        pos[t, p] = here
        if engage > 0:
            engage -= 1
            if engage == 0:
                target = None
            t += 1
            continue
        if target is not None and (np.isnan(tr.pos[t, target, 0]) or chase * dt > inj.give_up_s):
            target = None
        if target is None:
            target = _unseen_target(tr, t, p, here, spec=spec, known=known[t])
            chase = 0
            if target is None:
                return t
        offset = tr.pos[t, target] - here
        d = float(np.hypot(*offset))
        if d <= inj.contact_m:
            engage = tr.steps(inj.engage_s)
            continue
        angle = np.arctan2(offset[1], offset[0]) + rng.normal(0.0, inj.heading_noise)
        here = here + min(speed * dt, d) * np.array([np.cos(angle), np.sin(angle)])
        chase += 1
        t += 1
    return t


def _unseen_target(
    tr: Trajectories, t: int, p: int, here: np.ndarray, *, spec: FeatureSpec, known: np.ndarray
) -> int | None:
    d = np.hypot(*(tr.pos[t] - here).T)
    d[p] = np.nan
    with np.errstate(invalid="ignore"):
        ok = (d > tr.awareness[t, p] + spec.sight_margin_m) & (d <= spec.far_m) & ~known
    if not ok.any():
        return None
    return int(np.nanargmin(np.where(ok, d, np.nan)))


@dataclass(frozen=True)
class SimServer:
    """One simulated server for the benchmark: trajectory windows and the hidden archetype of every player."""

    name: str
    windows: list[tuple[Trajectories, float]]  # (trajectories, count_from)
    archetypes: dict[str, str]


def sim_frames(
    kind: str, seed: int, hours: float, *, cheaters: tuple[str, ...] | None = None
) -> tuple[pd.DataFrame, dict[str, str]]:
    """
    Record a simulated server.

    :param kind: ``mixed`` (:func:`tools.sim.scenarios.mixed_world`) or ``clan``
        (:func:`tools.sim.scenarios.clan_world`)
    :param seed: rng seed
    :param hours: simulated time
    :param cheaters: the cheater archetypes of a mixed world (one player each); the scenario's default if None
    :return: samples (metres, seconds from 0) and player id -> archetype
    :raises ValueError: for an unknown kind
    """
    if kind not in ("mixed", "clan"):
        raise ValueError(f"unknown world kind {kind!r} (mixed or clan)")
    if kind == "clan":
        world = clan_world(seed)
    else:
        world = mixed_world(seed) if cheaters is None else mixed_world(seed, cheaters=cheaters)
    frame = record(world, hours * 3600)
    return frame, archetypes(world)


def sim_servers(
    kinds: Sequence[str],
    seeds: Sequence[int],
    hours: float,
    config: ScoringConfig,
    profile: GameProfile | None = None,
) -> Iterator[SimServer]:
    """
    Simulated servers cut into scoring windows (with 30 minutes of context, as the scoring job reads them).

    :param kinds: world kinds, see :func:`sim_frames`
    :param seeds: rng seeds (one server per kind and seed)
    :param hours: simulated time per server
    :param config: scoring config (window and context lengths)
    :param profile: game profile; the default if None
    :yield: one server at a time
    """
    profile = profile or load_profile()
    window = config.window_minutes * 60
    context = config.context_minutes * 60
    for kind in kinds:
        for seed in seeds:
            frame, arch = sim_frames(kind, seed, hours)
            windows = []
            for start in np.arange(0.0, hours * 3600 - 1, window):
                part = frame[(frame["t"] >= start - context) & (frame["t"] < start + window)]
                windows.append((from_frame(part, f"sim-{kind}-{seed}", config, profile), float(start)))
            yield SimServer(f"{kind}-{seed}", windows, arch)
