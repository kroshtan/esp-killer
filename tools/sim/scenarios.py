"""
Ready-made worlds and a recorder, for scoring tests and the evaluation report.

A recording is a long-format DataFrame with the same columns :func:`server.scoring.trajectories.from_frame`
expects (``t`` seconds, ``player_id``, ``x``/``y`` metres, ``dino_class``), sampled every few seconds the way the
agent polls.
"""

from collections.abc import Callable

import pandas as pd

from server.scoring.game import GameProfile, load_profile
from tools.sim.behaviours import AmbushCheater, BeelineCheater, Behaviour, GroupMember, Hunter, Roamer
from tools.sim.world import DINO_CLASSES, World

HONEST = ("roamer", "waterhole", "camper", "hunter", "group")
CHEATERS = ("beeline_cheater", "ambush_cheater", "subtle_cheater")

# Cheater archetype -> behaviour, given a walking speed and the class's awareness range ("far" means beyond it).
CHEATER_BEHAVIOURS: dict[str, Callable[[float, float], Behaviour]] = {
    "beeline_cheater": lambda speed, aware: BeelineCheater(speed_mps=speed, awareness_m=aware),
    "ambush_cheater": lambda _speed, aware: AmbushCheater(awareness_m=aware),
    # Hunts with ESP only 40% of the time and plays normally otherwise.
    "subtle_cheater": lambda speed, aware: BeelineCheater(speed_mps=speed, awareness_m=aware, active_fraction=0.4),
}


def mixed_world(
    seed: int,
    *,
    roamers: int = 6,
    waterhole: int = 5,
    campers: int = 3,
    hunters: int = 3,
    groups: int = 2,
    group_size: int = 3,
    cheaters: tuple[str, ...] = CHEATERS,
    half_size_m: float = 3000.0,
    profile: GameProfile | None = None,
) -> World:
    """
    A server with every honest archetype and the given cheaters.

    Half the players join during the first 20 minutes, and everyone dies now and then, so there are spawns for
    the time-to-contact feature. Waterholes are shared by everyone who likes points of interest.

    :param seed: rng seed
    :param roamers: players wandering between random points
    :param waterhole: players who mostly go between the waterholes
    :param campers: players who rest for long periods at waterholes
    :param hunters: carnivores who chase anyone they can see
    :param groups: number of groups (friends on voice chat)
    :param group_size: players per group
    :param cheaters: archetypes from :data:`CHEATERS` to add, one player each
    :param half_size_m: half the map width
    :param profile: game profile for per-class awareness (hunters and cheaters see with their class's range)
    :return: the world, at time 0
    :raises ValueError: for an unknown cheater archetype
    """
    profile = profile or load_profile()
    unknown = set(cheaters) - set(CHEATER_BEHAVIOURS)
    if unknown:
        raise ValueError(f"unknown cheater archetype(s): {sorted(unknown)}")
    world = World(half_size_m=half_size_m, seed=seed, death_rate_per_s=1 / 1500)
    pois = [world.random_point(margin_m=600.0) for _ in range(4)]

    for _ in range(roamers):
        world.add_player(Roamer(speed_mps=_speed(world)), archetype="roamer", join_at=_join_time(world))
    for _ in range(waterhole):
        world.add_player(
            Roamer(speed_mps=_speed(world), pois=pois, poi_bias=0.9), archetype="waterhole", join_at=_join_time(world)
        )
    for _ in range(campers):
        camper = Roamer(speed_mps=_speed(world), pois=pois, poi_bias=0.8, pause_s=(300.0, 900.0))
        world.add_player(camper, archetype="camper", join_at=_join_time(world))
    for _ in range(hunters):
        cls = _dino_class(world)
        hunter = Hunter(awareness_m=profile.awareness_m(cls), speed_mps=_speed(world), pois=pois, poi_bias=0.3)
        world.add_player(hunter, archetype="hunter", join_at=_join_time(world), dino_class=cls)
    for _ in range(groups):
        leader = world.add_player(Roamer(speed_mps=5.0, pois=pois, poi_bias=0.5), archetype="group")
        for _ in range(group_size - 1):
            world.add_player(GroupMember(leader=leader, speed_mps=5.0), archetype="group", pos=leader.pos)
    for kind in cheaters:
        cls = _dino_class(world)
        behaviour = CHEATER_BEHAVIOURS[kind](_speed(world), profile.awareness_m(cls))
        world.add_player(behaviour, archetype=kind, join_at=_join_time(world), dino_class=cls)
    return world


def _dino_class(world: World) -> str:
    return str(world.rng.choice(DINO_CLASSES))


def _join_time(world: World) -> float:
    """Half the players are there from the start, the rest join during the first 20 minutes."""
    return float(world.rng.uniform(0, 1200)) if world.rng.random() < 0.5 else 0.0  # noqa: PLR2004


def _speed(world: World) -> float:
    return float(world.rng.uniform(4.0, 8.0))


def record(world: World, duration_s: float, *, dt: float = 1.0, sample_every_s: float = 3.0) -> pd.DataFrame:
    """
    Run the world and sample every present player's position, as the agent's polls would.

    :param world: the world to run
    :param duration_s: how long to run it
    :param dt: simulation step
    :param sample_every_s: poll interval
    :return: samples with columns ``t``, ``player_id``, ``x``, ``y`` (metres), ``dino_class``
    """
    rows: list[tuple[float, str, float, float, str]] = []
    every = max(1, round(sample_every_s / dt))
    for step in range(round(duration_s / dt)):
        if step % every == 0:
            rows.extend(
                (world.time_s, p.player_id, float(p.pos[0]), float(p.pos[1]), p.dino_class)
                for p in world.players
                if p.present
            )
        world.step(dt)
    return pd.DataFrame(rows, columns=["t", "player_id", "x", "y", "dino_class"])


def archetypes(world: World) -> dict[str, str]:
    """
    The (hidden) archetype of every player, for evaluation.

    :param world: the world
    :return: player id -> archetype
    """
    return {p.player_id: p.archetype for p in world.players}
