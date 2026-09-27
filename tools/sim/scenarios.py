"""
Ready-made worlds and a recorder, for scoring tests and the evaluation report.

A recording is a long-format DataFrame with the same columns :func:`server.scoring.trajectories.from_frame`
expects (``t`` seconds, ``player_id``, ``x``/``y`` metres, ``dino_class``), sampled every few seconds the way the
agent polls.
"""

from collections.abc import Callable

import numpy as np
import pandas as pd

from server.scoring.game import GameProfile, load_profile
from tools.sim.behaviours import (
    AmbushCheater,
    BeelineCheater,
    Behaviour,
    Clan,
    ClanMember,
    EspClanMember,
    GroupMember,
    Hunter,
    Roamer,
)
from tools.sim.world import DINO_CLASSES, SimPlayer, World

HONEST = ("roamer", "waterhole", "camper", "hunter", "group")
CHEATERS = ("beeline_cheater", "ambush_cheater", "subtle_cheater")
# Clan archetypes (see :func:`clan_world`): honest members, and a member who shares what ESP shows with the clan.
CLAN_HONEST = ("clan_member", "clan_hunter", "clan_spotter")
CLAN_CHEATERS = ("clan_esp",)
SOLO_HONEST = ("roamer", "waterhole", "camper", "hunter")  # honest archetypes that work without a group
PTERANODON = "Pteranodon"

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

    for kind, count in (("roamer", roamers), ("waterhole", waterhole), ("camper", campers), ("hunter", hunters)):
        for _ in range(count):
            _add_solo_honest(world, kind, pois, profile)
    for _ in range(groups):
        leader = world.add_player(Roamer(speed_mps=5.0, pois=pois, poi_bias=0.5), archetype="group")
        for _ in range(group_size - 1):
            world.add_player(GroupMember(leader=leader, speed_mps=5.0), archetype="group", pos=leader.pos)
    for kind in cheaters:
        _add_cheater(world, kind, profile)
    return world


def _add_solo_honest(world: World, kind: str, pois: list[np.ndarray], profile: GameProfile) -> None:
    """Add one honest player of a :data:`SOLO_HONEST` archetype."""
    if kind == "roamer":
        world.add_player(Roamer(speed_mps=_speed(world)), archetype=kind, join_at=_join_time(world))
    elif kind == "waterhole":
        world.add_player(
            Roamer(speed_mps=_speed(world), pois=pois, poi_bias=0.9), archetype=kind, join_at=_join_time(world)
        )
    elif kind == "camper":
        camper = Roamer(speed_mps=_speed(world), pois=pois, poi_bias=0.8, pause_s=(300.0, 900.0))
        world.add_player(camper, archetype=kind, join_at=_join_time(world))
    elif kind == "hunter":
        cls = _dino_class(world)
        hunter = Hunter(awareness_m=profile.awareness_m(cls), speed_mps=_speed(world), pois=pois, poi_bias=0.3)
        world.add_player(hunter, archetype=kind, join_at=_join_time(world), dino_class=cls)
    else:
        raise ValueError(f"unknown solo honest archetype: {kind}")


def _add_cheater(world: World, kind: str, profile: GameProfile) -> None:
    cls = _dino_class(world)
    behaviour = CHEATER_BEHAVIOURS[kind](_speed(world), profile.awareness_m(cls))
    world.add_player(behaviour, archetype=kind, join_at=_join_time(world), dino_class=cls)


def clan_world(
    seed: int,
    *,
    clans: int = 2,
    clan_size: tuple[int, int] = (6, 10),
    include_ptera: bool = True,
    esp_in_clan: bool = True,
    solo_honest: tuple[str, ...] = ("roamer",) * 5 + ("waterhole",) * 4 + ("camper",) * 2 + ("hunter",) * 3,
    solo_cheaters: tuple[str, ...] = ("beeline_cheater",),
    half_size_m: float = 3000.0,
    profile: GameProfile | None = None,
) -> World:
    """
    A server with rival clans, solo honest players and solo cheaters.

    Each clan (team id ``clan-<i>``) has a random size in ``clan_size`` and mixed species. Its members start spread
    over the map (about a third next to a teammate, the rest alone), share a base where they regroup every 20-40
    minutes, and call every stranger they see; about 40% of them (at least one) are hunters who respond to calls
    (``clan_hunter``), the others only roam and call (``clan_member``). The first clan gets a Pteranodon spotter
    (``clan_spotter``) if ``include_ptera``; the last clan gets a member who calls and hunts targets it knows about
    only through ESP (``clan_esp``, a cheater) if ``esp_in_clan``. These special members count towards the clan
    size. Ground truth teams: :func:`teams`; the clan objects: :func:`clans`.

    :param seed: rng seed
    :param clans: number of rival clans
    :param clan_size: inclusive range of the number of members per clan
    :param include_ptera: give the first clan a Pteranodon spotter
    :param esp_in_clan: give the last clan an ESP-using member
    :param solo_honest: archetypes from :data:`SOLO_HONEST` to add, one player each
    :param solo_cheaters: archetypes from :data:`CHEATERS` to add, one player each
    :param half_size_m: half the map width
    :param profile: game profile for per-class awareness
    :return: the world, at time 0
    :raises ValueError: for an unknown archetype or an invalid clan size
    """
    profile = profile or load_profile()
    unknown = (set(solo_cheaters) - set(CHEATER_BEHAVIOURS)) | (set(solo_honest) - set(SOLO_HONEST))
    if unknown:
        raise ValueError(f"unknown archetype(s): {sorted(unknown)}")
    if not 2 <= clan_size[0] <= clan_size[1]:  # noqa: PLR2004
        raise ValueError(f"invalid clan size range: {clan_size}")
    world = World(half_size_m=half_size_m, seed=seed, death_rate_per_s=1 / 1500)
    pois = [world.random_point(margin_m=600.0) for _ in range(4)]

    bases: list[np.ndarray] = []
    for c in range(clans):
        base = _clan_base(world, bases)
        bases.append(base)
        clan = Clan(f"clan-{c}", base, rng=np.random.default_rng(int(world.rng.integers(2**32))))
        size = int(world.rng.integers(clan_size[0], clan_size[1] + 1))
        roles = ["member"] * size
        n_hunters = max(1, round(0.4 * size))
        roles[:n_hunters] = ["hunter"] * n_hunters
        if include_ptera and c == 0:
            roles[-1] = "spotter"
        if esp_in_clan and c == clans - 1:
            roles[0] = "esp"
        members: list[SimPlayer] = []
        for role in roles:
            members.append(_add_clan_member(world, clan, role, members=members, pois=pois, profile=profile))
    for kind in solo_honest:
        _add_solo_honest(world, kind, pois, profile)
    for kind in solo_cheaters:
        _add_cheater(world, kind, profile)
    return world


def _clan_base(world: World, bases: list[np.ndarray]) -> np.ndarray:
    """A base for a new clan, a map half-width from the other clans' bases if a few tries find such a spot."""
    base = world.random_point(margin_m=800.0)
    for _ in range(50):
        if all(float(np.hypot(*(base - b))) >= world.half_size_m for b in bases):
            break
        base = world.random_point(margin_m=800.0)
    return base


def _add_clan_member(
    world: World, clan: Clan, role: str, *, members: list[SimPlayer], pois: list[np.ndarray], profile: GameProfile
) -> SimPlayer:
    """Add a clan member with ``role`` member, hunter, spotter (a Pteranodon) or esp."""
    join_at = _join_time(world)
    # Alone or in pairs: a third start next to a teammate, the others anywhere on the map.
    near = members[int(world.rng.integers(len(members)))] if members and world.rng.random() < 1 / 3 else None
    pos = near.pos + world.rng.normal(0.0, 20.0, size=2) if near is not None else world.random_point()
    if role == "spotter":
        cls = PTERANODON
        behaviour: Behaviour = ClanMember(
            clan,
            awareness_m=profile.awareness_m(cls),
            speed_mps=float(world.rng.uniform(12.0, 15.0)),
            pause_s=(5.0, 30.0),
        )
    else:
        cls = _dino_class(world)
        member = EspClanMember if role == "esp" else ClanMember
        behaviour = member(
            clan,
            awareness_m=profile.awareness_m(cls),
            hunter=role in ("hunter", "esp"),
            speed_mps=_speed(world),
            pois=pois,
            poi_bias=0.3,
        )
    return world.add_player(
        behaviour, archetype=f"clan_{role}", pos=pos, dino_class=cls, join_at=join_at, team=clan.name
    )


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


def teams(world: World) -> dict[str, str | None]:
    """
    The (hidden) team of every player, for evaluation.

    :param world: the world
    :return: player id -> team id, None for solo players
    """
    return {p.player_id: p.team for p in world.players}


def clans(world: World) -> dict[str, Clan]:
    """
    The shared state of every clan in the world (calls, responses, kills, regroups), for tests and statistics.

    :param world: the world
    :return: team id -> clan
    """
    found: dict[str, Clan] = {}
    for p in world.players:
        if isinstance(p.behaviour, ClanMember):
            found[p.behaviour.clan.name] = p.behaviour.clan
    return found
