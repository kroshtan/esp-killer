"""
A small 2-D world of players moving around a square map.

Internally everything is in metres and seconds. :meth:`World.snapshot` converts to Unreal units (cm), which is
what the game server reports over RCON. The simulator is deterministic for a given seed, so the fake RCON server,
the scoring tests and the evaluation report all see reproducible trajectories.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from tools.sim.behaviours import Behaviour

UNREAL_UNITS_PER_METRE = 100.0

DINO_CLASSES: Sequence[str] = (
    "Carnotaurus",
    "Ceratosaurus",
    "Deinosuchus",
    "Dilophosaurus",
    "Gallimimus",
    "Herrerasaurus",
    "Hypsilophodon",
    "Maiasaura",
    "Omniraptor",
    "Pachycephalosaurus",
    "Stegosaurus",
    "Tenontosaurus",
    "Troodon",
)


@dataclass
class SimPlayer:
    player_id: str
    name: str
    dino_class: str
    growth: float
    pos: np.ndarray  # shape (2,), metres
    behaviour: Behaviour
    z: float = 0.0
    archetype: str = "honest"
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))


@dataclass(frozen=True, slots=True)
class SnapshotRow:
    player_id: str
    name: str
    dino_class: str
    growth: float
    x: float  # Unreal units
    y: float
    z: float


class World:
    def __init__(self, half_size_m: float = 4000.0, seed: int = 0) -> None:
        self.half_size_m = half_size_m
        self.rng = np.random.default_rng(seed)
        self.players: list[SimPlayer] = []
        self.time_s = 0.0

    def add_player(
        self,
        behaviour: Behaviour,
        *,
        archetype: str = "honest",
        pos: np.ndarray | None = None,
        name: str | None = None,
        dino_class: str | None = None,
    ) -> SimPlayer:
        """
        Add a player at ``pos`` (or a random position) with the given behaviour.

        :param behaviour: decides the player's velocity each step
        :param archetype: label used by tests and the evaluation report; never exposed over RCON
        :param pos: start position in metres
        :param name: display name; a generated one if omitted
        :param dino_class: class name; a random one if omitted
        :return: the new player
        """
        index = len(self.players)
        player = SimPlayer(
            # Steam64-shaped ids, so the parser and validation see realistic values.
            player_id=str(76561198000000000 + index),
            name=name if name is not None else f"Player {index}",
            dino_class=dino_class if dino_class is not None else str(self.rng.choice(DINO_CLASSES)),
            growth=float(self.rng.uniform(0.5, 1.0)),
            pos=pos.astype(float).copy() if pos is not None else self.random_point(),
            behaviour=behaviour,
            archetype=archetype,
        )
        self.players.append(player)
        return player

    def random_point(self, margin_m: float = 100.0) -> np.ndarray:
        """
        A uniformly random point inside the map.

        :param margin_m: distance to keep from the map edge
        :return: the point, in metres
        """
        lim = self.half_size_m - margin_m
        return self.rng.uniform(-lim, lim, size=2)

    def step(self, dt: float) -> None:
        """
        Advance the world by ``dt`` seconds.

        Velocities are decided from the positions at the start of the step, then everyone moves at once.

        :param dt: time step in seconds
        """
        velocities = [p.behaviour.velocity(p, self, dt) for p in self.players]
        for player, v in zip(self.players, velocities, strict=True):
            player.velocity = v
            player.pos = np.clip(player.pos + v * dt, -self.half_size_m, self.half_size_m)
        self.time_s += dt

    def run(self, duration_s: float, dt: float) -> None:
        """
        Advance the world by ``duration_s`` in steps of ``dt``.

        :param duration_s: total time to simulate
        :param dt: time step
        """
        for _ in range(round(duration_s / dt)):
            self.step(dt)

    def others(self, player: SimPlayer) -> list[SimPlayer]:
        """
        Every player except ``player``.

        :param player: the player to exclude
        :return: the other players
        """
        return [p for p in self.players if p is not player]

    def snapshot(self) -> list[SnapshotRow]:
        """
        The current state, in Unreal units, as the game server would report it.

        :return: one row per player
        """
        return [
            SnapshotRow(
                player_id=p.player_id,
                name=p.name,
                dino_class=p.dino_class,
                growth=p.growth,
                x=float(p.pos[0] * UNREAL_UNITS_PER_METRE),
                y=float(p.pos[1] * UNREAL_UNITS_PER_METRE),
                z=float(p.z * UNREAL_UNITS_PER_METRE),
            )
            for p in self.players
        ]
