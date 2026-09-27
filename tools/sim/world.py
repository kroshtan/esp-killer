"""
A small 2-D world of players moving around a square map.

Internally everything is in metres and seconds. :meth:`World.snapshot` converts to Unreal units (cm), which is
what the game server reports over RCON. The simulator is deterministic for a given seed, so the fake RCON server,
the scoring tests and the evaluation report all see reproducible trajectories.

Players have lives: they join at some time, may die (randomly, or killed by a hunter or cheater), are absent
for a while (death screen, class selection) and respawn at a random point. Absent players are not in snapshots.
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
    team: str | None = None  # ground truth: the clan the player belongs to, None if solo; never exposed over RCON
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))
    present: bool = True
    respawn_at: float | None = None  # while absent: when the player comes back
    deaths: int = 0


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
    def __init__(
        self,
        half_size_m: float = 4000.0,
        seed: int = 0,
        death_rate_per_s: float = 0.0,
        down_s: tuple[float, float] = (40.0, 90.0),
    ) -> None:
        self.half_size_m = half_size_m
        self.rng = np.random.default_rng(seed)
        self.players: list[SimPlayer] = []
        self.time_s = 0.0
        self.death_rate_per_s = death_rate_per_s
        self.down_s = down_s

    def add_player(
        self,
        behaviour: Behaviour,
        *,
        archetype: str = "honest",
        pos: np.ndarray | None = None,
        name: str | None = None,
        dino_class: str | None = None,
        join_at: float = 0.0,
        team: str | None = None,
    ) -> SimPlayer:
        """
        Add a player at ``pos`` (or a random position) with the given behaviour.

        :param behaviour: decides the player's velocity each step
        :param archetype: label used by tests and the evaluation report; never exposed over RCON
        :param pos: start position in metres
        :param name: display name; a generated one if omitted
        :param dino_class: class name; a random one if omitted
        :param join_at: simulation time at which the player joins (absent until then)
        :param team: the clan the player belongs to (ground truth, never exposed over RCON); None for solo players
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
            team=team,
            present=join_at <= self.time_s,
            respawn_at=join_at if join_at > self.time_s else None,
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

    def kill(self, player: SimPlayer) -> None:
        """
        Kill a player: absent for a while, then respawned at a random point.

        :param player: the victim
        """
        if not player.present:
            return
        player.present = False
        player.deaths += 1
        player.respawn_at = self.time_s + float(self.rng.uniform(*self.down_s))

    def step(self, dt: float) -> None:
        """
        Advance the world by ``dt`` seconds.

        Velocities are decided from the positions at the start of the step, then everyone moves at once.

        :param dt: time step in seconds
        """
        for p in self.players:
            if not p.present and p.respawn_at is not None and p.respawn_at <= self.time_s:
                p.present, p.respawn_at = True, None
                p.pos = self.random_point()
                reset = getattr(p.behaviour, "reset", None)
                if reset is not None:
                    reset()
        live = [p for p in self.players if p.present]
        velocities = [p.behaviour.velocity(p, self, dt) for p in live]
        for player, v in zip(live, velocities, strict=True):
            player.velocity = v
            player.pos = np.clip(player.pos + v * dt, -self.half_size_m, self.half_size_m)
        if self.death_rate_per_s > 0:
            for p in live:
                if self.rng.random() < self.death_rate_per_s * dt:
                    self.kill(p)
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
        Every present player except ``player``.

        :param player: the player to exclude
        :return: the other players
        """
        return [p for p in self.players if p is not player and p.present]

    def snapshot(self) -> list[SnapshotRow]:
        """
        The current state, in Unreal units, as the game server would report it.

        :return: one row per present player
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
            if p.present
        ]
