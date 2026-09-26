"""
Movement behaviours for simulated players.

Each behaviour returns a velocity (m/s) per step. Honest behaviours only use what a real player could know:
their own position, points of interest, and other players within ``awareness_m``. Cheating behaviours also use
the positions of players far beyond that range, which is exactly what ESP gives them.
"""

from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from tools.sim.world import SimPlayer, World


class Behaviour(Protocol):
    def velocity(self, me: "SimPlayer", world: "World", dt: float) -> np.ndarray:
        """
        Decide the player's velocity for the next step.

        :param me: the player being moved
        :param world: the world, for positions of other players and the rng
        :param dt: step length in seconds
        :return: velocity in m/s, shape (2,)
        """
        ...


def _towards(src: np.ndarray, dst: np.ndarray, speed: float, dt: float) -> np.ndarray:
    delta = dst - src
    dist = float(np.hypot(*delta))
    if dist < 1e-9:
        return np.zeros(2)
    # Never overshoot the destination within one step.
    velocity: np.ndarray = delta / dist * min(speed, dist / dt)
    return velocity


class Roamer:
    """
    Wander between waypoints, pausing at each one.

    Waypoints are random points, or with probability ``poi_bias`` one of ``pois`` (waterholes, trails), so honest
    players also converge on the same places the way real players do.
    """

    def __init__(
        self,
        speed_mps: float = 6.0,
        pause_s: tuple[float, float] = (10.0, 120.0),
        pois: list[np.ndarray] | None = None,
        poi_bias: float = 0.0,
        heading_noise: float = 0.15,
    ) -> None:
        self.speed = speed_mps
        self.pause_range = pause_s
        self.pois = pois or []
        self.poi_bias = poi_bias
        self.heading_noise = heading_noise
        self._waypoint: np.ndarray | None = None
        self._pause_left = 0.0

    def velocity(self, me: "SimPlayer", world: "World", dt: float) -> np.ndarray:  # noqa: D102
        rng = world.rng
        if self._pause_left > 0:
            self._pause_left -= dt
            return np.zeros(2)
        if self._waypoint is None or np.hypot(*(self._waypoint - me.pos)) < self.speed * dt:
            if self._waypoint is not None:
                self._pause_left = float(rng.uniform(*self.pause_range))
            if self.pois and rng.random() < self.poi_bias:
                poi = self.pois[int(rng.integers(len(self.pois)))]
                self._waypoint = poi + rng.normal(0.0, 30.0, size=2)
            else:
                self._waypoint = world.random_point()
            return np.zeros(2)
        v = _towards(me.pos, self._waypoint, self.speed, dt)
        # Real paths are not ruler-straight: rotate the heading by a little noise.
        angle = rng.normal(0.0, self.heading_noise)
        c, s = np.cos(angle), np.sin(angle)
        return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


class BeelineCheater:
    """
    Head straight for the nearest player that is *beyond* awareness range, as ESP makes possible.

    After reaching a target (within ``contact_m``) it lingers for ``engage_s`` (a fight or a kill) and then picks
    the next target. It moves no faster than an honest player: the tell is direction, not speed.
    """

    def __init__(
        self,
        speed_mps: float = 6.0,
        awareness_m: float = 300.0,
        contact_m: float = 30.0,
        engage_s: float = 60.0,
        heading_noise: float = 0.05,
    ) -> None:
        self.speed = speed_mps
        self.awareness_m = awareness_m
        self.contact_m = contact_m
        self.engage_s = engage_s
        self.heading_noise = heading_noise
        self._target: SimPlayer | None = None
        self._engage_left = 0.0
        self._fallback = Roamer(speed_mps=speed_mps)

    def velocity(self, me: "SimPlayer", world: "World", dt: float) -> np.ndarray:  # noqa: D102
        if self._engage_left > 0:
            self._engage_left -= dt
            if self._engage_left <= 0:
                self._target = None
            return np.zeros(2)
        others = world.others(me)
        if self._target is None:
            far = [p for p in others if np.hypot(*(p.pos - me.pos)) > self.awareness_m]
            if not far:
                return self._fallback.velocity(me, world, dt)
            self._target = min(far, key=lambda p: float(np.hypot(*(p.pos - me.pos))))
        if np.hypot(*(self._target.pos - me.pos)) <= self.contact_m:
            self._engage_left = self.engage_s
            return np.zeros(2)
        v = _towards(me.pos, self._target.pos, self.speed, dt)
        angle = world.rng.normal(0.0, self.heading_noise)
        c, s = np.cos(angle), np.sin(angle)
        return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])
