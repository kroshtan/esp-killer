"""
Movement behaviours for simulated players.

Each behaviour returns a velocity (m/s) per step. Honest behaviours only use what a real player could know: their
own position, points of interest, players within ``awareness_m``, and (for group members) where their friends
are, as voice chat would tell them. Cheating behaviours also use the positions of strangers far beyond awareness
range, which is exactly what ESP gives them.

The honest archetypes are chosen to be the hard cases for the detector: waterhole crowds (everyone converges on
the same spots), campers (long waits), hunters (chasing players they can see) and groups (heading straight for
a friend who is far away).
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


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.hypot(*(a - b)))


def _towards(src: np.ndarray, dst: np.ndarray, speed: float, dt: float) -> np.ndarray:
    delta = dst - src
    dist = float(np.hypot(*delta))
    if dist < 1e-9:
        return np.zeros(2)
    # Never overshoot the destination within one step.
    velocity: np.ndarray = delta / dist * min(speed, dist / dt)
    return velocity


def _jitter(v: np.ndarray, world: "World", sigma: float) -> np.ndarray:
    """Rotate ``v`` by a small random angle: real paths are not ruler-straight."""
    angle = world.rng.normal(0.0, sigma)
    c, s = np.cos(angle), np.sin(angle)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


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

    def reset(self) -> None:
        """Forget the current waypoint (after a respawn)."""
        self._waypoint = None
        self._pause_left = 0.0

    def velocity(self, me: "SimPlayer", world: "World", dt: float) -> np.ndarray:  # noqa: D102
        if self._pause_left > 0:
            self._pause_left -= dt
            return np.zeros(2)
        if self._waypoint is None or _dist(self._waypoint, me.pos) < self.speed * dt:
            if self._waypoint is not None:
                self._pause_left = float(world.rng.uniform(*self.pause_range))
            self._waypoint = self._next_waypoint(world)
            return np.zeros(2)
        return _jitter(_towards(me.pos, self._waypoint, self.speed, dt), world, self.heading_noise)

    def _next_waypoint(self, world: "World") -> np.ndarray:
        if self.pois and world.rng.random() < self.poi_bias:
            poi = self.pois[int(world.rng.integers(len(self.pois)))]
            return np.asarray(poi + world.rng.normal(0.0, 30.0, size=2))
        return world.random_point()


class Hunter(Roamer):
    """
    Roam, but chase and kill any player who comes within awareness range: aggressive but legitimate.

    It may keep chasing a target that runs out of range, which is fine: it saw them first.
    """

    def __init__(self, awareness_m: float = 300.0, contact_m: float = 30.0, **kw: object) -> None:
        super().__init__(**kw)  # type: ignore[arg-type]
        self.awareness_m = awareness_m
        self.contact_m = contact_m
        self._target: SimPlayer | None = None

    def reset(self) -> None:  # noqa: D102
        super().reset()
        self._target = None

    def velocity(self, me: "SimPlayer", world: "World", dt: float) -> np.ndarray:  # noqa: D102
        if self._target is not None and (not self._target.present or _dist(self._target.pos, me.pos) > 800):
            self._target = None
        if self._target is None:
            visible = [p for p in world.others(me) if _dist(p.pos, me.pos) <= self.awareness_m]
            if visible:
                self._target = min(visible, key=lambda p: _dist(p.pos, me.pos))
        if self._target is None:
            return super().velocity(me, world, dt)
        if _dist(self._target.pos, me.pos) <= self.contact_m:
            if world.rng.random() < 0.5 * dt / 30.0:  # kills take a while
                world.kill(self._target)
                self._target = None
            return np.zeros(2)
        return _jitter(_towards(me.pos, self._target.pos, self.speed * 1.3, dt), world, 0.05)


class GroupMember(Roamer):
    """
    Travel with a group: follow the leader, sometimes wander off alone, then head straight back to them.

    Heading straight back to a friend who is far away is exactly what a beeline looks like. The detector must
    recognise these players as associates (they spend a lot of time together) and not count it.
    """

    def __init__(self, leader: "SimPlayer | None" = None, wander_rate_per_s: float = 1 / 900, **kw: object) -> None:
        super().__init__(**kw)  # type: ignore[arg-type]
        self.leader = leader
        self.wander_rate_per_s = wander_rate_per_s
        self._offset: np.ndarray | None = None
        self._wander_left = 0.0

    def reset(self) -> None:  # noqa: D102
        super().reset()
        self._wander_left = 0.0

    def velocity(self, me: "SimPlayer", world: "World", dt: float) -> np.ndarray:  # noqa: D102
        leader = self.leader
        if leader is None or leader is me or not leader.present:
            return super().velocity(me, world, dt)
        if self._offset is None:
            self._offset = world.rng.normal(0.0, 15.0, size=2)
        if self._wander_left > 0:
            self._wander_left -= dt
            return super().velocity(me, world, dt)
        if world.rng.random() < self.wander_rate_per_s * dt:
            self._wander_left = float(world.rng.uniform(180.0, 480.0))
            self._waypoint = me.pos + world.rng.uniform(-1500.0, 1500.0, size=2)
            return np.zeros(2)
        # Regroup (straight at the leader, like a friend giving directions over voice) or keep formation.
        return _jitter(_towards(me.pos, leader.pos + self._offset, self.speed * 1.2, dt), world, 0.05)


class BeelineCheater:
    """
    Head straight for the nearest player that is *beyond* awareness range, as ESP makes possible.

    After reaching a target (within ``contact_m``) it fights for ``engage_s``, usually kills, and picks the next
    target. It moves no faster than an honest player: the tell is direction, not speed. With ``active_fraction``
    below 1 it alternates between hunting and roaming like an honest player, to be harder to catch.
    """

    def __init__(
        self,
        *,
        speed_mps: float = 6.0,
        awareness_m: float = 300.0,
        contact_m: float = 30.0,
        engage_s: float = 45.0,
        heading_noise: float = 0.05,
        active_fraction: float = 1.0,
        phase_s: float = 600.0,
        give_up_s: float = 300.0,
    ) -> None:
        self.speed = speed_mps
        self.awareness_m = awareness_m
        self.contact_m = contact_m
        self.engage_s = engage_s
        self.heading_noise = heading_noise
        self.active_fraction = active_fraction
        self.phase_s = phase_s
        self.give_up_s = give_up_s
        self._target: SimPlayer | None = None
        self._chase_s = 0.0
        self._engage_left = 0.0
        self._phase_left = 0.0
        self._active = True
        self._roam = Roamer(speed_mps=speed_mps)

    def reset(self) -> None:
        """Forget the current target (after a respawn)."""
        self._target = None
        self._engage_left = 0.0
        self._roam.reset()

    def velocity(self, me: "SimPlayer", world: "World", dt: float) -> np.ndarray:  # noqa: D102
        if self.active_fraction < 1.0:
            self._phase_left -= dt
            if self._phase_left <= 0:
                self._active = bool(world.rng.random() < self.active_fraction)
                self._phase_left = self.phase_s
                self._target = None
            if not self._active:
                return self._roam.velocity(me, world, dt)
        if self._engage_left > 0:
            self._engage_left -= dt
            if self._engage_left <= 0 and self._target is not None:
                world.kill(self._target)
                self._target = None
            return np.zeros(2)
        if self._target is not None and (not self._target.present or self._chase_s > self.give_up_s):
            self._target = None  # gone, or too fast to catch: pick someone else
        if self._target is None:
            far = [p for p in world.others(me) if _dist(p.pos, me.pos) > self.awareness_m]
            if not far:
                return self._roam.velocity(me, world, dt)
            self._target = min(far, key=lambda p: _dist(p.pos, me.pos))
            self._chase_s = 0.0
        self._chase_s += dt
        if _dist(self._target.pos, me.pos) <= self.contact_m:
            self._engage_left = self.engage_s
            return np.zeros(2)
        return _jitter(_towards(me.pos, self._target.pos, self.speed, dt), world, self.heading_noise)


def _route_point(p: "SimPlayer", ahead_m: float) -> np.ndarray:
    """Where ``p`` will be after ``ahead_m`` of travel: towards its waypoint if it has one, else straight on."""
    waypoint = getattr(p.behaviour, "_waypoint", None)
    if waypoint is None:
        return np.asarray(p.pos + p.velocity / max(float(np.hypot(*p.velocity)), 1e-9) * ahead_m)
    to_go = _dist(waypoint, p.pos)
    return np.asarray(p.pos + (waypoint - p.pos) * min(1.0, ahead_m / max(to_go, 1e-9)))


class AmbushCheater:
    """
    Use ESP to predict where a distant player is going, get there first, and wait.

    It picks a moving target beyond awareness range and a spot on the target's route ``lead_s`` ahead, walks there
    if it can arrive with time to spare, and waits up to ``max_wait_s``. ESP shows position and heading; knowing
    the route (the simulated target's next waypoint) stands in for the game sense an experienced player adds to
    that. It never heads *at* the target, so it does not look like a beeline.
    """

    def __init__(
        self,
        *,
        speed_mps: float = 7.0,
        awareness_m: float = 300.0,
        contact_m: float = 30.0,
        lead_s: float = 200.0,
        margin_s: float = 100.0,
        max_wait_s: float = 240.0,
    ) -> None:
        self.speed = speed_mps
        self.awareness_m = awareness_m
        self.contact_m = contact_m
        self.lead_s = lead_s
        self.margin_s = margin_s
        self.max_wait_s = max_wait_s
        self._target: SimPlayer | None = None
        self._spot: np.ndarray | None = None
        self._wait_left = 0.0
        self._roam = Roamer(speed_mps=speed_mps, pause_s=(5.0, 30.0))

    def reset(self) -> None:
        """Forget the current plan (after a respawn)."""
        self._target, self._spot, self._wait_left = None, None, 0.0
        self._roam.reset()

    def velocity(self, me: "SimPlayer", world: "World", dt: float) -> np.ndarray:  # noqa: D102
        if self._target is not None and not self._target.present:
            self.reset()
        if self._target is None:
            self._plan(me, world)
            if self._target is None:
                return self._roam.velocity(me, world, dt)
        assert self._target is not None and self._spot is not None  # noqa: S101
        if _dist(self._target.pos, me.pos) <= self.contact_m:
            world.kill(self._target)
            self.reset()
            return np.zeros(2)
        if _dist(self._spot, me.pos) > self.speed * dt:
            return _towards(me.pos, self._spot, self.speed, dt)
        self._wait_left -= dt
        if self._wait_left <= 0:
            self.reset()
        return np.zeros(2)

    def _plan(self, me: "SimPlayer", world: "World") -> None:
        for p in sorted(world.others(me), key=lambda p: _dist(p.pos, me.pos)):
            speed = float(np.hypot(*p.velocity))
            if _dist(p.pos, me.pos) <= self.awareness_m or speed < 1.0:
                continue
            spot = _route_point(p, speed * self.lead_s)
            if np.any(np.abs(spot) > world.half_size_m):
                continue
            if _dist(spot, me.pos) / self.speed <= self.lead_s - self.margin_s:
                self._target, self._spot = p, spot
                self._wait_left = self.max_wait_s
                return
