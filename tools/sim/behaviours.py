"""
Movement behaviours for simulated players.

Each behaviour returns a velocity (m/s) per step. Honest behaviours only use what a real player could know: their
own position, points of interest, players within ``awareness_m``, and (for group members) where their friends
are, as voice chat would tell them. Cheating behaviours also use the positions of strangers far beyond awareness
range, which is exactly what ESP gives them.

The honest archetypes are chosen to be the hard cases for the detector: waterhole crowds (everyone converges on
the same spots), campers (long waits), hunters (chasing players they can see) and groups (heading straight for
a friend who is far away).

Clans (:class:`Clan`, :class:`ClanMember`, :class:`EspClanMember`) are large, mixed-species teams spread over the
map that share sightings over voice chat: one member spots a lone player and hunters far away head straight for
them. That is legitimate as long as some member really saw the target first.
"""

from dataclasses import dataclass
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


@dataclass(frozen=True, slots=True)
class ClanCall:
    """A sighting shared over voice chat: ``caller`` reports where ``target`` is."""

    target: "SimPlayer"
    time_s: float
    caller: "SimPlayer"
    distance_m: float  # caller to target when called
    esp: bool = False  # ground truth: the caller could not see the target


@dataclass(frozen=True, slots=True)
class ClanResponse:
    """A hunter setting off for a called target."""

    time_s: float
    hunter: "SimPlayer"
    target: "SimPlayer"
    distance_m: float  # hunter to target when it set off
    caller: "SimPlayer"


class Clan:
    """
    What a clan shares over voice chat: its base, the targets members have called, and when to regroup.

    Calls are keyed by target; calling a target again refreshes the call. For tests and statistics,
    :attr:`call_log` keeps each *new* call (a target that was not called recently), :attr:`responses` each hunter
    response to a call, :attr:`kills` each kill by a member and :attr:`regroups` the regroup windows. The schedule
    uses the clan's own rng, so it does not disturb the world's random sequence.

    :param name: team id; members have ``SimPlayer.team == name``
    :param base: the nest / regroup spot, metres
    :param rng: random generator for the schedule
    :param call_ttl_s: how long a call stays actionable after it was last refreshed
    :param regroup_every_s: range of the time from the end of one regroup to the start of the next
    :param gather_s: time allowed for travelling to the base
    :param stay_s: range of the time spent at the base after ``gather_s``
    :param base_radius_m: members wait within about this distance of the base
    """

    def __init__(
        self,
        name: str,
        base: np.ndarray,
        *,
        rng: np.random.Generator,
        call_ttl_s: float = 120.0,
        regroup_every_s: tuple[float, float] = (1200.0, 2400.0),
        gather_s: float = 900.0,
        stay_s: tuple[float, float] = (180.0, 480.0),
        base_radius_m: float = 40.0,
    ) -> None:
        self.name = name
        self.base = np.asarray(base, dtype=float)
        self.rng = rng
        self.call_ttl_s = call_ttl_s
        self.regroup_every_s = regroup_every_s
        self.gather_s = gather_s
        self.stay_s = stay_s
        self.base_radius_m = base_radius_m
        self.calls: dict[str, ClanCall] = {}
        self.call_log: list[ClanCall] = []
        self.responses: list[ClanResponse] = []
        self.kills: list[tuple[float, SimPlayer, SimPlayer]] = []  # (time, killer, victim)
        self.regroups: list[tuple[float, float]] = []  # (start, end) of every regroup scheduled so far
        self._schedule_next(0.0)

    def _schedule_next(self, after_s: float) -> None:
        start = after_s + float(self.rng.uniform(*self.regroup_every_s))
        self.regroups.append((start, start + self.gather_s + float(self.rng.uniform(*self.stay_s))))

    def regrouping(self, now_s: float) -> bool:
        """
        Whether the clan is gathering at (or staying at) its base.

        :param now_s: simulation time; must not go backwards between calls
        :return: True during a regroup
        """
        while now_s >= self.regroups[-1][1]:
            self._schedule_next(self.regroups[-1][1])
        start, end = self.regroups[-1]
        return start <= now_s < end

    def call(
        self, target: "SimPlayer", caller: "SimPlayer", now_s: float, distance_m: float, *, esp: bool = False
    ) -> None:
        """
        Report a sighting of ``target`` to the clan, or refresh an earlier one.

        :param target: the player seen
        :param caller: the member reporting it
        :param now_s: simulation time
        :param distance_m: distance from caller to target
        :param esp: ground truth: the caller only knows through ESP
        """
        new = ClanCall(target, now_s, caller, distance_m, esp)
        if not self.is_called(target, now_s):
            self.call_log.append(new)
        self.calls[target.player_id] = new

    def is_called(self, target: "SimPlayer", now_s: float) -> bool:
        """
        Whether ``target`` was called recently enough to act on.

        :param target: the player
        :param now_s: simulation time
        :return: True if the last call on ``target`` is younger than ``call_ttl_s``
        """
        call = self.calls.get(target.player_id)
        return call is not None and now_s - call.time_s < self.call_ttl_s

    def recent_calls(self, now_s: float) -> list[ClanCall]:
        """
        The calls still worth acting on.

        :param now_s: simulation time
        :return: calls younger than ``call_ttl_s`` whose target is present
        """
        return [c for c in self.calls.values() if now_s - c.time_s < self.call_ttl_s and c.target.present]


class ClanMember(Roamer):
    """
    A clan member: roams alone or with a teammate, regroups at the base, and calls every stranger it sees.

    Priorities, highest first: a hunter that just killed eats for a while; a hunter chases a stranger it sees
    itself (or keeps chasing its current target); everyone goes to the base while the clan regroups; a hunter
    responds (with probability ``response_prob``; always to its own calls) to a recent call up to
    ``max_response_m`` away, heading for the target's current position as voice chat would describe it; a member
    visiting a teammate travels straight to them and follows them for a while; otherwise it roams. Non-hunters
    only call. Kills happen on contact, as for :class:`Hunter`; clanmates are never targets.

    A Pteranodon spotter is a non-hunter with a 900 m range and a high speed that roams the whole map. How
    Pteranodons really play in clans is not known; that is an assumption.

    :param clan: the shared clan state
    :param awareness_m: this member's own range (from its class)
    :param hunter: whether it chases seen and called targets
    :param contact_m: distance at which a fight happens
    :param max_response_m: calls on targets further than this are ignored
    :param give_up_s: a chase is abandoned after this long
    :param response_prob: chance that a hunter responds to a call it hears; if not, it ignores calls on that
        target for ``decline_s``
    :param decline_s: how long a declined target is ignored
    :param feed_s: range of how long a hunter stays at a kill (eating) before doing anything else
    :param visit_rate_per_s: rate of setting off to join a teammate
    :param follow_s: range of how long it stays with the teammate once there
    """

    def __init__(
        self,
        clan: Clan,
        *,
        awareness_m: float = 300.0,
        hunter: bool = False,
        contact_m: float = 30.0,
        max_response_m: float = 2500.0,
        give_up_s: float = 600.0,
        response_prob: float = 0.5,
        decline_s: float = 300.0,
        feed_s: tuple[float, float] = (180.0, 480.0),
        visit_rate_per_s: float = 1 / 1200,
        follow_s: tuple[float, float] = (300.0, 900.0),
        **kw: object,
    ) -> None:
        super().__init__(**kw)  # type: ignore[arg-type]
        self.clan = clan
        self.awareness_m = awareness_m
        self.hunter = hunter
        self.contact_m = contact_m
        self.max_response_m = max_response_m
        self.give_up_s = give_up_s
        self.response_prob = response_prob
        self.decline_s = decline_s
        self.feed_s = feed_s
        self.visit_rate_per_s = visit_rate_per_s
        self.follow_s = follow_s
        self._target: SimPlayer | None = None
        self._chase_s = 0.0
        self._declined: dict[str, float] = {}  # target id -> when this hunter decided not to respond
        self._feed_until = 0.0
        self._base_spot: np.ndarray | None = None
        self._mate: SimPlayer | None = None
        self._mate_offset = np.zeros(2)
        self._follow_until: float | None = None

    def reset(self) -> None:  # noqa: D102
        super().reset()
        self._target = None
        self._feed_until = 0.0
        self._base_spot = None
        self._mate = None
        self._follow_until = None

    def velocity(self, me: "SimPlayer", world: "World", dt: float) -> np.ndarray:  # noqa: D102
        now = world.time_s
        strangers = [p for p in world.others(me) if p.team != self.clan.name]
        dists = [_dist(p.pos, me.pos) for p in strangers]
        seen = [(p, d) for p, d in zip(strangers, dists, strict=True) if d <= self.awareness_m]
        for p, d in seen:
            self.clan.call(p, me, now, d)
        self._extra_calls(me, world, strangers, dists)
        regrouping = self.clan.regrouping(now)
        if self.hunter:
            v = self._hunt(me, world, dt, seen, regrouping=regrouping)
            if v is not None:
                return v
        if regrouping:
            return self._regroup(me, world, dt)
        self._base_spot = None
        v = self._visit(me, world, dt)
        if v is not None:
            return v
        return super().velocity(me, world, dt)

    def _extra_calls(self, me: "SimPlayer", world: "World", strangers: list["SimPlayer"], dists: list[float]) -> None:
        """Hook for calls on strangers the member cannot see (ESP)."""

    def _candidate_calls(self, now_s: float) -> list[ClanCall]:
        return self.clan.recent_calls(now_s)

    def _hunt(
        self, me: "SimPlayer", world: "World", dt: float, seen: list[tuple["SimPlayer", float]], *, regrouping: bool
    ) -> np.ndarray | None:
        now = world.time_s
        if now < self._feed_until:
            return np.zeros(2)
        if self._target is not None:
            self._chase_s += dt
            self._drop_lost_target(me, now)
        if self._target is None and seen:
            self._target, self._chase_s = min(seen, key=lambda pd: pd[1])[0], 0.0
        if self._target is None and not regrouping:
            self._respond_to_call(me, world)
        t = self._target
        if t is None:
            return None
        self._waypoint, self._pause_left = None, 0.0  # after the chase, roam somewhere new
        if _dist(t.pos, me.pos) <= self.contact_m:
            if world.rng.random() < 0.5 * dt / 30.0:  # kills take a while
                world.kill(t)
                self.clan.kills.append((now, me, t))
                self._target = None
                self._feed_until = now + float(world.rng.uniform(*self.feed_s))
            return np.zeros(2)
        return _jitter(_towards(me.pos, t.pos, self.speed * 1.3, dt), world, 0.05)

    def _drop_lost_target(self, me: "SimPlayer", now_s: float) -> None:
        """Stop chasing a target that is gone, no longer seen or called, or not worth it any more."""
        t = self._target
        assert t is not None  # noqa: S101
        d = _dist(t.pos, me.pos)
        known = d <= self.awareness_m or any(c.target is t for c in self._candidate_calls(now_s))
        if not t.present or not known or self._chase_s > self.give_up_s or d > self.max_response_m * 1.2:
            self._target = None

    def _respond_to_call(self, me: "SimPlayer", world: "World") -> None:
        """Maybe set off for the nearest called target in reach (always for this member's own calls)."""
        now = world.time_s
        accepted: list[tuple[ClanCall, float]] = []
        for c in self._candidate_calls(now):
            d = _dist(c.target.pos, me.pos)
            if d > self.max_response_m or c.target.team == self.clan.name:
                continue
            if now - self._declined.get(c.target.player_id, -np.inf) < self.decline_s:
                continue
            if c.caller is not me and world.rng.random() >= self.response_prob:
                self._declined[c.target.player_id] = now
                continue
            accepted.append((c, d))
        if accepted:
            call, d = min(accepted, key=lambda cd: cd[1])
            self._target, self._chase_s = call.target, 0.0
            self.clan.responses.append(ClanResponse(now, me, call.target, d, call.caller))

    def _regroup(self, me: "SimPlayer", world: "World", dt: float) -> np.ndarray:
        self._mate, self._follow_until, self._waypoint, self._pause_left = None, None, None, 0.0
        if self._base_spot is None:
            self._base_spot = self.clan.base + world.rng.normal(0.0, self.clan.base_radius_m / 2, size=2)
        if _dist(self._base_spot, me.pos) <= self.speed * dt:
            return np.zeros(2)
        return _jitter(_towards(me.pos, self._base_spot, self.speed * 1.2, dt), world, 0.1)

    def _visit(self, me: "SimPlayer", world: "World", dt: float) -> np.ndarray | None:
        now = world.time_s
        mate = self._mate
        if mate is not None and (not mate.present or (self._follow_until is not None and now > self._follow_until)):
            self._mate, self._follow_until, self._waypoint = None, None, None
            mate = None
        if mate is None:
            if world.rng.random() >= self.visit_rate_per_s * dt:
                return None
            mates = [p for p in world.others(me) if p.team == self.clan.name]
            if not mates:
                return None
            mate = self._mate = mates[int(world.rng.integers(len(mates)))]
            self._mate_offset = world.rng.normal(0.0, 15.0, size=2)
        dest = mate.pos + self._mate_offset
        if self._follow_until is None and _dist(dest, me.pos) < 50.0:  # noqa: PLR2004
            self._follow_until = now + float(world.rng.uniform(*self.follow_s))
        # Straight at the teammate, like a friend giving directions over voice.
        return _jitter(_towards(me.pos, dest, self.speed * 1.2, dt), world, 0.05)


class EspClanMember(ClanMember):
    """
    A clan member with ESP (a cheater): picks strangers it cannot see, calls them on the clan and hunts them.

    ``cooldown_s`` after its last target it picks the nearest stranger beyond its own range but within
    ``esp_max_m`` and keeps calling it (ESP keeps showing where it is) for up to ``hold_s``. With ``relay`` the call
    goes to the clan, whose hunters respond as to any call (so honest clanmates end up acting on ESP knowledge);
    without it only this member acts on it. As a hunter (the default) it responds to its own calls like to any
    other, so it still regroups with the clan.

    :param clan: the shared clan state
    :param esp_max_m: ignore strangers further than this
    :param relay: share ESP targets with the clan
    :param hold_s: how long it keeps calling one target
    :param cooldown_s: pause between ESP targets
    :param hunter: whether it chases targets itself
    """

    def __init__(
        self,
        clan: Clan,
        *,
        esp_max_m: float = 2500.0,
        relay: bool = True,
        hold_s: float = 600.0,
        cooldown_s: float = 120.0,
        hunter: bool = True,
        **kw: object,
    ) -> None:
        super().__init__(clan, hunter=hunter, **kw)  # type: ignore[arg-type]
        self.esp_max_m = esp_max_m
        self.relay = relay
        self.hold_s = hold_s
        self.cooldown_s = cooldown_s
        self._esp_target: SimPlayer | None = None
        self._esp_since = 0.0
        self._esp_next = 0.0
        self._private: ClanCall | None = None

    def reset(self) -> None:  # noqa: D102
        super().reset()
        self._esp_target = None
        self._private = None

    def _extra_calls(self, me: "SimPlayer", world: "World", strangers: list["SimPlayer"], dists: list[float]) -> None:
        now = world.time_s
        t = self._esp_target
        if t is not None and (not t.present or now - self._esp_since > self.hold_s):
            self._esp_target, self._private, self._esp_next = None, None, now + self.cooldown_s
        if self._esp_target is None and now >= self._esp_next:
            far = [(p, d) for p, d in zip(strangers, dists, strict=True) if self.awareness_m < d <= self.esp_max_m]
            if far:
                self._esp_target, self._esp_since = min(far, key=lambda pd: pd[1])[0], now
        t = self._esp_target
        if t is None:
            return
        d = _dist(t.pos, me.pos)
        if d <= self.awareness_m:
            return  # in sight now: the ordinary call covers it
        if self.relay:
            self.clan.call(t, me, now, d, esp=True)
        else:
            self._private = ClanCall(t, now, me, d, esp=True)

    def _candidate_calls(self, now_s: float) -> list[ClanCall]:
        calls = super()._candidate_calls(now_s)
        p = self._private
        if p is not None and now_s - p.time_s < self.clan.call_ttl_s and p.target.present:
            calls.append(p)
        return calls
