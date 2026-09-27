"""The simulator's clans: ground truth, regrouping, shared calls, ESP relaying, and the pair metrics."""

import math

import numpy as np
import pytest

from server.scoring.game import load_profile
from tools.sim.behaviours import Clan, ClanMember, EspClanMember
from tools.sim.evaluate import pair_metrics
from tools.sim.scenarios import CLAN_CHEATERS, CLAN_HONEST, archetypes, clan_world, clans, record, teams
from tools.sim.world import SimPlayer, World

NEVER = (1e9, 1e9)  # a regroup schedule that never starts


class Still:
    """Stand still (a resting target)."""

    def velocity(self, me: SimPlayer, world: World, dt: float) -> np.ndarray:
        return np.zeros(2)


def make_clan(**kw: object) -> Clan:
    return Clan("clan-a", np.zeros(2), rng=np.random.default_rng(0), **kw)  # type: ignore[arg-type]


def at(x: float, y: float = 0.0) -> np.ndarray:
    return np.array([x, y])


def spotter(clan: Clan) -> ClanMember:
    """A clan member who stays put and calls what it sees."""
    return ClanMember(clan, speed_mps=0.0, visit_rate_per_s=0.0)


def test_clan_world_assigns_ground_truth_teams() -> None:
    world = clan_world(3)
    truth = teams(world)
    arch = archetypes(world)
    assert set(truth) == set(arch)
    for pid, team in truth.items():
        assert (team is not None) == arch[pid].startswith("clan_"), (pid, arch[pid], team)
    assert set(clans(world)) == {"clan-0", "clan-1"}
    for name in ("clan-0", "clan-1"):
        size = sum(t == name for t in truth.values())
        assert 6 <= size <= 10
    labels = set(arch.values())
    assert {"clan_member", "clan_hunter", "clan_spotter", "clan_esp", "beeline_cheater"} <= labels
    assert set(CLAN_HONEST) | set(CLAN_CHEATERS) >= {a for a in labels if a.startswith("clan_")}
    spotter = next(p for p in world.players if p.archetype == "clan_spotter")
    assert spotter.dino_class == "Pteranodon"
    assert spotter.behaviour.awareness_m == pytest.approx(900.0)  # type: ignore[attr-defined]
    assert len({p.dino_class for p in world.players if p.team == "clan-1"}) > 1  # mixed species


def test_members_regroup_at_the_base() -> None:
    world = World(half_size_m=3000.0, seed=1)
    clan = make_clan(regroup_every_s=(600.0, 900.0), stay_s=(300.0, 300.0))
    member = world.add_player(ClanMember(clan, speed_mps=6.0, visit_rate_per_s=0.0), pos=at(2000, 1500), team="clan-a")
    distances = []
    for _ in range(round(3600 / 10)):
        world.run(10.0, 1.0)
        distances.append(float(np.hypot(*(member.pos - clan.base))))
    assert min(distances) < 60.0
    assert max(distances) > 500.0
    # Staying at the base during the last part of each regroup.
    start, end = clan.regroups[0]
    world2 = World(half_size_m=3000.0, seed=1)
    clan2 = make_clan(regroup_every_s=(600.0, 900.0), stay_s=(300.0, 300.0))
    member2 = world2.add_player(
        ClanMember(clan2, speed_mps=6.0, visit_rate_per_s=0.0), pos=at(2000, 1500), team="clan-a"
    )
    world2.run(end - 30.0, 1.0)
    assert np.hypot(*(member2.pos - clan2.base)) < 60.0


def test_hunter_heads_for_a_called_target_far_outside_its_range() -> None:
    world = World(half_size_m=3000.0, seed=2)
    clan = make_clan(regroup_every_s=NEVER)
    hunter = world.add_player(
        ClanMember(clan, hunter=True, response_prob=1.0, speed_mps=6.0), pos=at(0), team="clan-a", name="hunter"
    )
    target = world.add_player(Still(), pos=at(1500), name="target")
    world.add_player(spotter(clan), pos=at(1450, 50), team="clan-a", name="caller")  # sees and calls the target
    world.run(10.0, 1.0)
    assert clan.is_called(target, world.time_s)
    assert [(r.hunter, r.target) for r in clan.responses] == [(hunter, target)]
    assert clan.responses[0].distance_m > 1400.0
    # Heading straight for it.
    v = hunter.velocity
    assert float(np.dot(v, at(1.0))) / float(np.hypot(*v)) > 0.95
    before = float(np.hypot(*(target.pos - hunter.pos)))
    world.run(60.0, 1.0)
    assert float(np.hypot(*(target.pos - hunter.pos))) < before - 300.0


def test_hunter_ignores_calls_beyond_its_response_distance() -> None:
    world = World(half_size_m=4000.0, seed=2)
    clan = make_clan(regroup_every_s=NEVER)
    hunter = world.add_player(
        ClanMember(clan, hunter=True, response_prob=1.0, max_response_m=1000.0), pos=at(-2000), team="clan-a"
    )
    world.add_player(Still(), pos=at(1500))
    world.add_player(spotter(clan), pos=at(1450, 50), team="clan-a")
    world.run(10.0, 1.0)
    assert clan.call_log and not clan.responses
    assert np.hypot(*hunter.velocity) > 0  # just roaming


def test_pteranodon_spotter_calls_targets_within_900_m() -> None:
    profile = load_profile()
    world = World(half_size_m=3000.0, seed=3)
    clan = make_clan(regroup_every_s=NEVER)
    world.add_player(
        ClanMember(clan, awareness_m=profile.awareness_m("Pteranodon"), speed_mps=13.0),
        pos=at(0),
        dino_class="Pteranodon",
        team="clan-a",
    )
    near = world.add_player(Still(), pos=at(700))
    far = world.add_player(Still(), pos=at(-1500))
    world.step(1.0)
    assert clan.is_called(near, world.time_s)
    assert not clan.is_called(far, world.time_s)
    assert 600.0 < clan.call_log[0].distance_m <= 900.0


def test_esp_clan_member_calls_targets_outside_its_range() -> None:
    world = World(half_size_m=3000.0, seed=4)
    clan = make_clan(regroup_every_s=NEVER)
    cheat = world.add_player(EspClanMember(clan, awareness_m=300.0), pos=at(0), team="clan-a", archetype="clan_esp")
    mate = world.add_player(Still(), pos=at(-1000), team="clan-a")
    target = world.add_player(Still(), pos=at(1200))
    world.run(5.0, 1.0)
    assert clan.is_called(target, world.time_s)
    call = clan.calls[target.player_id]
    assert call.esp and call.caller is cheat and call.distance_m > 300.0
    assert not clan.is_called(mate, world.time_s)  # never calls clanmates
    assert float(np.dot(cheat.velocity, at(1.0))) > 0  # and hunts it itself


def test_pair_metrics() -> None:
    truth: dict[str, str | None] = {"a": "x", "b": "x", "c": "x", "d": "y", "e": "y", "f": None}
    # Inferred: {a, b} right, {d, e, f} has one right pair (d, e) and two wrong ones; c missing.
    inferred: dict[str, int | None] = {"a": 1, "b": 1, "d": 2, "e": 2, "f": 2, "g": 1}
    m = pair_metrics(truth, inferred)
    assert m["true_pairs"] == 4  # ab ac bc de
    assert m["inferred_pairs"] == 4  # ab de df ef (g is not a known player)
    assert m["correct_pairs"] == 2
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(0.5)
    perfect = pair_metrics(truth, {"a": 7, "b": 7, "c": 7, "d": 8, "e": 8, "f": None})
    assert perfect["precision"] == perfect["recall"] == 1.0
    nothing = pair_metrics(truth, {})
    assert math.isnan(nothing["precision"]) and nothing["recall"] == 0.0


def test_clan_world_records_every_player() -> None:
    world = clan_world(5)
    frame = record(world, 1800.0)
    assert set(frame["player_id"]) == {p.player_id for p in world.players}
    last = frame[frame["t"] == frame["t"].max()]
    assert len(last) > 0.8 * len(world.players)
    assert frame.groupby("t").size().max() <= len(world.players)
    assert sum(len(c.call_log) for c in clans(world).values()) > 0
