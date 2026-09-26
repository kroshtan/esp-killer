import math

import pytest

from server.scoring.combine import OrgEvidence, count_z, score_evidence, ttc_scores
from server.scoring.config import ScoringConfig
from server.scoring.features import AmbushEvidence, BeelineEvidence, SpawnEpisode

CFG = ScoringConfig()


def test_count_z() -> None:
    assert count_z(3, 3) == 0
    assert count_z(10, 0) == 10
    assert count_z(1, 3) < 0
    assert count_z(9, 3) == pytest.approx(3.0)


def evidence(player: str, *, episodes: int = 0, null: float = 0.0, moving_s: float = 3600.0) -> OrgEvidence:
    ev = OrgEvidence()
    ev.beeline[player] = BeelineEvidence(moving_s=moving_s, episodes=episodes, null_episodes=null)
    ev.servers[player] = {"s1"}
    return ev


def test_evidence_adds_up_across_windows_and_servers() -> None:
    a = evidence("p", episodes=2, null=1.0)
    b = evidence("p", episodes=3, null=0.5)
    b.servers["p"] = {"s2"}
    b.ambush["p"] = AmbushEvidence(waits=4, hits=1, null_hits=0.25)
    b.episodes.append(SpawnEpisode("p", "Troodon", 60.0, censored=False))
    total = a + b
    assert (total.beeline["p"].episodes, total.beeline["p"].null_episodes) == (5, 1.5)
    assert total.beeline["p"].moving_s == 7200
    assert total.ambush["p"].waits == 4
    assert total.servers["p"] == {"s1", "s2"}
    assert len(total.episodes) == 1
    # Inputs are untouched.
    assert a.beeline["p"].episodes == 2


def test_beeline_score_ramps_from_the_floor_and_is_gated_on_moving_time() -> None:
    at_floor = score_evidence(evidence("p", episodes=2, null=0.0), CFG)[0]  # z = 2
    assert at_floor.beeline is not None
    assert at_floor.beeline.score == 0
    strong = score_evidence(evidence("p", episodes=10, null=1.0), CFG)[0]  # z = 6.4
    assert strong.beeline is not None
    assert strong.beeline.score == 1
    assert strong.score == 1
    idle = score_evidence(evidence("p", episodes=10, moving_s=60.0), CFG)[0]
    assert idle.beeline is None
    assert idle.score == 0


def test_noisy_or_weights() -> None:
    ev = evidence("p", episodes=10, null=1.0)
    ev.ambush["p"] = AmbushEvidence(waits=10, hits=10, null_hits=0.0)
    s = score_evidence(ev, CFG)[0]
    assert s.score == pytest.approx(1 - (1 - CFG.weight_beeline) * (1 - CFG.weight_ambush))
    assert set(s.details()) == {"servers", "beeline", "ambush"}


def spawns(player: str, durations: list[float], censored: bool = False, cls: str = "Troodon") -> list[SpawnEpisode]:
    return [SpawnEpisode(player, cls, d, censored) for d in durations]


def test_time_to_contact_ranks_against_other_players() -> None:
    baseline = [e for k in range(10) for e in spawns(f"b{k}", [300.0 + 30 * k, 600.0 + 30 * k])]
    fast = spawns("fast", [20.0, 25.0, 30.0, 35.0, 40.0])
    slow = spawns("slow", [5000.0, 6000.0], censored=True)
    scores = ttc_scores(baseline + fast + slow, CFG)
    assert scores["fast"].details["mean_rank"] == 0
    assert scores["fast"].details["z"] == pytest.approx(0.5 * math.sqrt(12 * 5), abs=0.01)
    assert scores["fast"].score == 1
    assert scores["slow"].score == 0
    assert scores["slow"].details["mean_rank"] > 0.9


def test_time_to_contact_needs_enough_episodes_and_baseline() -> None:
    # "a" has one episode (fewer than ttc_min_episodes); "b" has only a's episode as baseline.
    assert ttc_scores(spawns("a", [10.0]) + spawns("b", [100.0] * 20), CFG) == {}
    # "a" has enough episodes, but only three others to compare with.
    assert ttc_scores(spawns("a", [10.0, 20.0]) + spawns("b", [100.0] * 3), CFG) == {}


def test_time_to_contact_falls_back_to_all_classes() -> None:
    baseline = [e for k in range(10) for e in spawns(f"b{k}", [300.0, 600.0], cls="Stegosaurus")]
    fast = spawns("fast", [20.0, 25.0], cls="Troodon")
    assert ttc_scores(baseline + fast, CFG)["fast"].details["mean_rank"] == 0
