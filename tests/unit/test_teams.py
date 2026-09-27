from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from server.db.database import Database
from server.db.scoring import ScoringRepository
from server.scoring.combine import OrgEvidence
from server.scoring.config import ScoringConfig
from server.scoring.features import associates, beeline_evidence, tip_evidence
from server.scoring.teams import PairEvidence, infer_teams, pair_evidence, pair_key, team_matrix
from server.scoring.trajectories import from_frame
from tests.helpers import killed, track

CFG = ScoringConfig(null_shifts_s=(600.0,))


def meets(*tracks: pd.DataFrame, config: ScoringConfig = CFG) -> PairEvidence:
    tr = from_frame(pd.concat(tracks, ignore_index=True), "srv", config)
    shift = tr.steps(config.null_shifts_s[0])
    return pair_evidence(tr, config, [shift]).get(pair_key("a", "b"), PairEvidence())


def test_separate_meetups_are_counted_and_a_short_step_aside_is_not() -> None:
    # a stays home; b visits three times, going 1 km away in between, and once only steps 200 m aside.
    home = track("a", [(0, 0, 0), (2400, 0, 0)])
    visits = track(
        "b",
        [
            (0, 50, 0), (300, 50, 0), (400, 1000, 0), (700, 1000, 0), (800, 50, 0), (1000, 50, 0),
            (1030, 250, 0), (1060, 50, 0), (1300, 50, 0), (1400, 1000, 0), (1900, 1000, 0), (2000, 50, 0),
            (2400, 50, 0),
        ],
    )  # fmt: skip
    assert meets(home, visits).meets == 3


def test_a_meeting_that_ends_in_a_death_is_a_fight() -> None:
    victim = pd.concat([track("a", [(0, 0, 0), (400, 0, 0)]), track("a", [(900, 3000, 3000), (1200, 3000, 3000)])])
    hunter = track("b", [(0, 2000, 0), (200, 20, 0), (400, 20, 0), (600, 1500, 0), (1200, 1500, 0)])
    assert meets(victim, hunter).meets == 0


def test_strangers_resting_at_the_same_spot_are_explained_by_the_null() -> None:
    # Both sit at the same waterhole all the time: the time-shifted null meets just as often.
    a = track("a", [(0, 0, 0), (2400, 0, 0)])
    b = track("b", [(0, 30, 0), (2400, 30, 0)])
    ev = meets(a, b)
    assert ev.meets == 1
    assert ev.null_meets == 1
    assert ev.z() == 0


def test_clans_are_connected_links_and_weak_pairs_are_not_linked() -> None:
    strong = PairEvidence(meets=12, null_meets=0.5)
    pairs = {
        pair_key("a", "b"): strong,
        pair_key("b", "c"): strong,  # a-b-c: one clan through b
        pair_key("d", "e"): PairEvidence(meets=2, null_meets=0.0),  # too few meetups
        pair_key("f", "g"): PairEvidence(meets=6, null_meets=4.0),  # chance at a busy spot
    }
    teams = infer_teams(pairs, CFG)
    assert teams["a"] == teams["b"] == teams["c"]
    assert not {"d", "e", "f", "g"} & set(teams)


def scouted_hunt(scout_sees_first: bool) -> tuple[int, int]:
    """
    Count the hunter's beelines without and with the scout as a clanmate.

    A ground hunter walks straight to a prey 1.2 km away; a clanmate scout passes within 100 m of the prey either
    before the hunter sets off or only after.
    """
    prey = killed(track("prey", [(0, 1200, 0), (900, 1200, 0)]), at=310)  # killed on arrival
    hunter = track(
        "hunter", [(0, 0, 0), (100, 0, 0), (300, 1150, 0), (360, 1150, 0), (500, 1150, 900), (900, 1150, 900)]
    )
    if scout_sees_first:
        scout = track("scout", [(0, 0, -2000), (20, 1200, -100), (60, 1200, -100), (900, 3000, -2000)], "Troodon")
    else:
        # Joins the server next to the prey only after the hunter has lined up, so it was not heading there first.
        scout = track("scout", [(250, 1200, -100), (290, 1200, -100), (900, 3000, -2000)], "Troodon")
    tr = from_frame(pd.concat([prey, hunter, scout], ignore_index=True), "srv", CFG)
    assoc = associates(tr, CFG)
    alone = beeline_evidence(tr, CFG, assoc)["hunter"].episodes
    team = team_matrix(tr, {"hunter": 0, "scout": 0})
    together = beeline_evidence(tr, CFG, assoc | team, team=team)["hunter"].episodes
    return alone, together


def test_a_clanmates_earlier_sighting_explains_the_approach() -> None:
    assert scouted_hunt(scout_sees_first=True) == (1, 0)


def test_a_sighting_after_the_approach_began_explains_nothing() -> None:
    assert scouted_hunt(scout_sees_first=False) == (1, 1)


def test_team_matrix() -> None:
    tr = from_frame(pd.concat([track(p, [(0, 0, 0), (60, 0, 0)]) for p in "abc"]), "srv", CFG)
    assert team_matrix(tr, {"a": 0, "c": 0}).tolist() == [
        [False, False, True],
        [False, False, False],
        [True, False, False],
    ]
    assert not team_matrix(tr, None).any()


def test_pair_evidence_is_stored_per_window_and_summed(tmp_path: Path) -> None:
    repo = ScoringRepository(Database(tmp_path / "x.db"))
    start = datetime(2026, 1, 1, tzinfo=UTC)
    for hours in (2, 4):
        ev = OrgEvidence()
        ev.pairs[pair_key("b", "a")] = PairEvidence(meets=3, null_meets=0.5)
        repo.save_window("org", start + timedelta(hours=hours), ev)
    assert repo.load_pair_evidence("org", since=start) == {("a", "b"): PairEvidence(6, 1.0)}
    assert repo.load_pair_evidence("org", since=start + timedelta(hours=3)) == {("a", "b"): PairEvidence(3, 0.5)}
    assert np.isclose(PairEvidence(6, 1.0).z(), 5 / np.sqrt(2))


def test_joining_a_clanmates_hunt_is_a_relay() -> None:
    # The leader heads for a prey 1.2 km away (its own evidence); a clanmate lines up on the same prey a little later.
    prey = killed(track("prey", [(0, 1200, 0), (900, 1200, 0)]), at=320)  # killed by the leader
    leader = track("leader", [(0, 0, -300), (300, 1150, -50), (360, 1150, -50), (500, 1150, -900), (900, 1150, -900)])
    follower = track("follower", [(0, 0, 300), (60, 0, 300), (310, 1150, 50), (420, 1150, 50), (900, 1150, 900)])
    tr = from_frame(pd.concat([prey, leader, follower], ignore_index=True), "srv", CFG)
    team = team_matrix(tr, {"leader": 0, "follower": 0})
    evidence = beeline_evidence(tr, CFG, associates(tr, CFG) | team, team=team)
    assert evidence["leader"].episodes == 1
    assert evidence["follower"].episodes == 0


def test_tips_count_approaches_to_what_someone_else_saw() -> None:
    # Three times, the scout sees a prey and the hunter then walks straight to it from far away.
    tracks = []
    for n, t0 in enumerate((0, 1000, 2000)):
        x = 2000 * n
        tracks.append(track(f"prey{n}", [(t0, x + 1200, 0), (t0 + 900, x + 1200, 0)]))
    scout_points: list[tuple[float, float, float]] = [
        (0, 1200, -250)
    ]  # spots each prey from 250 m (not travelling with it)
    hunter_points: list[tuple[float, float, float]] = [(0, 0, 0)]
    for n, t0 in enumerate((0, 1000, 2000)):
        x = 2000 * n
        scout_points += [(t0 + 10, x + 1200, -250), (t0 + 60, x + 1200, -250)]
        hunter_points += [
            (t0 + 100, x, 0),
            (t0 + 300, x + 1150, 0),
            (t0 + 400, x + 1150, 600),
            (t0 + 999, x + 2000, 0),
        ]
    tracks += [track("scout", scout_points), track("hunter", hunter_points)]
    tr = from_frame(pd.concat(tracks, ignore_index=True), "srv", CFG)
    tips = tip_evidence(tr, CFG, associates(tr, CFG), [tr.steps(600)])
    ev = tips[pair_key("hunter", "scout")]
    assert ev.tips >= 3
    assert ev.tips > ev.null_tips
