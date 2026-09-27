"""Scoring features on small hand-built trajectories, where the right answer is known."""

import numpy as np
import pandas as pd
import pytest

from server.scoring.config import ScoringConfig
from server.scoring.features import (
    ambush_evidence,
    associates,
    beeline_evidence,
    spawn_episodes,
)
from server.scoring.trajectories import Trajectories, from_frame
from tests.helpers import DT, track

# A short null shift so 20-minute test tracks have a null at all.
CFG = ScoringConfig(null_shifts_s=(300.0,), near_lookback_s=60.0)


def world(*tracks: pd.DataFrame) -> Trajectories:
    return from_frame(pd.concat(tracks, ignore_index=True), "srv", CFG)


def beeline(tr: Trajectories, player: str = "hunter", start: int = 0) -> tuple[int, float]:
    ev = beeline_evidence(tr, CFG, associates(tr, CFG), start)[player]
    return ev.episodes, ev.null_episodes


def test_resampling_interpolates_and_leaves_long_gaps_absent() -> None:
    frame = pd.DataFrame(
        {"t": [0.0, 10.0, 100.0], "player_id": "p", "x": [0.0, 10.0, 100.0], "y": 0.0, "dino_class": "Troodon"}
    )
    tr = from_frame(frame, "srv", CFG)
    assert tr.pos[1, 0, 0] == pytest.approx(5.0)  # t=5, interpolated
    assert np.isnan(tr.pos[4, 0, 0])  # t=20, inside a 90 s gap
    assert tr.pos[-1, 0, 0] == 100.0
    assert tr.classes[1, 0] == "Troodon"


def test_resampling_rejects_missing_columns() -> None:
    with pytest.raises(ValueError, match="missing"):
        from_frame(pd.DataFrame({"t": [0.0]}), "srv", CFG)


def test_empty_frame() -> None:
    tr = from_frame(pd.DataFrame(columns=["t", "player_id", "x", "y"]), "srv", CFG)
    assert len(tr.t) == 0


def test_pursuing_a_moving_player_out_of_range_is_a_beeline() -> None:
    # The target walks north; the hunter, 1 km away, heads straight for wherever the target is.
    target = track("target", [(0, 1000, 0), (1200, 1000, 3600)])
    ts = np.arange(0, 1200 + 1e-9, DT)
    pos = np.zeros(2)
    hx, hy = [], []
    caught = False
    for t in ts:
        hx.append(pos[0])
        hy.append(pos[1])
        goal = np.array([1000.0, 3.0 * t])
        step = goal - pos
        dist = float(np.hypot(*step))
        caught = caught or dist < 30
        if not caught:  # after the kill the hunter stays put and the target walks on (so they are not a group)
            pos = pos + step / dist * min(6.0 * DT, dist)
    hunter = pd.DataFrame({"t": ts, "player_id": "hunter", "x": hx, "y": hy, "dino_class": "Troodon"})
    episodes, null = beeline(world(hunter, target))
    assert episodes == 1
    assert null == 0  # the time-shifted target is somewhere else


def test_walking_to_a_resting_player_is_explained_by_the_null() -> None:
    # The target never moves: shifting it in time changes nothing, so it is no evidence of knowing where it is.
    resting = track("resting", [(0, 1500, 0), (1200, 1500, 0)])
    walker = track("hunter", [(0, 0, 0), (250, 1500, 0), (300, 1500, 0), (500, 1500, 1200), (1200, 1500, 1200)])
    episodes, null = beeline(world(walker, resting))
    assert episodes == 1
    assert null == 1


def test_target_seen_recently_does_not_count() -> None:
    # They were together, the target left, the hunter follows it: legitimate.
    target = track("target", [(0, 0, 0), (60, 0, 0), (300, 1500, 0), (1200, 1500, 0)])
    hunter = track("hunter", [(0, 50, 0), (100, 50, 0), (400, 1500, 0), (1200, 1500, 0)])
    assert beeline(world(hunter, target))[0] == 0


def test_target_coming_towards_the_player_does_not_count() -> None:
    target = track("target", [(0, 1500, 0), (200, 300, 0)])
    hunter = track("hunter", [(0, 0, 0), (200, 300, 0)])
    assert beeline(world(hunter, target))[0] == 0


def test_someone_stepping_into_your_path_does_not_count() -> None:
    # The hunter walks east the whole time; the target appears on its route later (an ambusher, say).
    hunter = track("hunter", [(0, 0, 0), (600, 3000, 0)])
    target = track("target", [(200, 2500, 0), (600, 2500, 0)])
    assert beeline(world(hunter, target))[0] == 0


def test_group_mates_are_never_targets() -> None:
    together = [(0, 0, 0), (700, 0, 0)]
    mate = track("mate", [*together, (800, 1500, 0), (1200, 1500, 0)])
    hunter = track("hunter", [*together, (1000, 0, 0), (1200, 1400, 0)])
    tr = world(hunter, mate)
    assert associates(tr, CFG)[0, 1]
    assert beeline(tr)[0] == 0


def test_episodes_before_the_window_start_are_context_only() -> None:
    resting = track("resting", [(0, 1500, 0), (1200, 1500, 0)])
    walker = track("hunter", [(0, 0, 0), (250, 1500, 0), (300, 1500, 0), (500, 1500, 1200), (1200, 1500, 1200)])
    tr = world(walker, resting)
    assert beeline(tr, start=int(np.searchsorted(tr.t, 600)))[0] == 0


def test_waiting_where_a_distant_player_arrives_is_an_ambush_hit() -> None:
    # The waiter walks in, waits at the origin from t=180 to t=330, then leaves; the passer arrives at t=240. In
    # the null the passer arrives 300 s later, after the waiter has gone.
    waiter = track("waiter", [(0, -1000, -1000), (180, 0, 0), (330, 0, 0), (600, 2000, 2000), (1200, 2000, 2000)])
    passer = track("passer", [(0, 1500, 0), (240, 0, 0), (600, -2000, 0), (1200, -2000, 0)])
    ev = ambush_evidence(world(waiter, passer), CFG, np.zeros((2, 2), dtype=bool))["waiter"]
    assert (ev.waits, ev.hits) == (2, 1)  # the second wait is standing still at the end, where nobody comes
    assert ev.null_hits == 0


def test_spawn_episodes() -> None:
    # "late" joins at t=300 next to nobody and reaches "other" 100 s later; "gone" joins and never meets anyone.
    other = track("other", [(0, 0, 0), (1200, 0, 0)])
    late = track("late", [(300, 600, 0), (400, 30, 0), (1200, 30, 0)])
    gone = track("gone", [(600, 5000, 5000), (1200, 5000, 5000)])
    episodes = {e.player_id: e for e in spawn_episodes(world(other, late, gone), CFG, np.zeros((3, 3), dtype=bool))}
    assert "other" not in episodes  # present from the start: when it spawned is unknown
    assert not episodes["late"].censored
    assert episodes["late"].duration_s == pytest.approx(100, abs=DT)
    assert episodes["gone"].censored
    assert episodes["gone"].duration_s == pytest.approx(600, abs=DT)


def test_respawn_by_teleport_starts_a_new_episode() -> None:
    other = track("other", [(0, 0, 0), (1200, 0, 0)])
    frame = pd.concat(
        [
            other,
            track("p", [(300, 2000, 0), (600, 2000, 0)]),
            track("p", [(605, -3000, 0), (1200, -3000, 0)]),
        ]
    )
    episodes = spawn_episodes(from_frame(frame, "srv", CFG), CFG, np.zeros((2, 2), dtype=bool))
    assert [e.player_id for e in episodes] == ["p", "p"]
