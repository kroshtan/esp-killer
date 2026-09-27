from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from server.scoring.config import ScoringConfig
from server.scoring.trajectories import Trajectories, from_frame
from server.training.leakage import (
    ABSOLUTE_COLUMNS,
    Calibration,
    FeatureSpec,
    LeakageStats,
    Samples,
    aggregate,
    build_samples,
    candidate_rows,
    feature_names,
    leakage_terms,
    stable_fold,
)
from tools.sim.scenarios import mixed_world, record

CONFIG = ScoringConfig()
SPEC = FeatureSpec()


def straight_line_world(far_target: tuple[float, float]) -> Trajectories:
    """
    Three players: "a" walks east and "near" alongside within sight.

    "far" stands at ``far_target``, out of everyone's range and nobody's companion, so it is hidden from "a" all the
    time.
    """
    t = np.arange(0.0, 3600.0, 3.0)
    rows = []
    for ti in t:
        rows.append((ti, "a", 5.0 * ti / 3.0 % 2000, 0.0))
        rows.append((ti, "near", 5.0 * ti / 3.0 % 2000, 150.0 + 20 * np.sin(ti / 60)))
        rows.append((ti, "far", far_target[0] + 3 * np.cos(ti / 30), far_target[1]))
    frame = pd.DataFrame(rows, columns=["t", "player_id", "x", "y"])
    return from_frame(frame, "s", CONFIG)


def test_hidden_players_never_reach_model_a() -> None:
    """The core property: model A's columns (and candidate rows) do not change when only a hidden player moves."""
    first = straight_line_world((1000.0, 2000.0))
    second = straight_line_world((-1500.0, -1800.0))
    a = first.player_ids.index("a")
    s1 = build_samples(first, SPEC, CONFIG, players=[a])
    s2 = build_samples(second, SPEC, CONFIG, players=[a])
    assert len(s1) > 100
    n_a = s1.n_a
    np.testing.assert_array_equal(s1.x[:, :n_a], s2.x[:, :n_a])
    assert not np.allclose(np.nan_to_num(s1.x[:, n_a:]), np.nan_to_num(s2.x[:, n_a:]))  # B does see it
    rows_a1, rows_b1 = candidate_rows(s1, SPEC)
    rows_a2, rows_b2 = candidate_rows(s2, SPEC)
    np.testing.assert_array_equal(rows_a1, rows_a2)
    assert not np.allclose(np.nan_to_num(rows_b1), np.nan_to_num(rows_b2))


def test_a_columns_are_a_prefix_and_b_rows_hold_only_hidden_columns() -> None:
    names_a, names_b = feature_names(SPEC)
    assert all(n.startswith(("hidden", "n_hidden", "track_hidden")) for n in names_b)
    assert not any("hidden" in n for n in names_a)


@pytest.fixture(scope="module")
def sim_tr() -> Trajectories:
    world = mixed_world(3, roamers=4, waterhole=3, campers=1, hunters=2, groups=1)
    return from_frame(record(world, 3000.0), "sim", CONFIG)


def rotate(tr: Trajectories, angle: float) -> Trajectories:
    c, s = np.cos(angle), np.sin(angle)
    x, y = tr.pos[..., 0], tr.pos[..., 1]
    return replace(tr, pos=np.stack([c * x - s * y, s * x + c * y], axis=-1))


def test_features_are_rotation_invariant(sim_tr: Trajectories) -> None:
    before = build_samples(sim_tr, SPEC, CONFIG)
    after = build_samples(rotate(sim_tr, 1.1), SPEC, CONFIG)
    assert len(before) == len(after) > 500
    names = [*feature_names(SPEC)[0], *feature_names(SPEC)[1]]
    relative = [i for i, n in enumerate(names) if n not in ABSOLUTE_COLUMNS]
    # Angles are wrapped to [-pi, pi): compare them on the circle.
    diff = before.x[:, relative].astype(float) - after.x[:, relative].astype(float)
    is_angle = np.array([names[i].endswith(("bearing", "heading_to_me", "turn", "turn_lag1")) for i in relative])
    diff[:, is_angle] = (diff[:, is_angle] + np.pi) % (2 * np.pi) - np.pi
    np.testing.assert_allclose(np.nan_to_num(diff), 0.0, atol=1e-2)
    assert np.mean(before.y == after.y) > 0.99  # a sample right on a bin edge may flip
    assert not np.allclose(before.x[:, names.index("x")], after.x[:, names.index("x")])


def test_samples_count_only_from_count_from(sim_tr: Trajectories) -> None:
    start = float(sim_tr.t[0] + 1200)
    samples = build_samples(sim_tr, SPEC, CONFIG, count_from=start)
    assert len(samples) and samples.t.min() >= start
    assert samples.x_null.shape == (len(CONFIG.null_shifts_s), len(samples), samples.x.shape[1] - samples.n_a)


def test_null_columns_differ_from_real_ones(sim_tr: Trajectories) -> None:
    samples = build_samples(sim_tr, SPEC, CONFIG)
    real = np.nan_to_num(samples.x[:, samples.n_a :])
    for shifted in samples.x_null:
        assert not np.allclose(real, np.nan_to_num(shifted))
    assert samples.null(0).x.shape == samples.x.shape


def test_terms_are_zero_when_real_and_null_agree() -> None:
    rng = np.random.default_rng(0)
    log_pa = np.log(rng.dirichlet(np.ones(5), size=20))
    log_pb = np.log(rng.dirichlet(np.ones(5), size=20))
    y = rng.integers(0, 5, size=20)
    np.testing.assert_allclose(leakage_terms(log_pa, log_pb, [log_pb, log_pb], y), 0.0)
    better = leakage_terms(log_pa, np.log(np.eye(5)[y] * 0.9 + 0.02), [log_pa], y)
    assert np.all(better > 0)


def test_stats_add_up_and_z_reflects_the_mean() -> None:
    rng = np.random.default_rng(1)
    noise = rng.normal(0.0, 1.0, 800)
    signal = rng.normal(0.3, 1.0, 800)
    assert abs(LeakageStats.of(noise, 8).z) < 3
    assert LeakageStats.of(signal, 8).z > 5
    whole = LeakageStats.of(signal, 8)
    parts = LeakageStats.of(signal[:400], 8) + LeakageStats.of(signal[400:], 8)
    assert parts.n == whole.n and parts.blocks == whole.blocks
    assert parts.z == pytest.approx(whole.z)
    assert LeakageStats.from_dict(whole.to_dict()) == whole
    assert LeakageStats().z == 0.0


def test_correlated_terms_do_not_look_significant() -> None:
    """A streak (the same value for a whole block) must not count as many independent samples."""
    rng = np.random.default_rng(2)
    streaky = np.repeat(rng.normal(0.0, 1.0, 50), 16)
    assert abs(LeakageStats.of(streaky, 16).z) < 3.5


def test_aggregate_groups_by_player_in_time_order() -> None:
    players = np.array(["b", "a", "b", "a"], dtype=object)
    t = np.array([2.0, 1.0, 1.0, 2.0])
    gains = np.array([1.0, 2.0, 3.0, 4.0])
    stats = aggregate(players, t, gains, block_samples=1)
    assert stats["a"].sum_gain == 6.0 and stats["b"].n == 2


def test_calibration_ramp_and_minimum_samples() -> None:
    cal = Calibration(z_floor=2.0, z_full=6.0, min_samples=10)
    assert cal.score(1.0, 100) == 0.0
    assert cal.score(4.0, 100) == pytest.approx(0.5)
    assert cal.score(9.0, 100) == 1.0
    assert cal.score(9.0, 5) == 0.0


def test_folds_are_stable_and_spread() -> None:
    folds = [stable_fold(f"player-{i}", 3) for i in range(300)]
    assert folds == [stable_fold(f"player-{i}", 3) for i in range(300)]
    assert min(np.bincount(folds)) > 70


def test_empty_trajectories_give_no_samples() -> None:
    empty = from_frame(pd.DataFrame({"t": [], "player_id": [], "x": [], "y": []}), "s", CONFIG)
    samples = build_samples(empty, SPEC, CONFIG)
    assert len(samples) == 0
    assert isinstance(samples, Samples)
