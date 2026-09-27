"""
The promotion gate: whether a freshly trained model may replace the current one, decided without labels.

A model that scores the wrong people is worse than no model, and nobody reviews a weekly cron job, so a candidate is
promoted only if all of these hold:

1. **Recall on synthetic ESP users**: real players from held-out folds with hunting phases injected into their tracks
   rank above untouched real players (AUC of z at least ``synthetic_auc``).
2. **The simulator benchmark**: on fresh simulated servers (``tools/sim``, seeds never used for training) with known
   archetypes, no honest player reaches the flag level, and beeline cheaters rank above honest players (AUC of z at
   least ``sim_auc``). The honest archetypes include the hard cases: hunters, and clans hunting on each other's calls.
3. **The real-data flag rate** (out of sample) is within the budget.
4. **No regression**: if a model is promoted already, the candidate is not worse than it on (1) and (2), beyond a
   small tolerance for run-to-run noise. The current model is re-run on the same benchmark servers, so that part
   is a like-for-like comparison.

Every number goes into the candidate's metadata, promoted or not.
"""

import logging
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from server.scoring.config import ScoringConfig
from server.training.leakage import LeakageModel, LeakageStats
from tools.sim.scenarios import CHEATERS, CLAN_CHEATERS
from trainer.synthetic import sim_servers
from trainer.train import auc

logger = logging.getLogger(__name__)

SIM_CHEATERS = (*CHEATERS, *CLAN_CHEATERS)
# The benchmark's recall criterion: full-time ESP users who hunt by heading for unseen players. Ambushers and
# part-time or in-clan ESP users are reported but not required: the rule-based scoring covers ambushes, and the others
# need more playing time than a benchmark server has.
GATED_CHEATERS = ("beeline_cheater",)


@dataclass(frozen=True)
class GateThresholds:
    # Just below what the method reaches on the simulated development data (synthetic 0.74, simulator 0.92): enough
    # to stop a model that learned nothing (about 0.5) or regressed badly, not a claim that these values are good.
    synthetic_auc: float = 0.7
    sim_auc: float = 0.85
    flag_budget: float = 0.005
    flag_score: float = 0.6  # the flag level: a sub-score at or above this counts as flagged
    tolerance: float = 0.03  # allowed shortfall against the current model


@dataclass(frozen=True)
class Benchmark:
    kinds: tuple[str, ...] = ("mixed", "clan")
    seeds: tuple[int, ...] = (9001, 9002)
    hours: float = 4.0


@dataclass
class GateResult:
    passed: bool
    failures: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)


def run_benchmark(model: LeakageModel, bench: Benchmark, config: ScoringConfig, flag_score: float) -> dict[str, Any]:
    """
    Score simulated servers window by window, as the backend would, and summarise per archetype.

    :param model: the model to evaluate
    :param bench: which servers to simulate
    :param config: scoring config
    :param flag_score: the flag level
    :return: per-archetype statistics, honest maximum, flagged honest players and AUCs (of z, cheaters vs honest)
    """
    z_by_arch: defaultdict[str, list[float]] = defaultdict(list)
    score_by_arch: defaultdict[str, list[float]] = defaultdict(list)
    for server in sim_servers(bench.kinds, bench.seeds, bench.hours, config):
        total: dict[str, LeakageStats] = {}
        for tr, count_from in server.windows:
            for pid, stats in model.player_stats([tr], config, count_from=count_from).items():
                total[pid] = total.get(pid, LeakageStats()) + stats
        for pid, stats in total.items():
            scored = model.rescore(stats)
            if scored.n >= model.calibration.min_samples:
                z_by_arch[server.archetypes[pid]].append(scored.z)
                score_by_arch[server.archetypes[pid]].append(scored.score_0_1)
    honest = [a for a in z_by_arch if a not in SIM_CHEATERS]
    honest_z = [z for a in honest for z in z_by_arch[a]]
    honest_scores = [s for a in honest for s in score_by_arch[a]]
    gated = [z for a in GATED_CHEATERS for z in z_by_arch.get(a, [])]
    return {
        "archetypes": {
            a: {
                "n": len(zs),
                "mean_z": round(float(np.mean(zs)), 3),
                "max_z": round(float(np.max(zs)), 3),
                "max_score": round(float(np.max(score_by_arch[a])), 3),
                "flagged": int(sum(s >= flag_score for s in score_by_arch[a])),
            }
            for a, zs in sorted(z_by_arch.items())
        },
        "honest_players": len(honest_z),
        "honest_max_z": round(float(max(honest_z, default=0.0)), 3),
        "honest_max_score": round(float(max(honest_scores, default=0.0)), 3),
        "honest_flagged": int(sum(s >= flag_score for s in honest_scores)),
        "auc": {a: round(auc(z_by_arch[a], honest_z), 4) for a in SIM_CHEATERS if a in z_by_arch},
        "gated_auc": round(auc(gated, honest_z), 4),
    }


def evaluate(
    metrics: dict[str, Any],
    benchmark: dict[str, Any],
    thresholds: GateThresholds,
    *,
    current_metrics: dict[str, Any] | None = None,
    current_benchmark: dict[str, Any] | None = None,
) -> GateResult:
    """
    Apply the gate.

    :param metrics: the candidate's training metrics (``synthetic_auc``, ``real_flag_rate``)
    :param benchmark: the candidate's :func:`run_benchmark` result
    :param thresholds: the thresholds
    :param current_metrics: the promoted model's recorded training metrics, if there is one
    :param current_benchmark: the promoted model's result on the same benchmark servers
    :return: whether it passed, and why not
    """
    failures = []
    synthetic_auc = float(metrics.get("synthetic_auc", float("nan")))
    if not synthetic_auc >= thresholds.synthetic_auc:  # NaN fails too
        failures.append(f"synthetic ESP AUC {synthetic_auc:.3f} < {thresholds.synthetic_auc}")
    if benchmark["honest_flagged"] > 0:
        failures.append(
            f"{benchmark['honest_flagged']} honest simulated player(s) reach the flag level"
            f" (highest sub-score {benchmark['honest_max_score']:.2f})"
        )
    if not benchmark["gated_auc"] >= thresholds.sim_auc:
        failures.append(f"simulator AUC {benchmark['gated_auc']:.3f} < {thresholds.sim_auc} for {GATED_CHEATERS}")
    if metrics.get("real_flag_rate", 1.0) > thresholds.flag_budget:
        failures.append(f"real-data flag rate {metrics['real_flag_rate']:.4f} > budget {thresholds.flag_budget}")
    if current_metrics is not None:
        previous = float(current_metrics.get("synthetic_auc", float("nan")))
        if synthetic_auc < previous - thresholds.tolerance:
            failures.append(f"synthetic ESP AUC {synthetic_auc:.3f} is worse than the current model's {previous:.3f}")
    if current_benchmark is not None:
        if benchmark["gated_auc"] < current_benchmark["gated_auc"] - thresholds.tolerance:
            failures.append(
                f"simulator AUC {benchmark['gated_auc']:.3f} is worse than the current model's"
                f" {current_benchmark['gated_auc']:.3f}"
            )
        if benchmark["honest_max_z"] > max(current_benchmark["honest_max_z"], 0.0) + 1.0:
            failures.append(
                f"highest honest z on the simulator {benchmark['honest_max_z']:.2f} is well above the current"
                f" model's {current_benchmark['honest_max_z']:.2f}"
            )
    return GateResult(not failures, failures)


def summary(result: GateResult, benchmark: dict[str, Any], metrics: dict[str, Any]) -> Sequence[str]:
    """
    Log lines describing a gate decision.

    :param result: the decision
    :param benchmark: the candidate's benchmark result
    :param metrics: the candidate's training metrics
    :return: the lines
    """
    lines = [
        f"gate: {'PASS' if result.passed else 'FAIL'}",
        f"synthetic AUC {metrics.get('synthetic_auc')}, real flag rate {metrics.get('real_flag_rate')}",
        (
            f"simulator: gated AUC {benchmark['gated_auc']}, AUC per cheater {benchmark['auc']},"
            f" honest max z {benchmark['honest_max_z']}, honest flagged {benchmark['honest_flagged']}"
        ),
    ]
    lines += [f"  failed: {f}" for f in result.failures]
    return lines
