"""
Measure how well the scoring separates cheaters from honest players on simulated servers.

Runs several seeds of :func:`tools.sim.scenarios.mixed_world` (or, with ``--clans``,
:func:`tools.sim.scenarios.clan_world`) for a few hours each, extracts evidence window by window exactly as the
scoring job does, and reports per archetype the score distribution and the key sub-score statistics, plus the
separation between each cheater type and all honest players (AUC) and the flag rates at the configured threshold.
It does this twice: for the first window alone, and for the evidence accumulated over all windows, which is what
the job scores, including clan inference carried across windows. It also reports how well the inferred clans match
the true teams, as pair precision and recall.

Run it with ``python -m tools.sim.evaluate --seeds 8 --hours 6`` (add ``--clans`` for rival clans, and
``--no-team-inference`` to see what clan inference changes).

Simulation, not reality: use it to catch regressions and
to see which honest behaviours come closest to the line, not as a promise about real servers.
"""

import argparse
import math
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

from server.scoring.combine import OrgEvidence, PlayerScore, extract_evidence, score_evidence
from server.scoring.config import ScoringConfig
from server.scoring.game import load_profile
from server.scoring.teams import infer_teams
from server.scoring.trajectories import from_frame
from tools.sim.scenarios import (
    CHEATERS,
    CLAN_CHEATERS,
    CLAN_HONEST,
    HONEST,
    archetypes,
    clan_world,
    mixed_world,
    record,
    teams,
)
from tools.sim.world import World

ALL_HONEST = (*HONEST, *CLAN_HONEST)
ALL_CHEATERS = (*CHEATERS, *CLAN_CHEATERS)

# Infers clans from accumulated evidence: player id -> clan id; players in no clan are absent or None.


@dataclass(frozen=True)
class Scored:
    seed: int
    archetype: str
    score: PlayerScore


@dataclass(frozen=True)
class TeamMetrics:
    seed: int
    metrics: dict[str, float]  # output of :func:`pair_metrics`


@dataclass(frozen=True)
class RunResult:
    first: list[Scored]  # scores from the first window only
    accumulated: list[Scored]  # scores from the evidence summed over all windows
    teams: list[TeamMetrics] = field(default_factory=list)  # one per seed


def run(
    seeds: Sequence[int],
    config: ScoringConfig,
    hours: float = 6.0,
    *,
    world_factory: Callable[[int], World] = mixed_world,
) -> RunResult:
    """
    Simulate one server per seed and score it window by window.

    :param seeds: rng seeds
    :param config: scoring config (window and context lengths)
    :param hours: simulated time per server
    :param world_factory: builds the world for a seed, e.g. :func:`tools.sim.scenarios.clan_world`
    :return: the scores, and how well the inferred clans match the true teams
    """
    profile = load_profile()
    first: list[Scored] = []
    accumulated: list[Scored] = []
    team_metrics: list[TeamMetrics] = []
    window = config.window_minutes * 60
    context = config.context_minutes * 60
    for seed in seeds:
        world = world_factory(seed)
        frame = record(world, hours * 3600)
        arch = archetypes(world)
        total = OrgEvidence()
        for i, start in enumerate(np.arange(0.0, hours * 3600 - window + 1, window)):
            chunk = frame[(frame["t"] >= start - context) & (frame["t"] < start + window)]
            tr = from_frame(chunk, f"sim-{seed}", config, profile)
            # Clans inferred from the windows so far plus this one, as the scoring job does.
            ev = extract_evidence([tr], config, count_from=start, prior_pairs=total.pairs)
            total = total + ev
            if i == 0:
                first += [Scored(seed, arch[s.player_id], s) for s in score_evidence(ev, config)]
        accumulated += [Scored(seed, arch[s.player_id], s) for s in score_evidence(total, config)]
        team_metrics.append(TeamMetrics(seed, pair_metrics(teams(world), infer_teams(total.pairs, config))))
    return RunResult(first, accumulated, team_metrics)


def pair_metrics(true_teams: Mapping[str, str | None], inferred: Mapping[str, int | None]) -> dict[str, float]:
    """
    Precision and recall of same-team pairs: how many inferred clanmate pairs are true teammates, and vice versa.

    Only players in ``true_teams`` count. Solo players (team None), and players absent from ``inferred`` or mapped
    to None, are in no team and so in no pair.

    :param true_teams: player id -> true team, None for solo players
    :param inferred: player id -> inferred clan id; absent or None for players in no clan
    :return: ``precision`` and ``recall`` (NaN when there are no inferred or no true pairs), ``f1``, and the pair
        counts ``true_pairs``, ``inferred_pairs`` and ``correct_pairs``
    """
    players = sorted(true_teams)
    guess = {p: inferred.get(p) for p in players}
    true_pairs = {
        (a, b) for a, b in combinations(players, 2) if true_teams[a] is not None and true_teams[a] == true_teams[b]
    }
    inferred_pairs = {(a, b) for a, b in combinations(players, 2) if guess[a] is not None and guess[a] == guess[b]}
    correct = len(true_pairs & inferred_pairs)
    precision = correct / len(inferred_pairs) if inferred_pairs else math.nan
    recall = correct / len(true_pairs) if true_pairs else math.nan
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else math.nan
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_pairs": float(len(true_pairs)),
        "inferred_pairs": float(len(inferred_pairs)),
        "correct_pairs": float(correct),
    }


def auc(positives: Sequence[float], negatives: Sequence[float]) -> float:
    """
    Probability that a random positive scores above a random negative (ties count half).

    :param positives: scores of the positive class
    :param negatives: scores of the negative class
    :return: the AUC, or NaN if either side is empty
    """
    if not positives or not negatives:
        return float("nan")
    p = np.asarray(positives)[:, None]
    n = np.asarray(negatives)[None, :]
    return float(np.mean((p > n) + 0.5 * (p == n)))


def report(results: Sequence[Scored], config: ScoringConfig) -> str:
    """
    Render the evaluation as a text table.

    :param results: scores from :func:`run`
    :param config: the config used (for the threshold)
    :return: the report
    """
    by_arch: defaultdict[str, list[Scored]] = defaultdict(list)
    for r in results:
        by_arch[r.archetype].append(r)

    def z(r: Scored, name: str) -> float:
        sub = getattr(r.score, name)
        return float(sub.details.get("z", np.nan)) if sub is not None else np.nan

    header = (
        f"{'archetype':16s} {'n':>3s} {'mean':>5s} {'p90':>5s} {'max':>5s} {'flag%':>6s}"
        f" {'bee z max':>9s} {'ttc z max':>9s} {'amb z max':>9s}"
    )
    lines = [header]
    for arch in (*ALL_HONEST, *ALL_CHEATERS):
        rows = by_arch.get(arch, [])
        if not rows:
            continue
        scores = np.array([r.score.score for r in rows])
        flagged = np.mean(scores >= config.flag_threshold) * 100
        zs = [[z(r, n) for r in rows] for n in ("beeline", "ttc", "ambush")]
        zmax = [max((v for v in col if not np.isnan(v)), default=np.nan) for col in zs]
        lines.append(
            f"{arch:16s} {len(rows):3d} {scores.mean():5.2f} {np.percentile(scores, 90):5.2f} {scores.max():5.2f}"
            f" {flagged:5.0f}% {zmax[0]:9.2f} {zmax[1]:9.2f} {zmax[2]:9.2f}"
        )
    honest = [r.score.score for r in results if r.archetype in ALL_HONEST]
    lines.append("")
    for arch in ALL_CHEATERS:
        cheat = [r.score.score for r in by_arch.get(arch, [])]
        if cheat:
            lines.append(f"AUC {arch} vs honest: {auc(cheat, honest):.3f}")
    fp = sum(s >= config.flag_threshold for s in honest)
    lines.append(f"honest players flagged: {fp}/{len(honest)} at threshold {config.flag_threshold}")
    return "\n".join(lines)


def team_report(results: Sequence[TeamMetrics]) -> str:
    """
    Render the clan inference metrics as a text table, one row per seed plus the pooled totals.

    :param results: team metrics from :func:`run`
    :return: the report
    """
    lines = [f"{'seed':>6s} {'true':>5s} {'infer':>5s} {'right':>5s} {'prec':>5s} {'recall':>6s}"]
    for r in results:
        m = r.metrics
        lines.append(
            f"{r.seed:6d} {m['true_pairs']:5.0f} {m['inferred_pairs']:5.0f} {m['correct_pairs']:5.0f}"
            f" {m['precision']:5.2f} {m['recall']:6.2f}"
        )
    true_pairs = sum(r.metrics["true_pairs"] for r in results)
    inferred = sum(r.metrics["inferred_pairs"] for r in results)
    correct = sum(r.metrics["correct_pairs"] for r in results)
    precision = correct / inferred if inferred else math.nan
    recall = correct / true_pairs if true_pairs else math.nan
    lines.append(f"{'all':>6s} {true_pairs:5.0f} {inferred:5.0f} {correct:5.0f} {precision:5.2f} {recall:6.2f}")
    return "\n".join(lines)


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, default=8, help="number of simulated servers")
    parser.add_argument("--first-seed", type=int, default=100)
    parser.add_argument("--hours", type=float, default=6.0, help="simulated time per server")
    parser.add_argument("--clans", action="store_true", help="simulate servers with rival clans (clan_world)")
    parser.add_argument(
        "--no-team-inference", action="store_true", help="never link players into clans (for comparison)"
    )
    args = parser.parse_args()
    # An unreachable number of meetups switches clan inference off without touching the code paths.
    config = ScoringConfig(team_min_meets=10**9) if args.no_team_inference else ScoringConfig()
    result = run(
        range(args.first_seed, args.first_seed + args.seeds),
        config,
        hours=args.hours,
        world_factory=clan_world if args.clans else mixed_world,
    )
    print(f"== first {config.window_minutes:.0f}-minute window only\n{report(result.first, config)}\n")  # noqa: T201
    print(f"== accumulated over {args.hours:g} hours\n{report(result.accumulated, config)}")  # noqa: T201
    print(f"\n== clan inference: same-team pairs\n{team_report(result.teams)}")  # noqa: T201


if __name__ == "__main__":
    main()
