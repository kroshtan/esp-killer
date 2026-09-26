"""
Measure how well the scoring separates cheaters from honest players on simulated servers.

Runs several seeds of :func:`tools.sim.scenarios.mixed_world` for a few hours each, extracts evidence window by
window exactly as the scoring job does, and reports per archetype the score distribution and the key sub-score
statistics, plus the separation between each cheater type and all honest players (AUC) and the flag rates at the
configured threshold. It does this twice: for the first window alone, and for the evidence accumulated over all
windows, which is what the job scores.

Run it with ``python -m tools.sim.evaluate --seeds 8 --hours 6``.

Simulation, not reality: use it to catch regressions and
to see which honest behaviours come closest to the line, not as a promise about real servers.
"""

import argparse
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from server.scoring.combine import OrgEvidence, PlayerScore, extract_evidence, score_evidence
from server.scoring.config import ScoringConfig
from server.scoring.trajectories import from_frame
from tools.sim.scenarios import CHEATERS, HONEST, archetypes, mixed_world, record


@dataclass(frozen=True)
class Scored:
    seed: int
    archetype: str
    score: PlayerScore


def run(seeds: Sequence[int], config: ScoringConfig, hours: float = 6.0) -> tuple[list[Scored], list[Scored]]:
    """
    Simulate one server per seed and score it window by window.

    :param seeds: rng seeds
    :param config: scoring config (window and context lengths)
    :param hours: simulated time per server
    :return: scores from the first window only, and from the evidence summed over all windows
    """
    first: list[Scored] = []
    accumulated: list[Scored] = []
    window = config.window_minutes * 60
    context = config.context_minutes * 60
    for seed in seeds:
        world = mixed_world(seed)
        frame = record(world, hours * 3600)
        arch = archetypes(world)
        total = OrgEvidence()
        for i, start in enumerate(np.arange(0.0, hours * 3600 - window + 1, window)):
            chunk = frame[(frame["t"] >= start - context) & (frame["t"] < start + window)]
            ev = extract_evidence([from_frame(chunk, f"sim-{seed}", config)], config, count_from=start)
            total = total + ev
            if i == 0:
                first += [Scored(seed, arch[s.player_id], s) for s in score_evidence(ev, config)]
        accumulated += [Scored(seed, arch[s.player_id], s) for s in score_evidence(total, config)]
    return first, accumulated


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

    :param results: output of :func:`run`
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
    for arch in (*HONEST, *CHEATERS):
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
    honest = [r.score.score for r in results if r.archetype in HONEST]
    lines.append("")
    for arch in CHEATERS:
        cheat = [r.score.score for r in by_arch.get(arch, [])]
        if cheat:
            lines.append(f"AUC {arch} vs honest: {auc(cheat, honest):.3f}")
    fp = sum(s >= config.flag_threshold for s in honest)
    lines.append(f"honest players flagged: {fp}/{len(honest)} at threshold {config.flag_threshold}")
    return "\n".join(lines)


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, default=8, help="number of simulated servers")
    parser.add_argument("--first-seed", type=int, default=100)
    parser.add_argument("--hours", type=float, default=6.0, help="simulated time per server")
    args = parser.parse_args()
    config = ScoringConfig()
    first, accumulated = run(range(args.first_seed, args.first_seed + args.seeds), config, hours=args.hours)
    print(f"== first {config.window_minutes:.0f}-minute window only\n{report(first, config)}\n")  # noqa: T201
    print(f"== accumulated over {args.hours:g} hours\n{report(accumulated, config)}")  # noqa: T201


if __name__ == "__main__":
    main()
