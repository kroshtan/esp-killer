"""
The scripted cheaters score clearly above normal players on simulated servers.

These seeds were not used to tune the thresholds (tuning used 100-107; ``python -m tools.sim.evaluate`` reports
larger runs). Evidence is accumulated window by window exactly as the scoring job does.
"""

import pytest

from server.scoring.config import ScoringConfig
from tools.sim.evaluate import Scored, auc, run
from tools.sim.scenarios import HONEST

CFG = ScoringConfig()
SEEDS = (900, 901, 902)


@pytest.fixture(scope="module")
def scored() -> list[Scored]:
    # Evidence accumulates with play time; a few hours is not always enough for one cheater (see NOTES.md).
    _, accumulated = run(SEEDS, CFG, hours=6.0)
    return accumulated


def scores(results: list[Scored], archetype: str) -> list[float]:
    return [r.score.score for r in results if r.archetype == archetype]


def test_no_honest_player_is_flagged(scored: list[Scored]) -> None:
    honest = [r for r in scored if r.archetype in HONEST]
    assert len(honest) == 23 * len(SEEDS)
    worst = max(honest, key=lambda r: r.score.score)
    assert worst.score.score < CFG.flag_threshold, worst


@pytest.mark.parametrize("cheater", ["beeline_cheater", "ambush_cheater"])
def test_full_time_cheaters_are_flagged(scored: list[Scored], cheater: str) -> None:
    assert all(s >= CFG.flag_threshold for s in scores(scored, cheater))


@pytest.mark.parametrize("cheater", ["beeline_cheater", "ambush_cheater"])
def test_full_time_cheaters_rank_above_every_honest_player(scored: list[Scored], cheater: str) -> None:
    honest = [r.score.score for r in scored if r.archetype in HONEST]
    assert auc(scores(scored, cheater), honest) == 1.0


def test_part_time_cheater_scores_above_honest_players_on_average(scored: list[Scored]) -> None:
    # Hunting with ESP only 40% of the time (in random phases), it needs more playing time than a few hours to be
    # flagged reliably; python -m tools.sim.evaluate reports its detection rate over many seeds.
    honest = [r.score.score for r in scored if r.archetype in HONEST]
    subtle = scores(scored, "subtle_cheater")
    assert sum(subtle) / len(subtle) > 3 * sum(honest) / len(honest)
