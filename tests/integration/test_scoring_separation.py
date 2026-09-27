"""
Fair players are never flagged, and cheaters still stand out, on simulated servers.

The priority is the operator's: a false flag punishes a fair player, a missed cheater is caught later with more
play. So the hard assertion is that no honest player (including clans hunting on each other's calls) crosses the
threshold; for cheaters the tests check that they rank above honest players, and that ambushers are flagged.

These seeds were not used to tune the thresholds (see NOTES.md for the full evaluation). Evidence is accumulated
window by window exactly as the scoring job does, including clan inference.
"""

import pytest

from server.scoring.config import ScoringConfig
from tools.sim.evaluate import Scored, auc, run
from tools.sim.scenarios import CLAN_HONEST, HONEST, SOLO_HONEST, clan_world

CFG = ScoringConfig()


@pytest.fixture(scope="module")
def mixed() -> list[Scored]:
    return run((900, 901, 902), CFG, hours=6.0).accumulated


@pytest.fixture(scope="module")
def clans() -> list[Scored]:
    return run((910, 911), CFG, hours=6.0, world_factory=clan_world).accumulated


def scores(results: list[Scored], archetypes: tuple[str, ...] | str) -> list[float]:
    names = (archetypes,) if isinstance(archetypes, str) else archetypes
    return [r.score.score for r in results if r.archetype in names]


def test_no_honest_player_is_flagged(mixed: list[Scored]) -> None:
    honest = [r for r in mixed if r.archetype in HONEST]
    assert len(honest) == 23 * 3
    worst = max(honest, key=lambda r: r.score.score)
    assert worst.score.score < CFG.flag_threshold, worst


def test_no_clan_member_is_flagged_for_hunting_on_calls(clans: list[Scored]) -> None:
    honest = [r for r in clans if r.archetype in (*SOLO_HONEST, *CLAN_HONEST)]
    assert any(r.archetype == "clan_hunter" for r in honest)
    worst = max(honest, key=lambda r: r.score.score)
    assert worst.score.score < CFG.flag_threshold, worst


def test_ambushers_are_flagged(mixed: list[Scored]) -> None:
    assert all(s >= CFG.flag_threshold for s in scores(mixed, "ambush_cheater"))


def test_ambushers_rank_above_every_honest_player(mixed: list[Scored]) -> None:
    assert auc(scores(mixed, "ambush_cheater"), scores(mixed, HONEST)) == 1.0


def test_beeline_cheaters_score_well_above_honest_players(mixed: list[Scored]) -> None:
    # Fair players first: after only 6 hours some beeline cheaters are not separated yet (they need more play), so
    # on three servers the promise is "clearly above on average"; the 12-server evaluation in NOTES.md gives the rates.
    honest = scores(mixed, HONEST)
    cheaters = scores(mixed, "beeline_cheater")
    assert sum(cheaters) / len(cheaters) > 3 * sum(honest) / len(honest)
    assert auc(cheaters, honest) >= 0.75


def test_part_time_cheater_scores_above_honest_players_on_average(mixed: list[Scored]) -> None:
    # Hunting with ESP only 40% of the time (in random phases), it needs more playing time than a few hours to be
    # flagged reliably; python -m tools.sim.evaluate reports its detection rate over many seeds.
    honest = scores(mixed, HONEST)
    subtle = scores(mixed, "subtle_cheater")
    assert sum(subtle) / len(subtle) > 3 * sum(honest) / len(honest)


def test_cheaters_on_clan_servers_stand_out(clans: list[Scored]) -> None:
    honest = scores(clans, (*SOLO_HONEST, *CLAN_HONEST))
    cheaters = scores(clans, ("beeline_cheater", "clan_esp"))
    assert sum(cheaters) / len(cheaters) > 3 * sum(honest) / len(honest)
