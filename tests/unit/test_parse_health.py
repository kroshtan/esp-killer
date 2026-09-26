from agent.health import HealthTally
from shared.models import ParseHealth
from shared.playerdata import parse_player_data

GOOD = "Name: A, PlayerID: 76561198000000001, Location: X=1 Y=2 Z=3, Class: Troodon, Mood: Grumpy"
SALVAGEABLE = "Who=B, Id 76561198000000002, At X=4 Y=5 Z=6"
LOST = "Name: C, PlayerID: 76561198000000003"


def test_tally_counts_lines_by_outcome_without_values() -> None:
    tally = HealthTally()
    tally.record(parse_player_data("\n".join([GOOD, SALVAGEABLE, LOST, "PlayerDataEnd"])))
    tally.record(parse_player_data(GOOD))
    health = tally.take()
    assert health is not None
    assert (health.polls, health.lines, health.players, health.salvaged, health.unparsed, health.ended) == (
        2,
        4,
        3,
        1,
        1,
        1,
    )
    assert health.errors == {"salvaged: line does not start with 'Name:'": 1, "no Location field": 1}
    assert health.unknown_keys == ["Mood"]
    dumped = health.model_dump_json()
    assert "7656" not in dumped and "Grumpy" not in dumped
    assert tally.take() is None  # taken


def test_given_back_tallies_are_sent_next_time() -> None:
    tally = HealthTally()
    tally.record(parse_player_data(GOOD))
    first = tally.take()
    tally.record(parse_player_data(GOOD))
    tally.give_back(first)
    again = tally.take()
    assert again is not None
    assert again.polls == 2


def test_health_adds_up_and_caps_lists() -> None:
    a = ParseHealth(
        polls=1, errors={f"r{i}": 1 for i in range(20)}, unknown_keys=[f"K{chr(65 + i)}" for i in range(20)]
    )
    b = ParseHealth(polls=2, errors={"r0": 2, "new": 1}, unknown_keys=["Zz"])
    total = a + b
    assert total.polls == 3
    assert total.errors["r0"] == 3
    assert "new" not in total.errors  # capped at 20 reasons
    assert len(total.unknown_keys) == 20
    assert total.problems == 0
