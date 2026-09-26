import dataclasses
from datetime import datetime
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from shared.playerdata import line_shape, parse_player_data
from tools.rcon_format import format_player_data
from tools.sim.world import SnapshotRow

FIXTURES = Path(__file__).parent.parent / "fixtures" / "rcon"


def load(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def test_expected_format() -> None:
    result = parse_player_data(load("player_data_expected.txt"))

    assert result.errors == []
    assert result.header_timestamp == datetime(2025, 6, 1, 12, 34, 56)
    alice, bob, eos = result.players
    assert (alice.player_id, alice.player_name, alice.dino_class, alice.growth) == (
        "76561198000000001",
        "Alice",
        "Carnotaurus",
        0.75,
    )
    assert (alice.x, alice.y, alice.z) == (-12345.6, 78901.2, 345.0)
    assert bob.player_name == "Bob, the, Builder"
    assert eos.player_id == "0002a1b2c3d4e5f60718293a4b5c6d7e"
    assert eos.player_name == "Ünïcødé 恐竜"
    assert (eos.x, eos.y, eos.z) == (-1500.0, 225.0, -0.5)


def test_vitals_are_not_part_of_the_result() -> None:
    player = parse_player_data(load("player_data_expected.txt")).players[0]
    assert not {"health", "stamina", "hunger", "thirst"} & {f.name for f in dataclasses.fields(player)}


def test_empty_server() -> None:
    result = parse_player_data(load("player_data_empty.txt"))
    assert result.players == []
    assert result.errors == []


@pytest.mark.parametrize("text", ["", "\x00", "   \n\n", "[2025.06.01-12.34.56] PlayerData\x00"])
def test_blank_responses(text: str) -> None:
    result = parse_player_data(text)
    assert (result.players, result.errors) == ([], [])


def test_tolerated_variants() -> None:
    result = parse_player_data(load("player_data_variants.txt"))

    assert result.errors == []
    by_name = {p.player_name: p for p in result.players}
    assert by_name["SameLine"].dino_class == "Troodon"
    assert by_name["Reordered"].dino_class == "Deinosuchus"
    assert by_name["Reordered"].growth == pytest.approx(0.8)
    assert (by_name["Reordered"].x, by_name["Reordered"].y) == (10.5, -20.25)
    assert by_name["NoClass"].dino_class is None
    assert by_name["NoClass"].growth is None
    assert by_name["Extra"].dino_class == "Maiasaura"


def test_broken_lines_become_errors_not_exceptions() -> None:
    result = parse_player_data(load("player_data_broken.txt"))

    assert [p.player_name for p in result.players] == ["Good"]
    reasons = [e.reason for e in result.errors]
    assert reasons == [
        "no Location field",
        "Location has no Z coordinate",
        "PlayerID is not id-shaped",
        "duplicate player id",
        "line does not start with 'Name:'",
        "no Location field",
    ]


def test_error_shape_contains_keys_but_no_values() -> None:
    result = parse_player_data("Name: Secret Person, PlayerID: 76561198000000099, Location: X=1 Y=2")
    (error,) = result.errors
    assert error.shape == "Name PlayerID Location X Y"
    assert "Secret" not in error.shape
    assert "7656" not in error.shape


@pytest.mark.parametrize(("raw", "expected"), [("0.5", 0.5), ("50%", 0.5), ("1.5", None), ("-0.1", None), ("x", None)])
def test_growth_values(raw: str, expected: float | None) -> None:
    line = f"Name: G, PlayerID: 1, Location: X=0 Y=0 Z=0, Growth: {raw}"
    assert parse_player_data(line).players[0].growth == expected


def test_line_shape_truncates() -> None:
    assert len(line_shape("A: " * 500)) <= 200


names = st.text(
    alphabet=st.characters(blacklist_categories=("Cs", "Cc", "Zl", "Zp"), blacklist_characters="\x00\n\r"),
    min_size=0,
    max_size=40,
).map(str.strip)
coords = st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)
rows = st.builds(
    SnapshotRow,
    player_id=st.from_regex(r"^[0-9]{17}$|^[0-9a-f]{32}$", fullmatch=True),
    name=names,
    dino_class=st.sampled_from(["Carnotaurus", "Stegosaurus", "Pachycephalosaurus"]),
    growth=st.floats(min_value=0.0, max_value=1.0),
    x=coords,
    y=coords,
    z=coords,
)


@pytest.mark.parametrize("style", ["2025", "2026"])
@given(players=st.lists(rows, max_size=20, unique_by=lambda r: r.player_id))
def test_round_trip_with_the_fake_server_format(style: str, players: list[SnapshotRow]) -> None:
    text = format_player_data(players, datetime(2025, 1, 1, 0, 0, 0), style)
    result = parse_player_data(text)

    assert result.errors == []
    assert [p.player_id for p in result.players] == [r.player_id for r in players]
    for parsed, row in zip(result.players, players, strict=True):
        assert parsed.player_name == row.name
        assert parsed.dino_class == row.dino_class
        # The formatter writes 3 decimals and 2 for growth.
        assert parsed.x == pytest.approx(row.x, abs=1e-3)
        assert parsed.y == pytest.approx(row.y, abs=1e-3)
        assert parsed.growth == pytest.approx(row.growth, abs=0.01)


@given(st.text(max_size=300))
def test_never_raises_on_arbitrary_text(text: str) -> None:
    parse_player_data(text)


def test_2025_format_with_the_first_player_glued_to_the_header() -> None:
    result = parse_player_data(load("player_data_2025.txt"))
    assert result.errors == []
    assert [p.player_name for p in result.players] == ["First Player", "Second, Player"]
    assert [p.dino_class for p in result.players] == ["Carnotaurus", "Hypsilophodon"]
    assert result.header_timestamp == datetime(2025, 5, 14, 20, 11, 3)
    assert not result.ended


def test_2026_format_with_gender_mutations_and_end_marker() -> None:
    result = parse_player_data(load("player_data_2026.txt"))
    assert result.errors == []
    assert result.ended
    alpha, beta = result.players
    assert (alpha.player_id, alpha.dino_class, alpha.growth) == ("76561198000000041", "Tyrannosaurus", 1.0)
    assert (alpha.x, alpha.y, alpha.z) == (483129.59, -76220.472, 23256.202)
    assert beta.player_name == "Beta, Dryo"
    # Fields we know but do not use are not "unknown".
    assert result.unknown_keys == set()


def test_2026_empty_server() -> None:
    result = parse_player_data(load("player_data_2026_empty.txt"))
    assert (result.players, result.errors, result.ended) == ([], [], True)


def test_nothing_after_the_end_marker_is_parsed() -> None:
    text = load("player_data_2026.txt") + "Name: Late, PlayerID: 1, Location: X=0 Y=0 Z=0\n"
    assert len(parse_player_data(text).players) == 2


def test_unknown_fields_are_reported_by_name_only() -> None:
    line = "Name: A, PlayerID: 1, Location: X=0 Y=0 Z=0, Class: Troodon, Nesting: Secret Spot, Diet: Meat"
    result = parse_player_data(line)
    assert result.errors == []
    assert result.unknown_keys == {"Nesting", "Diet"}


def test_unrecognisable_lines_are_salvaged_from_id_and_location() -> None:
    # A made-up future format: different keys, but an id and an X/Y/Z triple.
    line = "Player=Rex, SteamId 76561198000000051, Pos: (X=10.5 Y=-20.0 Z=3), Species=Allosaurus"
    result = parse_player_data(line)
    (player,) = result.players
    assert (player.player_id, player.x, player.y, player.z) == ("76561198000000051", 10.5, -20.0, 3.0)
    (error,) = result.errors
    assert error.salvaged


def test_salvage_refuses_ambiguous_lines() -> None:
    line = "Kill: 76561198000000051 by 76561198000000052 at X=1 Y=2 Z=3"
    result = parse_player_data(line)
    assert result.players == []
    assert not result.errors[0].salvaged
