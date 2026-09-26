"""
Render RCON responses in the format the parser expects (see shared/playerdata.py).

Used by the fake RCON server and by the parser's round-trip tests. If a real server turns out to format things
differently, update this module and the fixtures in tests/fixtures/rcon/ together.
"""

from collections.abc import Iterable
from datetime import datetime

from tools.sim.world import SnapshotRow

FORMATS = ("2026", "2025")
_SLOTS = "[1=None,2=None,3=None,4=None]"


def format_player_data(rows: Iterable[SnapshotRow], now: datetime, style: str = "2026") -> str:
    """
    Render a ``PLAYER_DATA`` response, including the fields the agent is expected to ignore.

    :param rows: the players to include
    :param now: server-local time for the header
    :param style: ``2026`` (header on its own line, Gender, mutation slots, bare class, ``PlayerDataEnd``) or
        ``2025`` (first player glued to the header, ``BP_..._C`` classes, no end marker)
    :return: the response text
    :raises ValueError: for an unknown style
    """
    if style not in FORMATS:
        raise ValueError(f"unknown format style {style!r}")
    header = f"[{now:%Y.%m.%d-%H.%M.%S}] PlayerData"
    vitals = "Health: 1.00, Stamina: 1.00, Hunger: 0.80, Thirst: 0.70"
    lines = []
    for r in rows:
        location = f"Location: X={r.x:.3f} Y={r.y:.3f} Z={r.z:.3f}"
        if style == "2026":
            lines.append(
                f"Name: {r.name}, PlayerID: {r.player_id}, Gender: Female, {location}, Class: {r.dino_class}, "
                f"Growth: {r.growth:.2f}, {vitals}, MutationSlots: {_SLOTS}, ParentMutationSlots: {_SLOTS}, "
                "ElderMutationSlotsA: [1=None,2=None], ElderMutationSlotsB: [1=None,2=None], PrimeElder: false"
            )
        else:
            lines.append(
                f"Name: {r.name}, PlayerID: {r.player_id}, {location}, Class: BP_{r.dino_class}_C, "
                f"Growth: {r.growth:.2f}, {vitals}"
            )
    if style == "2025":
        # The first player follows the header directly: "...] PlayerDataName: ...".
        return header + "\n".join(lines) + "\n"
    return "\n".join([header, *lines, "PlayerDataEnd"]) + "\n"


def format_player_list(rows: Iterable[SnapshotRow], now: datetime) -> str:
    """
    Render a ``PLAYER_LIST`` response: a line of ids and a line of names, each item followed by a comma.

    The agent never parses it; ``capture`` saves it for inspection.

    :param rows: the players to include
    :param now: unused; the real response has no timestamp
    :return: the response text
    """
    rows = list(rows)
    ids = "".join(f"{r.player_id}," for r in rows)
    names = "".join(f"{r.name}," for r in rows)
    return f"PlayerList\n{ids}\n{names}\n"
