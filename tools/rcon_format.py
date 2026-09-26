"""
Render RCON responses in the format the parser expects (see shared/playerdata.py).

Used by the fake RCON server and by the parser's round-trip tests. If a real server turns out to format things
differently, update this module and the fixtures in tests/fixtures/rcon/ together.
"""

from collections.abc import Iterable
from datetime import datetime

from tools.sim.world import SnapshotRow


def format_player_data(rows: Iterable[SnapshotRow], now: datetime) -> str:
    """
    Render a ``PLAYER_DATA`` response, including the vitals fields the agent is expected to drop.

    :param rows: the players to include
    :param now: server-local time for the header
    :return: the response text
    """
    lines = [f"[{now:%Y.%m.%d-%H.%M.%S}] PlayerData"]
    lines.extend(
        f"Name: {r.name}, PlayerID: {r.player_id}, Location: X={r.x:.3f} Y={r.y:.3f} Z={r.z:.3f}, "
        f"Class: BP_{r.dino_class}_C, Growth: {r.growth:.2f}, Health: 1.00, Stamina: 1.00, Hunger: 0.80, "
        f"Thirst: 0.70"
        for r in rows
    )
    return "\n".join(lines) + "\n"


def format_player_list(rows: Iterable[SnapshotRow], now: datetime) -> str:
    """
    Render a ``PLAYER_LIST`` response. Its format is even less certain than player data; the agent never parses it.

    :param rows: the players to include
    :param now: server-local time for the header
    :return: the response text
    """
    body = "".join(f"{r.player_id},{r.name},\n" for r in rows)
    return f"[{now:%Y.%m.%d-%H.%M.%S}] PlayerList\n{body}"
