"""
Parser for the response to the ``PLAYER_DATA`` (0x77) RCON command.

The exact format has not been verified against a real server yet. The expected shape, inferred from the
butt4cak3/theislercon Go client (the only reference implementation that parses it), is one line per player
(wrapped here)::

    [2025.06.01-12.34.56] PlayerData
    Name: Some, Name, PlayerID: 76561198000000000, Location: X=-12345.6 Y=78901.2 Z=345.0,
        Class: BP_Carnotaurus_C, Growth: 0.75, Health: 1.0, Stamina: 0.9, Hunger: 0.8, Thirst: 0.7

The parser is deliberately more tolerant than that shape (every tolerance is an assumption listed in NOTES.md):

* the ``[timestamp] PlayerData`` header is optional, and the first player may follow it on the same line;
* the name runs up to the *last* ``, PlayerID:`` on the line, because names may contain commas;
* the fields after the id may come in any order, unknown fields are ignored, and ``Class``/``Growth`` are optional;
* ``Location`` may be ``X=.. Y=.. Z=..``, comma separated, or wrapped in parentheses;
* ``Class`` may be the blueprint name (``BP_Carnotaurus_C``) or the bare class name;
* ``Growth`` may be a fraction (``0.75``) or a percentage (``75%``).

A line that cannot be parsed never raises. It becomes a :class:`LineError` whose ``shape`` lists only the field
*keys* found on the line, never values, so errors can be logged without logging personal data.
"""

import math
import re
from dataclasses import dataclass, field
from datetime import datetime

_FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_HEADER = re.compile(r"^\s*(?:\[(?P<ts>[^\]\n]*)\])?\s*(?:PlayerData\b:?)?", re.IGNORECASE)
_PLAYER_ID_SEP = re.compile(r",\s*PlayerID\s*:\s*", re.IGNORECASE)
# Splits "id, Location: ..., Class: ..." at commas that start a new "Key:" field. Commas inside a value that are
# not followed by "Key:" (e.g. "X=1, Y=2") are left alone.
_FIELD_SEP = re.compile(r",\s*(?=[A-Za-z][A-Za-z ]*:)")
_KEY_VALUE = re.compile(r"^\s*(?P<key>[A-Za-z][A-Za-z ]*?)\s*:\s*(?P<value>.*?)\s*$", re.DOTALL)
_AXIS = {axis: re.compile(rf"\b{axis}\s*=\s*(?P<v>{_FLOAT})", re.IGNORECASE) for axis in "XYZ"}
_PLAYER_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_CLASS = re.compile(r"^(?:BP_)?(?P<name>[A-Za-z0-9_]+?)(?:_C)?$")
_GROWTH = re.compile(rf"^(?P<v>{_FLOAT})\s*(?P<pct>%)?$")
_SHAPE_KEYS = re.compile(r"([A-Za-z]+)\s*[:=]")
_TIMESTAMP_FORMAT = "%Y.%m.%d-%H.%M.%S"


@dataclass(frozen=True, slots=True)
class ParsedPlayer:
    player_id: str
    player_name: str
    x: float
    y: float
    z: float
    dino_class: str | None = None
    growth: float | None = None


@dataclass(frozen=True, slots=True)
class LineError:
    line_no: int
    reason: str
    shape: str  # field keys only, e.g. "Name PlayerID Location X Y Z"; safe to log


@dataclass(slots=True)
class ParseResult:
    players: list[ParsedPlayer] = field(default_factory=list)
    errors: list[LineError] = field(default_factory=list)
    header_timestamp: datetime | None = None  # naive, server-local; informational only


class _LineParseError(ValueError):
    pass


def parse_player_data(text: str) -> ParseResult:
    """
    Parse a ``PLAYER_DATA`` response.

    :param text: the decoded response
    :return: the players that parsed, plus one error per line that did not
    """
    result = ParseResult()
    text = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    header = _HEADER.match(text)
    if header:
        result.header_timestamp = _parse_timestamp(header.group("ts"))
        text = text[header.end() :]

    seen: set[str] = set()
    for line_no, raw_line in enumerate(text.split("\n"), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            player = _parse_line(line)
        except _LineParseError as e:
            result.errors.append(LineError(line_no=line_no, reason=str(e), shape=line_shape(line)))
            continue
        if player.player_id in seen:
            result.errors.append(LineError(line_no=line_no, reason="duplicate player id", shape=line_shape(line)))
            continue
        seen.add(player.player_id)
        result.players.append(player)
    return result


def line_shape(line: str) -> str:
    """
    Describe a line by its field keys only, so it can be logged without names, ids or positions.

    :param line: a raw response line
    :return: the keys in order of appearance, space separated
    """
    return " ".join(_SHAPE_KEYS.findall(line))[:200]


def _parse_line(line: str) -> ParsedPlayer:
    if not line.lower().startswith("name:"):
        raise _LineParseError("line does not start with 'Name:'")
    separators = list(_PLAYER_ID_SEP.finditer(line))
    if not separators:
        raise _LineParseError("no PlayerID field")
    sep = separators[-1]
    name = line[len("name:") : sep.start()].strip()
    rest = line[sep.end() :]

    chunks = _FIELD_SEP.split(rest)
    # The id ends at the first comma, even if what follows is not a well-formed field (e.g. a truncated line).
    player_id = chunks[0].split(",", 1)[0].strip()
    if not _PLAYER_ID.match(player_id):
        raise _LineParseError("PlayerID is not id-shaped")

    fields: dict[str, str] = {}
    for chunk in chunks[1:]:
        kv = _KEY_VALUE.match(chunk)
        if kv:
            fields[kv.group("key").strip().lower()] = kv.group("value")

    if "location" not in fields:
        raise _LineParseError("no Location field")
    x, y, z = _parse_location(fields["location"])
    return ParsedPlayer(
        player_id=player_id,
        player_name=name,
        x=x,
        y=y,
        z=z,
        dino_class=_parse_class(fields.get("class")),
        growth=_parse_growth(fields.get("growth")),
    )


def _parse_location(value: str) -> tuple[float, float, float]:
    coords = []
    for axis, pattern in _AXIS.items():
        m = pattern.search(value)
        if not m:
            raise _LineParseError(f"Location has no {axis} coordinate")
        v = float(m.group("v"))
        if not math.isfinite(v):
            raise _LineParseError(f"Location {axis} is not finite")
        coords.append(v)
    return coords[0], coords[1], coords[2]


def _parse_class(value: str | None) -> str | None:
    if not value:
        return None
    m = _CLASS.match(value.strip())
    return m.group("name") if m else None


def _parse_growth(value: str | None) -> float | None:
    if not value:
        return None
    m = _GROWTH.match(value.strip())
    if not m:
        return None
    v = float(m.group("v"))
    if m.group("pct"):
        v /= 100.0
    return v if 0.0 <= v <= 1.0 else None


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.strptime(value.strip(), _TIMESTAMP_FORMAT)
    except ValueError:
        return None
