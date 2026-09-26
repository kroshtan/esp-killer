"""
Parser for the response to the ``PLAYER_DATA`` (0x77) RCON command.

Nobody has published a complete raw response, so the format is pieced together from the parsers, regexes, test
fixtures and logs of about a dozen independent Evrima tools (see NOTES.md, "RCON format evidence"). Two
generations are known, one line per spawned player::

    2025:  [2025.06.01-12.34.56] PlayerDataName: A, PlayerID: 76561198000000001, Location: X=1.0 Y=2.0 Z=3.0,
               Class: BP_Carnotaurus_C, Growth: 0.75, Health: 1.00, Stamina: 1.00, Hunger: 0.80, Thirst: 0.70
           Name: B, ...

    2026:  [2026.04.01-12.34.56] PlayerData
           Name: A, PlayerID: 76561198000000001, Gender: Male, Location: X=483129.590 Y=-76220.472 Z=23256.202,
               Class: Tyrannosaurus, Growth: 1.00, Health: 1.00, ..., MutationSlots: [1=None,2=None,3=None,4=None],
               ..., PrimeElder: false
           PlayerDataEnd

(wrapped here). The parser takes fields by key, so both work, and it tolerates more:

* the ``[timestamp]`` and ``PlayerData`` header are optional, and the first player may be glued to the header;
* parsing stops at ``PlayerDataEnd``;
* the name runs up to the *last* ``, PlayerID:`` on the line, because names may contain commas;
* fields may come in any order; unknown fields are ignored (and reported by name in :attr:`ParseResult.unknown_keys`),
  commas inside a value (``[1=None,2=None]``, ``X=1, Y=2``) do not split fields;
* ``Location`` may be ``X=.. Y=.. Z=..``, comma separated, or wrapped in parentheses;
* ``Class`` may be the blueprint name (``BP_Carnotaurus_C``) or the bare class name;
* ``Growth`` may be a fraction (``0.75``) or a percentage (``75%``).

A line that does not fit is *salvaged* if it still contains an id-shaped token (Steam64 or EOS) and an
``X= Y= Z=`` triple, so a small format change degrades detection instead of silencing it. Either way the line
becomes a :class:`LineError` whose ``shape`` lists only the field *keys* found on the line, never values, so
problems can be logged and reported without personal data.
"""

import math
import re
from dataclasses import dataclass, field
from datetime import datetime

_FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
# No word boundary after "PlayerData": 2025 servers glue the first player to it ("PlayerDataName: ...").
_HEADER = re.compile(r"^\s*(?:\[(?P<ts>[^\]\n]*)\])?\s*(?:PlayerData(?!End):?)?", re.IGNORECASE)
END_MARKER = "PlayerDataEnd"
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
_SALVAGE_ID = re.compile(r"(?<![0-9A-Za-z])(7656119\d{10}|[0-9a-fA-F]{32})(?![0-9A-Za-z])")
_SALVAGE_NAME = re.compile(r"Name\s*:\s*(?P<name>.*?)\s*,\s*PlayerID", re.IGNORECASE)
# Keys we know about but do not use; anything else is reported as unknown, to spot format changes.
KNOWN_KEYS = frozenset(
    {
        "location",
        "class",
        "growth",
        "health",
        "stamina",
        "hunger",
        "thirst",
        "gender",
        "mutationslots",
        "parentmutationslots",
        "eldermutationslotsa",
        "eldermutationslotsb",
        "primeelder",
    }
)
_MAX_UNKNOWN_KEYS = 20
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
    salvaged: bool = False  # the player was still extracted, from the id and X/Y/Z alone


@dataclass(slots=True)
class ParseResult:
    players: list[ParsedPlayer] = field(default_factory=list)
    errors: list[LineError] = field(default_factory=list)
    header_timestamp: datetime | None = None  # naive, server-local; informational only
    ended: bool = False  # the PlayerDataEnd marker was present
    unknown_keys: set[str] = field(default_factory=set)  # field names only, never values


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
        if line.lower() == END_MARKER.lower():
            result.ended = True
            break
        try:
            player = _parse_line(line, result.unknown_keys)
        except _LineParseError as e:
            salvaged = _salvage(line)
            result.errors.append(
                LineError(line_no=line_no, reason=str(e), shape=line_shape(line), salvaged=salvaged is not None)
            )
            if salvaged is None:
                continue
            player = salvaged
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


def _parse_line(line: str, unknown_keys: set[str]) -> ParsedPlayer:
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
            key = kv.group("key").strip()
            fields[key.lower()] = kv.group("value")
            if key.lower() not in KNOWN_KEYS and len(unknown_keys) < _MAX_UNKNOWN_KEYS:
                unknown_keys.add(key[:40])

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


def _salvage(line: str) -> ParsedPlayer | None:
    """Last resort for a line that does not fit the format: an id-shaped token plus an X/Y/Z triple."""
    ids = _SALVAGE_ID.findall(line)
    if len(ids) != 1:
        return None  # none, or ambiguous
    try:
        x, y, z = _parse_location(line)
    except _LineParseError:
        return None
    name = _SALVAGE_NAME.search(line)
    return ParsedPlayer(player_id=ids[0], player_name=name.group("name") if name else "", x=x, y=y, z=z)


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
