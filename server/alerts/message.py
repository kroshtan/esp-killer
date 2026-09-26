"""
What an alert says, independent of how it is delivered.

Alerts go to server admins, who decide whether a player is cheating. The message therefore has to explain the
numbers behind a score in plain words ("14 walks straight to players out of sight vs 2.3 expected"), and say
clearly that a score is evidence for review, not proof.

Everything here is pure: :class:`Alert` in, Discord payload or email text out. Player names are chosen by the
players, so anywhere they end up in Discord markdown they are escaped and their mentions defused, and anywhere
they end up in an email header their line breaks are removed.
"""

import html
import math
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal, TypeGuard

AlertKind = Literal["flag", "rejoin"]

# The name the evidence image is sent under, both as the Discord attachment and the email attachment.
IMAGE_FILENAME = "evidence.png"

DISCLAIMER = "Scores are evidence for human review, not proof of cheating. Nothing has been kicked or banned."

# Embed colours: red for a new flag, amber for a flagged player showing up again.
COLOURS: dict[AlertKind, int] = {"flag": 0xC0392B, "rejoin": 0xE67E22}

# Discord's embed limits (https://discord.com/developers/docs/resources/message#embed-object-embed-limits).
EMBED_TITLE_MAX = 256
EMBED_DESCRIPTION_MAX = 4096
EMBED_FIELD_NAME_MAX = 256
EMBED_FIELD_VALUE_MAX = 1024
EMBED_FOOTER_MAX = 2048
EMBED_FIELDS_MAX = 25
EMBED_TOTAL_MAX = 6000

# Names are cut to this before anything else. Evrima names are short; this only stops abuse.
NAME_MAX = 64

ELLIPSIS = "…"

# Every character Discord markdown gives a meaning to, including "<" (mentions, custom emoji, timestamps) and ":"
# (emoji shortcodes, autolinks).
_MARKDOWN_SPECIAL = re.compile(r"([\\*_~`|>#<\[\]()\-:.!+=])")
_CONTROL = re.compile(r"[\x00-\x1f\x7f-\x9f  ]")
_ZERO_WIDTH_SPACE = "\u200b"


@dataclass(frozen=True)
class Alert:
    """
    One alert about one player, ready to be delivered.

    ``kind`` is ``"flag"`` when the player's score crossed the threshold, ``"rejoin"`` when an already flagged
    player joined one of the org's servers again. ``details`` is :meth:`PlayerScore.details` as stored with the
    flag. ``flag_id`` identifies the flag the alert belongs to, so admins can refer to it (e.g. to mark a false
    positive).
    """

    kind: AlertKind
    org_id: str
    player_id: str
    player_name: str
    server_ids: tuple[str, ...]
    score: float
    details: dict[str, Any]
    created_at: datetime
    flag_id: int
    image_png: bytes | None = None


@dataclass(frozen=True)
class EmailContent:
    subject: str
    text: str
    html: str


# --- behaviour lines -----------------------------------------------------------------------------------------


def behaviour_lines(details: dict[str, Any]) -> list[str]:
    """
    One plain-English line per behaviour that contributed to the score.

    Behaviours that are absent (not enough evidence) or scored 0 are skipped. Missing numbers are tolerated, since
    ``details`` may have been stored by an older version.

    :param details: :meth:`PlayerScore.details`
    :return: lines such as ``"Beeline: 14 walks straight to players out of sight vs 2.3 expected (z 6.4)"``
    """
    lines = []
    for name, describe in (("beeline", _beeline_line), ("ambush", _ambush_line), ("ttc", _ttc_line)):
        sub = details.get(name)
        if not isinstance(sub, dict) or not _positive(sub.get("score")):
            continue
        lines.append(describe(sub))
    return lines


def _beeline_line(sub: dict[str, Any]) -> str:
    text = f"Beeline: {_num(sub.get('episodes'))} walks straight to players out of sight"
    if "null_episodes" in sub:
        text += f" vs {_num(sub['null_episodes'], 1)} expected"
    extras = []
    if "moving_minutes" in sub:
        extras.append(f"in {_num(sub['moving_minutes'])} min of moving")
    if "z" in sub:
        extras.append(f"z {_num(sub['z'], 1)}")
    return text + (f" ({', '.join(extras)})" if extras else "")


def _ambush_line(sub: dict[str, Any]) -> str:
    text = f"Ambush: {_num(sub.get('hits'))} of {_num(sub.get('waits'))} waits ended with a distant player arriving"
    if "null_hits" in sub:
        text += f" vs {_num(sub['null_hits'], 1)} expected"
    if "z" in sub:
        text += f" (z {_num(sub['z'], 1)})"
    return text


def _ttc_line(sub: dict[str, Any]) -> str:
    median = sub.get("median_ttc_s")
    server_median = sub.get("server_median_ttc_s")
    # -1 means "no contact at all" (player) or "nobody made contact" (server).
    if _is_number(median) and median >= 0:
        text = f"Time to contact: median {_duration(median)} after spawning"
    else:
        text = "Time to contact: no contact after spawning"
    if _is_number(server_median) and server_median >= 0:
        text += f" vs {_duration(server_median)} server median"
    extras = []
    if "episodes" in sub:
        contacts = f"{_num(sub['contacts'])} contacts in " if "contacts" in sub else ""
        extras.append(f"{contacts}{_num(sub['episodes'])} spawns")
    if "z" in sub:
        extras.append(f"z {_num(sub['z'], 1)}")
    return text + (f" ({', '.join(extras)})" if extras else "")


def _is_number(value: Any) -> TypeGuard[int | float]:
    """A real, finite number of sane size (details come from storage, so don't trust them to be)."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    if isinstance(value, int):
        return abs(value) < 10**15
    return math.isfinite(value)


def _positive(value: Any) -> bool:
    return _is_number(value) and value > 0


def _num(value: Any, decimals: int = 0) -> str:
    """Format a number from ``details``; ints stay ints, floats get ``decimals`` places (at most)."""
    if not _is_number(value):
        return "?"
    if isinstance(value, int) or decimals == 0:
        return str(round(value))
    return f"{value:.{decimals}f}"


def _duration(seconds: float) -> str:
    if seconds < 600:  # noqa: PLR2004
        return f"{round(seconds)} s"
    return f"{seconds / 60:.0f} min"


# --- sanitising ----------------------------------------------------------------------------------------------


def clean_text(text: str, limit: int = NAME_MAX) -> str:
    """
    Make user-controlled text safe to put on one line: control characters and line breaks become spaces.

    :param text: e.g. a player name
    :param limit: maximum length of the result
    :return: the cleaned, truncated text
    """
    return truncate(_CONTROL.sub(" ", text).strip(), limit) or "(no name)"


def escape_discord(text: str) -> str:
    """
    Escape Discord markdown and defuse mentions in user-controlled text.

    Markdown characters get a backslash. ``@everyone``, ``@here`` and ``<@123>`` would not ping anyone with
    ``allowed_mentions`` empty, but they would still render as mentions, so a zero-width space goes after every
    ``@`` too.

    :param text: the raw text
    :return: text that renders literally
    """
    escaped = _MARKDOWN_SPECIAL.sub(r"\\\1", text)
    return escaped.replace("@", "@" + _ZERO_WIDTH_SPACE)


def truncate(text: str, limit: int) -> str:
    """
    Cut text to at most ``limit`` characters, ending with an ellipsis if anything was cut.

    Never leaves a dangling escape backslash, which would escape the ellipsis.

    :param text: the text
    :param limit: maximum length, at least 1
    :return: the text, possibly shortened
    """
    if len(text) <= limit:
        return text
    cut = text[: limit - len(ELLIPSIS)]
    trailing = len(cut) - len(cut.rstrip("\\"))
    if trailing % 2:
        cut = cut[:-1]
    return cut + ELLIPSIS


def _join_limited(items: list[str], limit: int, sep: str = ", ") -> str:
    """Join items, replacing the tail with "+N more" if they don't fit in ``limit`` characters."""
    for shown in range(len(items), 0, -1):
        text = sep.join(items[:shown])
        if shown < len(items):
            text += f"{sep}+{len(items) - shown} more"
        if len(text) <= limit:
            return text
    return truncate(items[0], limit) if items else ""


# --- wording shared by Discord and email ---------------------------------------------------------------------


def headline(alert: Alert) -> str:
    """
    The alert's one-line title, without the player name.

    :param alert: the alert
    :return: e.g. ``"Possible ESP user"``
    """
    return "Possible ESP user" if alert.kind == "flag" else "Flagged player rejoined"


def summary(alert: Alert) -> str:
    """
    One or two sentences saying what happened, in plain text (no user-controlled content).

    :param alert: the alert
    :return: the summary
    """
    if alert.kind == "flag":
        return (
            f"This player's movement scored {alert.score:.2f}, above the alert threshold. "
            "Please review their play before taking any action."
        )
    return (
        f"A player flagged earlier (flag #{alert.flag_id}, score {alert.score:.2f}) has joined a server again. "
        "You may want to watch them now."
    )


def _timestamp(alert: Alert) -> str:
    return alert.created_at.isoformat(timespec="seconds")


# --- Discord -------------------------------------------------------------------------------------------------


def discord_payload(alert: Alert) -> dict[str, Any]:
    """
    Build the Discord webhook payload (``payload_json`` when an image is attached).

    The embed stays within Discord's limits however long the name or server list is. Mentions are disabled
    outright with ``allowed_mentions``.

    :param alert: the alert
    :return: JSON-serialisable payload
    """
    name = escape_discord(clean_text(alert.player_name))
    player_id = clean_text(alert.player_id).replace("`", "'")
    servers = [escape_discord(clean_text(s)) for s in alert.server_ids] or ["(unknown)"]
    # Behaviour lines are our own text (numbers from details), so they need no escaping.
    lines = behaviour_lines(alert.details)
    behaviours = _join_limited(lines, EMBED_FIELD_VALUE_MAX, sep="\n") if lines else "No behaviour details."

    embed: dict[str, Any] = {
        "title": truncate(f"{headline(alert)}: {name}", EMBED_TITLE_MAX),
        "description": truncate(summary(alert), EMBED_DESCRIPTION_MAX),
        "color": COLOURS[alert.kind],
        "timestamp": _timestamp(alert),
        "fields": [
            {"name": "Player", "value": truncate(f"{name}\n`{player_id}`", EMBED_FIELD_VALUE_MAX), "inline": True},
            {
                "name": "Server" if len(servers) == 1 else "Servers",
                "value": _join_limited(servers, EMBED_FIELD_VALUE_MAX),
                "inline": True,
            },
            {"name": "Score", "value": f"{alert.score:.2f}", "inline": True},
            {"name": "Behaviours", "value": behaviours, "inline": False},
        ][:EMBED_FIELDS_MAX],
        "footer": {
            # Footers don't render markdown, and org ids are slugs.
            "text": truncate(f"{alert.org_id} · flag #{alert.flag_id} · {DISCLAIMER}", EMBED_FOOTER_MAX)
        },
    }
    payload: dict[str, Any] = {"embeds": [embed], "allowed_mentions": {"parse": []}}
    if alert.image_png is not None:
        embed["image"] = {"url": f"attachment://{IMAGE_FILENAME}"}
        payload["attachments"] = [{"id": 0, "filename": IMAGE_FILENAME}]
    _fit_total(embed)
    return payload


def embed_length(embed: dict[str, Any]) -> int:
    """
    The length Discord counts against the 6000-character embed limit.

    :param embed: an embed object
    :return: total characters in title, description, field names and values, footer text and author name
    """
    total = len(embed.get("title", "")) + len(embed.get("description", ""))
    total += len(embed.get("footer", {}).get("text", "")) + len(embed.get("author", {}).get("name", ""))
    return total + sum(len(f["name"]) + len(f["value"]) for f in embed.get("fields", []))


def _fit_total(embed: dict[str, Any]) -> None:
    """Shorten the longest field values until the embed fits Discord's total limit (in place)."""
    while (excess := embed_length(embed) - EMBED_TOTAL_MAX) > 0:
        longest = max(embed["fields"], key=lambda f: len(f["value"]))
        if len(longest["value"]) <= len(ELLIPSIS):  # pragma: no cover - cannot happen with the fields above
            break
        longest["value"] = truncate(longest["value"], max(len(ELLIPSIS), len(longest["value"]) - excess))


# --- email ---------------------------------------------------------------------------------------------------


def email_content(alert: Alert) -> EmailContent:
    """
    Build the email subject, plain-text body and HTML body.

    :param alert: the alert
    :return: subject and bodies
    """
    name = clean_text(alert.player_name)
    player_id = clean_text(alert.player_id)
    servers = ", ".join(clean_text(s) for s in alert.server_ids) or "(unknown)"
    lines = behaviour_lines(alert.details) or ["No behaviour details."]
    subject = clean_text(f"[esp-killer] {headline(alert)}: {name} ({alert.org_id})", 200)

    facts = [
        ("Player", f"{name} ({player_id})"),
        ("Server" if len(alert.server_ids) == 1 else "Servers", servers),
        ("Score", f"{alert.score:.2f}"),
        ("Flag", f"#{alert.flag_id}"),
        ("Time", _timestamp(alert)),
    ]
    image_note = "The attached image shows the movement behind the score." if alert.image_png is not None else ""

    text_parts = [
        summary(alert),
        "",
        *(f"{label}: {value}" for label, value in facts),
        "",
        "Behaviours:",
        *(f"- {line}" for line in lines),
        "",
    ]
    if image_note:
        text_parts += [image_note, ""]
    text_parts.append(DISCLAIMER)
    text = "\n".join(text_parts) + "\n"

    e = html.escape
    rows = "".join(f"<tr><th align='left'>{e(label)}</th><td>{e(value)}</td></tr>" for label, value in facts)
    items = "".join(f"<li>{e(line)}</li>" for line in lines)
    html_body = (
        f"<html><body><h2>{e(headline(alert))}: {e(name)}</h2><p>{e(summary(alert))}</p>"
        f"<table>{rows}</table><h3>Behaviours</h3><ul>{items}</ul>"
        + (f"<p>{e(image_note)}</p>" if image_note else "")
        + f"<p><small>{e(DISCLAIMER)}</small></p></body></html>"
    )
    return EmailContent(subject=subject, text=text, html=html_body)
