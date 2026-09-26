import re
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

import pytest

from server.alerts.message import (
    DISCLAIMER,
    EMBED_FIELD_VALUE_MAX,
    EMBED_FIELDS_MAX,
    EMBED_TITLE_MAX,
    EMBED_TOTAL_MAX,
    Alert,
    behaviour_lines,
    clean_text,
    discord_payload,
    email_content,
    embed_length,
    escape_discord,
    truncate,
)

DETAILS: dict[str, Any] = {
    "servers": ["gateway-1"],
    "beeline": {
        "score": 1.0,
        "episodes": 14,
        "null_episodes": 2.31,
        "z": 6.42,
        "moving_minutes": 312.5,
        "mean_start_distance_m": 410,
    },
    "ambush": {"score": 0.4, "waits": 23, "hits": 8, "null_hits": 1.27, "z": 4.6},
    "ttc": {
        "score": 0.7,
        "episodes": 9,
        "contacts": 7,
        "mean_rank": 0.12,
        "z": 3.9,
        "median_ttc_s": 145,
        "server_median_ttc_s": 350,
    },
}


def make_alert(**changes: Any) -> Alert:
    alert = Alert(
        kind="flag",
        org_id="demo",
        player_id="76561198000000001",
        player_name="Rex",
        server_ids=("gateway-1",),
        score=0.87,
        details=DETAILS,
        created_at=datetime(2026, 9, 26, 12, 30, tzinfo=UTC),
        flag_id=42,
    )
    return replace(alert, **changes)


def test_behaviour_lines_explain_the_numbers() -> None:
    lines = behaviour_lines(DETAILS)
    assert lines == [
        "Beeline: 14 walks straight to players out of sight vs 2.3 expected (in 312 min of moving, z 6.4)",
        "Ambush: 8 of 23 waits ended with a distant player arriving vs 1.3 expected (z 4.6)",
        "Time to contact: median 145 s after spawning vs 350 s server median (7 contacts in 9 spawns, z 3.9)",
    ]


@pytest.mark.parametrize("name", ["beeline", "ambush", "ttc"])
def test_behaviour_lines_skip_zero_and_absent(name: str) -> None:
    zeroed = {**DETAILS, name: {**DETAILS[name], "score": 0.0}}
    absent = {k: v for k, v in DETAILS.items() if k != name}
    for details in (zeroed, absent):
        lines = behaviour_lines(details)
        assert len(lines) == 2
    assert behaviour_lines({"servers": []}) == []


def test_behaviour_lines_tolerate_missing_numbers_and_no_contact() -> None:
    assert behaviour_lines({"beeline": {"score": 0.5}}) == ["Beeline: ? walks straight to players out of sight"]
    ttc = {"score": 0.5, "median_ttc_s": -1, "server_median_ttc_s": 900}
    assert behaviour_lines({"ttc": ttc}) == ["Time to contact: no contact after spawning vs 15 min server median"]
    assert behaviour_lines({"ambush": {"score": True}, "beeline": "junk"}) == []


def test_escape_discord_neutralises_markdown_and_mentions() -> None:
    for raw in ("@everyone", "@here", "<@123456>", "<@&987>"):
        escaped = escape_discord(raw)
        assert "@everyone" not in escaped
        assert "@here" not in escaped
        assert re.search(r"<@[\d&!]", escaped) is None
    assert escape_discord("**bold** _it_ `code` [x](http://e.vil)") == (
        r"\*\*bold\*\* \_it\_ \`code\` \[x\]\(http\://e\.vil\)"
    )


def test_clean_text_removes_line_breaks_and_caps_length() -> None:
    assert clean_text("a\nb\r\nc d") == "a b  c d"
    assert len(clean_text("x" * 500)) == 64
    assert clean_text("\n") == "(no name)"


def test_truncate_never_leaves_a_dangling_escape() -> None:
    text = escape_discord("a" * 8 + "*" * 20)
    for limit in range(2, len(text)):
        cut = truncate(text, limit)
        assert len(cut) <= limit
        body = cut.removesuffix("…")
        assert (len(body) - len(body.rstrip("\\"))) % 2 == 0


def test_discord_payload_flag() -> None:
    payload = discord_payload(make_alert())
    assert payload["allowed_mentions"] == {"parse": []}
    (embed,) = payload["embeds"]
    assert embed["title"] == "Possible ESP user: Rex"
    assert embed["timestamp"] == "2026-09-26T12:30:00+00:00"
    fields = {f["name"]: f["value"] for f in embed["fields"]}
    assert fields["Player"] == "Rex\n`76561198000000001`"
    assert fields["Server"] == "gateway\\-1"
    assert fields["Score"] == "0.87"
    assert "14 walks straight to players" in fields["Behaviours"]
    assert DISCLAIMER in embed["footer"]["text"]
    assert "#42" in embed["footer"]["text"]
    assert "image" not in embed
    assert "attachments" not in payload


def test_discord_payload_rejoin_differs_from_flag() -> None:
    flag = discord_payload(make_alert())["embeds"][0]
    rejoin = discord_payload(make_alert(kind="rejoin"))["embeds"][0]
    assert rejoin["title"].startswith("Flagged player rejoined")
    assert "joined a server again" in rejoin["description"]
    assert rejoin["color"] != flag["color"]


def test_discord_payload_references_the_image() -> None:
    payload = discord_payload(make_alert(image_png=b"\x89PNG"))
    assert payload["embeds"][0]["image"] == {"url": "attachment://evidence.png"}
    assert payload["attachments"] == [{"id": 0, "filename": "evidence.png"}]


def test_discord_payload_escapes_hostile_names() -> None:
    payload = discord_payload(make_alert(player_name="@everyone **FREE NITRO** <@1>\n# big"))
    embed = payload["embeds"][0]
    for text in (embed["title"], embed["fields"][0]["value"]):
        assert "@everyone" not in text
        assert "<@1>" not in text
        assert "**" not in text
        assert "\n#" not in text


def test_discord_payload_respects_limits() -> None:
    lines = {f"b{i}": {"score": 1.0} for i in range(100)}
    alert = make_alert(
        player_name="W" * 10_000,
        player_id="9" * 10_000,
        server_ids=tuple(f"server-{i:04d}-" + "x" * 40 for i in range(500)),
        details={**DETAILS, **lines, "beeline": {**DETAILS["beeline"], "episodes": 10**900}},
    )
    embed = discord_payload(alert)["embeds"][0]
    assert len(embed["title"]) <= EMBED_TITLE_MAX
    assert len(embed["fields"]) <= EMBED_FIELDS_MAX
    assert all(len(f["value"]) <= EMBED_FIELD_VALUE_MAX for f in embed["fields"])
    assert embed_length(embed) <= EMBED_TOTAL_MAX
    servers = next(f["value"] for f in embed["fields"] if f["name"] == "Servers")
    assert servers.endswith("more")


def test_email_content_has_the_key_information() -> None:
    content = email_content(make_alert(player_name="Rex\r\nBcc: victim@example.com"))
    assert "\n" not in content.subject
    assert "\r" not in content.subject
    assert content.subject.startswith("[esp-killer] Possible ESP user: Rex")
    assert "(demo)" in content.subject
    for needle in ("76561198000000001", "gateway-1", "0.87", "#42", "14 walks straight", "145 s"):
        assert needle in content.text
    assert DISCLAIMER in content.text
    assert "attached image" not in content.text


def test_email_content_rejoin_and_html_escaping() -> None:
    content = email_content(make_alert(kind="rejoin", player_name="<script>x</script>", image_png=b"png"))
    assert content.subject.startswith("[esp-killer] Flagged player rejoined")
    assert "joined a server again" in content.text
    assert "attached image" in content.text
    assert "<script>" not in content.html
    assert "&lt;script&gt;" in content.html
