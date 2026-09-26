"""
Alerting end to end.

Simulated positions -> worker -> flags -> Discord alerts with an evidence image, rejoin alerts, retries, and
retention.
"""

import json
import shutil
import uuid
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest

from server.alerts.pipeline import channels_for
from server.db.database import Database
from server.db.repository import Repository
from server.orgconfig import AlertDestinations, OrgConfig, OrgEntry, ServerEntry, save_config
from server.settings import ServerSettings
from server.worker import RunReport, Worker
from shared.models import PlayerSample, Snapshot
from tests.helpers import ingest_frame
from tools.sim.scenarios import archetypes, mixed_world, record

ORG = "sim-org"
START = datetime(2026, 1, 1, tzinfo=UTC)
HOURS = 4.5
WEBHOOK = "https://discord.com/api/webhooks/123/secret-token"


class FakeDiscord:
    def __init__(self, respond: Callable[[httpx.Request], httpx.Response] | None = None) -> None:
        self.requests: list[httpx.Request] = []
        self.respond = respond or (lambda _: httpx.Response(204))

    def client(self) -> httpx.AsyncClient:
        def handler(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            return self.respond(request)

        return httpx.AsyncClient(transport=httpx.MockTransport(handler))

    def payloads(self) -> list[dict[str, object]]:
        out = []
        for r in self.requests:
            body = r.content
            if r.headers["content-type"].startswith("multipart/"):
                start = body.index(b"{")
                end = body.index(b"\r\n--", start)
                out.append(json.loads(body[start:end]))
            else:
                out.append(json.loads(body))
        return out


@pytest.fixture(scope="module")
def template(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, str]]:
    """A database with 4.5 simulated hours of positions, built once and copied for each test."""
    path = tmp_path_factory.mktemp("template") / "espk.db"
    world = mixed_world(900)
    db = Database(path)
    ingest_frame(Repository(db), record(world, HOURS * 3600), ORG, "s1", START)
    with db.connect() as conn:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")  # everything in the main file, so a plain copy is complete
    return path, archetypes(world)


@pytest.fixture
def worker(tmp_path: Path, template: tuple[Path, dict[str, str]]) -> Iterator[tuple[Worker, dict[str, str]]]:
    config_path = tmp_path / "config.yaml"
    org = OrgEntry(alert=AlertDestinations.model_validate({"discord_webhook": WEBHOOK}), servers={"s1": ServerEntry()})
    save_config(config_path, OrgConfig(orgs={ORG: org}))
    shutil.copy(template[0], tmp_path / "espk.db")
    settings = ServerSettings(config_path=config_path, database_path=tmp_path / "espk.db", scoring_lag_s=300)
    yield Worker(settings), template[1]


def run(worker: Worker, now: datetime, discord: FakeDiscord) -> RunReport:
    return worker.run_once(now, http=discord.client())


END = START + timedelta(hours=HOURS, minutes=10)


def test_flags_produce_one_discord_alert_each_with_an_image(worker: tuple[Worker, dict[str, str]]) -> None:
    w, arch = worker
    discord = FakeDiscord()
    report = run(w, END, discord)

    flagged = w.scoring.list_flags(ORG)
    assert flagged
    assert {arch[f.player_id] for f in flagged} <= {"beeline_cheater", "ambush_cheater", "subtle_cheater"}
    assert report.queued == len(flagged)
    assert report.delivery == {"sent": len(flagged), "retry": 0, "failed": 0}
    assert all(str(r.url) == WEBHOOK for r in discord.requests)
    assert all(r.headers["content-type"].startswith("multipart/") for r in discord.requests)
    assert all(b"\x89PNG" in r.content for r in discord.requests)
    names = {f.player_name for f in flagged}
    titles = " ".join(json.dumps(p) for p in discord.payloads())
    assert all(name in titles for name in names)

    # Nothing new: no new alerts.
    again = FakeDiscord()
    report = run(w, END + timedelta(minutes=5), again)
    assert report.queued == 0
    assert again.requests == []


def test_flagged_player_rejoining_triggers_a_rejoin_alert(worker: tuple[Worker, dict[str, str]]) -> None:
    w, _ = worker
    run(w, END, FakeDiscord())
    flag = w.scoring.list_flags(ORG)[0]

    # The flagged player shows up on another server after a long break.
    later = END + timedelta(hours=1)
    snapshot = Snapshot(
        snapshot_id=uuid.uuid4(),
        captured_at=later,
        players=[PlayerSample(player_id=flag.player_id, player_name=flag.player_name, x=0, y=0, z=0)],
    )
    Repository(w.db).ingest(ORG, "s2", [snapshot], later)
    discord = FakeDiscord()
    report = run(w, later + timedelta(minutes=1), discord)

    assert report.queued == 1
    (payload,) = discord.payloads()
    assert "s2" in json.dumps(payload)
    assert discord.requests[0].headers["content-type"] == "application/json"  # rejoin alerts carry no image

    # Still online a minute later: not another rejoin.
    Repository(w.db).ingest(
        ORG,
        "s2",
        [snapshot.model_copy(update={"snapshot_id": uuid.uuid4(), "captured_at": later + timedelta(minutes=1)})],
        later,
    )
    assert run(w, later + timedelta(minutes=3), FakeDiscord()).queued == 0


def test_false_positive_flags_do_not_alert_on_rejoin(worker: tuple[Worker, dict[str, str]]) -> None:
    w, _ = worker
    run(w, END, FakeDiscord())
    for f in w.scoring.list_flags(ORG):
        w.scoring.mark_false_positive(f.id, END)
        later = END + timedelta(hours=1)
        snapshot = Snapshot(
            snapshot_id=uuid.uuid4(),
            captured_at=later,
            players=[PlayerSample(player_id=f.player_id, player_name=f.player_name, x=0, y=0, z=0)],
        )
        Repository(w.db).ingest(ORG, "s1", [snapshot], later)
    assert run(w, END + timedelta(hours=1, minutes=1), FakeDiscord()).queued == 0


def test_transient_failures_are_retried_and_permanent_ones_are_not(worker: tuple[Worker, dict[str, str]]) -> None:
    w, _ = worker
    down = FakeDiscord(lambda _: httpx.Response(503))
    report = run(w, END, down)
    n = report.queued
    assert report.delivery == {"sent": 0, "retry": n, "failed": 0}
    assert w.alerts.due(END) == []  # backing off

    up = FakeDiscord()
    report = run(w, END + timedelta(hours=1), up)
    assert report.delivery["sent"] == n
    assert w.alerts.status_counts() == {"sent": n}


def test_deleted_webhook_fails_permanently(worker: tuple[Worker, dict[str, str]]) -> None:
    w, _ = worker
    report = run(w, END, FakeDiscord(lambda _: httpx.Response(404, json={"message": "Unknown Webhook"})))
    n = report.queued
    assert report.delivery == {"sent": 0, "retry": 0, "failed": n}
    assert w.alerts.status_counts() == {"failed": n}


def test_retention_deletes_old_positions_and_keeps_flags(worker: tuple[Worker, dict[str, str]]) -> None:
    w, _ = worker
    run(w, END, FakeDiscord())
    flags = w.scoring.list_flags(ORG)
    report = w.run_once(START + timedelta(days=20), http=None)
    assert report.retention is not None
    assert report.retention["positions"] > 0
    assert Repository(w.db).count_positions() == 0
    assert len(w.scoring.list_flags(ORG)) == len(flags)


def test_channels_for() -> None:
    both = AlertDestinations.model_validate({"discord_webhook": WEBHOOK, "email": "a@b.org"})
    assert channels_for(both, email_enabled=True) == ["discord", "email"]
    assert channels_for(both, email_enabled=False) == ["discord"]
    assert channels_for(AlertDestinations(), email_enabled=True) == []
