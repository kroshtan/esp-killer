import gzip
import json
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from server.db.database import parse_ts
from server.db.repository import Repository
from server.keys import generate_key
from server.orgconfig import ServerEntry, load_config, save_config
from tests.conftest import Tenant
from tests.helpers import auth, batch_body, sample, snapshot


async def test_healthz(api: httpx.AsyncClient) -> None:
    response = await api.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


async def test_ingest_stores_rows(api: httpx.AsyncClient, tenant: Tenant, repo: Repository) -> None:
    captured = datetime.now(UTC).replace(microsecond=0)
    snap = snapshot(
        [
            sample("76561198000000001", x=1.5, y=-2.5, dino_class="Carnotaurus", growth=0.75),
            sample("0002a1b2c3d4e5f60718293a4b5c6d7e", x=3.0, y=4.0),
        ],
        captured_at=captured,
    )
    response = await api.post("/v1/ingest", content=batch_body(snap), headers=auth(tenant.key))

    assert response.status_code == 200, response.text
    body = response.json()
    assert (body["accepted_snapshots"], body["duplicate_snapshots"], body["rows"]) == (1, 0, 2)
    with repo.db.connect() as conn:
        rows = conn.execute("SELECT * FROM positions ORDER BY player_id").fetchall()
    first = rows[0]
    assert (first["org_id"], first["server_id"], first["player_id"]) == (
        "test-org",
        "srv-1",
        "0002a1b2c3d4e5f60718293a4b5c6d7e",
    )
    second = rows[1]
    assert (second["dino_class"], second["growth"], second["x"], second["y"]) == ("Carnotaurus", 0.75, 1.5, -2.5)
    assert parse_ts(second["server_ts"]) == captured
    assert parse_ts(second["received_ts"]) >= captured


async def test_uncompressed_body_is_accepted(api: httpx.AsyncClient, tenant: Tenant) -> None:
    response = await api.post(
        "/v1/ingest", content=batch_body(snapshot(), compress=False), headers=auth(tenant.key, gzip_body=False)
    )
    assert response.status_code == 200


async def test_empty_snapshot_is_accepted(api: httpx.AsyncClient, tenant: Tenant) -> None:
    response = await api.post("/v1/ingest", content=batch_body(snapshot([])), headers=auth(tenant.key))
    assert response.json()["accepted_snapshots"] == 1
    assert response.json()["rows"] == 0


async def test_retried_batch_is_not_stored_twice(api: httpx.AsyncClient, tenant: Tenant, repo: Repository) -> None:
    a, b = snapshot(), snapshot()
    await api.post("/v1/ingest", content=batch_body(a), headers=auth(tenant.key))
    # A retry after a lost response, regrouped with a new snapshot, and with a snapshot repeated in the batch.
    response = await api.post("/v1/ingest", content=batch_body(a, b, b), headers=auth(tenant.key))

    body = response.json()
    assert (body["accepted_snapshots"], body["duplicate_snapshots"]) == (1, 2)
    assert repo.count_positions() == 2


@pytest.mark.parametrize(
    "header",
    [None, "Bearer", "Bearer nonsense", "Basic dXNlcjpwYXNz", f"Bearer {generate_key()}"],
)
async def test_bad_credentials(api: httpx.AsyncClient, header: str | None) -> None:
    headers = {"Authorization": header} if header else {}
    response = await api.post("/v1/ingest", content=batch_body(snapshot()), headers=headers)
    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"


async def test_revoked_key(api: httpx.AsyncClient, tenant: Tenant) -> None:
    cfg = load_config(tenant.config_path)
    cfg.orgs["test-org"].servers["srv-1"] = ServerEntry(key_hash=None)
    save_config(tenant.config_path, cfg)
    response = await api.post("/v1/ingest", content=batch_body(snapshot()), headers=auth(tenant.key))
    assert response.status_code == 401


async def test_validation_errors_do_not_echo_player_data(api: httpx.AsyncClient, tenant: Tenant) -> None:
    body = (
        b'{"agent_version": "t", "snapshots": [{"snapshot_id": "00000000-0000-0000-0000-000000000001", '
        b'"captured_at": "2026-01-01T00:00:00Z", "players": [{"player_id": "SecretPlayer!", '
        b'"player_name": "Secret Name", "x": 1, "y": 2, "z": 3}]}]}'
    )
    response = await api.post("/v1/ingest", content=body, headers=auth(tenant.key, gzip_body=False))
    assert response.status_code == 422
    assert "Secret" not in response.text


@pytest.mark.parametrize("body", [b"not json", b"{}", b'{"agent_version": "t", "snapshots": []}'])
async def test_malformed_payloads(api: httpx.AsyncClient, tenant: Tenant, body: bytes) -> None:
    response = await api.post("/v1/ingest", content=body, headers=auth(tenant.key, gzip_body=False))
    assert response.status_code == 422


async def test_body_size_limit(api: httpx.AsyncClient, tenant: Tenant) -> None:
    body = b"x" * (2 * 1024 * 1024 + 1)
    response = await api.post("/v1/ingest", content=body, headers=auth(tenant.key, gzip_body=False))
    assert response.status_code == 413


async def test_gzip_bomb_is_rejected(api: httpx.AsyncClient, tenant: Tenant) -> None:
    bomb = gzip.compress(b" " * (64 * 1024 * 1024))
    assert len(bomb) < 2 * 1024 * 1024
    response = await api.post("/v1/ingest", content=bomb, headers=auth(tenant.key))
    assert response.status_code == 413


@pytest.mark.parametrize(("encoding", "status"), [("br", 415), ("gzip", 400)])
async def test_bad_encodings(api: httpx.AsyncClient, tenant: Tenant, encoding: str, status: int) -> None:
    headers = {**auth(tenant.key, gzip_body=False), "Content-Encoding": encoding}
    response = await api.post("/v1/ingest", content=b"not compressed", headers=headers)
    assert response.status_code == status


async def test_truncated_gzip(api: httpx.AsyncClient, tenant: Tenant) -> None:
    body = batch_body(snapshot())
    response = await api.post("/v1/ingest", content=body[: len(body) // 2], headers=auth(tenant.key))
    assert response.status_code == 400


async def test_timestamps_outside_the_window_are_rejected(
    api: httpx.AsyncClient, tenant: Tenant, repo: Repository
) -> None:
    now = datetime.now(UTC)
    future = snapshot(captured_at=now + timedelta(hours=1))
    ancient = snapshot(captured_at=now - timedelta(days=30))
    ok = snapshot(captured_at=now - timedelta(days=1))
    response = await api.post("/v1/ingest", content=batch_body(future, ancient, ok), headers=auth(tenant.key))
    body = response.json()
    assert (body["accepted_snapshots"], body["rejected_snapshots"]) == (1, 2)
    assert repo.count_positions() == 1


async def test_rate_limit(api: httpx.AsyncClient, tenant: Tenant) -> None:
    responses = [
        await api.post("/v1/ingest", content=batch_body(snapshot()), headers=auth(tenant.key)) for _ in range(25)
    ]
    statuses = [r.status_code for r in responses]
    assert statuses[:20] == [200] * 20
    limited = [r for r in responses[20:] if r.status_code == 429]
    assert limited
    assert int(limited[0].headers["Retry-After"]) >= 1


async def test_parse_health_is_recorded_and_new_problems_are_logged(
    api: httpx.AsyncClient, tenant: Tenant, repo: Repository, caplog: pytest.LogCaptureFixture
) -> None:
    clean = {"polls": 5, "lines": 10, "players": 10}
    broken = {"polls": 5, "lines": 10, "players": 8, "unparsed": 2, "errors": {"no Location field": 2}}
    for health in (clean, broken, broken):
        body = batch_body(snapshot(), compress=False)
        payload = json.loads(body)
        payload["parse_health"] = health
        response = await api.post("/v1/ingest", json=payload, headers=auth(tenant.key, gzip_body=False))
        assert response.status_code == 200
    (status,) = repo.agent_statuses()
    assert status.agent_version == "test"
    assert status.latest_health is not None and status.latest_health.unparsed == 2
    assert status.total_health is not None and status.total_health.polls == 15
    assert status.total_health.errors == {"no Location field": 4}
    # Logged once, when the problems first appeared.
    assert caplog.text.count("could not parse") == 1
