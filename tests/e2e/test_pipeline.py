"""
Fake RCON server -> real agent (poller + disk queue + uploader) -> FastAPI app -> rows in SQLite.

The agent's HTTP client is wired straight to the app with an ASGI transport; everything else is real, including
the TCP connection to the fake game server.
"""

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI

from agent.config import AgentSettings, BackendSettings, RconSettings
from agent.main import run_agent
from server.db.repository import Repository
from shared.rcon_protocol import ReadOnlyCommand
from tests.conftest import Tenant, rcon_port
from tools.fake_rcon import FakeRconServer


class FlakyTransport(httpx.AsyncBaseTransport):
    """ASGI transport to the app that can simulate the backend being down."""

    def __init__(self, app: FastAPI) -> None:
        self.inner = httpx.ASGITransport(app=app)
        self.down = False
        self.failed_requests = 0

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if self.down:
            self.failed_requests += 1
            raise httpx.ConnectError("backend down", request=request)
        return await self.inner.handle_async_request(request)


@pytest.fixture
def agent_settings(tmp_path: Path, tenant: Tenant, fake_rcon: FakeRconServer) -> AgentSettings:
    return AgentSettings(
        rcon=RconSettings(port=rcon_port(fake_rcon), password="secret", idle_timeout_s=0.05),
        backend=BackendSettings(url="http://testserver", api_key=tenant.key, allow_insecure_http=True),
        poll_interval_s=0.5,
        upload_interval_s=1.0,
        queue_path=tmp_path / "agent-queue.db",
    )


@pytest.fixture
async def transport(app: FastAPI) -> AsyncIterator[FlakyTransport]:
    t = FlakyTransport(app)
    yield t
    await t.inner.aclose()


async def run_for(settings: AgentSettings, transport: FlakyTransport, seconds: float) -> None:
    stop = asyncio.Event()
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as http:
        task = asyncio.create_task(run_agent(settings, stop, http))
        await asyncio.sleep(seconds)
        stop.set()
        await asyncio.wait_for(task, timeout=10)


def stored(repo: Repository) -> tuple[int, int, set[str]]:
    with repo.db.connect() as conn:
        n_snapshots = int(conn.execute("SELECT COUNT(*) FROM ingested_snapshots").fetchone()[0])
        n_rows = int(conn.execute("SELECT COUNT(*) FROM positions").fetchone()[0])
        players = {row[0] for row in conn.execute("SELECT DISTINCT player_id FROM positions")}
    return n_snapshots, n_rows, players


async def test_positions_flow_from_game_server_to_database(
    agent_settings: AgentSettings, transport: FlakyTransport, fake_rcon: FakeRconServer, repo: Repository
) -> None:
    fake_rcon.options.speedup = 20.0
    await run_for(agent_settings, transport, seconds=3.0)

    n_snapshots, n_rows, players = stored(repo)
    expected_players = {p.player_id for p in fake_rcon.world.players}
    assert n_snapshots >= 4
    assert n_rows == n_snapshots * len(expected_players)
    assert players == expected_players  # including the scripted cheater
    with repo.db.connect() as conn:
        row = conn.execute("SELECT * FROM positions LIMIT 1").fetchone()
    assert (row["org_id"], row["server_id"]) == ("test-org", "srv-1")
    assert row["dino_class"] is not None
    assert row["growth"] is not None
    # The agent only ever asked for player data.
    assert set(fake_rcon.received_opcodes) == {ReadOnlyCommand.PLAYER_DATA}
    # ...and reported that it parsed cleanly.
    (status,) = repo.agent_statuses()
    assert status.total_health is not None
    assert status.total_health.polls >= n_snapshots - 1
    assert status.total_health.problems == 0
    assert status.total_health.ended == status.total_health.polls


async def test_backend_outage_is_buffered_and_drained_without_duplicates(
    agent_settings: AgentSettings, transport: FlakyTransport, fake_rcon: FakeRconServer, repo: Repository
) -> None:
    transport.down = True
    stop = asyncio.Event()
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as http:
        task = asyncio.create_task(run_agent(agent_settings, stop, http))
        await asyncio.sleep(2.5)
        assert transport.failed_requests >= 1
        assert stored(repo)[0] == 0
        transport.down = False
        # The uploader's first backoff is at most 2 s.
        await asyncio.sleep(3.5)
        stop.set()
        await asyncio.wait_for(task, timeout=10)

    n_snapshots, n_rows, _ = stored(repo)
    assert n_snapshots >= 8  # polled throughout, including during the outage
    assert n_rows == n_snapshots * len(fake_rcon.world.players)


async def test_agent_survives_a_game_server_restart(
    agent_settings: AgentSettings, transport: FlakyTransport, fake_rcon: FakeRconServer, repo: Repository
) -> None:
    stop = asyncio.Event()
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as http:
        task = asyncio.create_task(run_agent(agent_settings, stop, http))
        await asyncio.sleep(1.2)
        await fake_rcon.drop_connections()
        await asyncio.sleep(3.0)
        stop.set()
        await asyncio.wait_for(task, timeout=10)

    assert fake_rcon.auth_attempts >= 2
    assert stored(repo)[0] >= 4
