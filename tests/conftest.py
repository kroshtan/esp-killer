from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI

from server.app import create_app
from server.db.repository import Repository
from server.ingest import AppState
from server.keys import generate_key, hash_key
from server.orgconfig import OrgConfig, OrgEntry, ServerEntry, save_config
from server.settings import ServerSettings
from tools.fake_rcon import FakeRconOptions, FakeRconServer
from tools.sim import demo_world

ORG = "test-org"
SERVER = "srv-1"


@dataclass
class Tenant:
    config_path: Path
    key: str
    org_id: str = ORG
    server_id: str = SERVER


@pytest.fixture
def tenant(tmp_path: Path) -> Tenant:
    """A config.yaml with one org and one server, and that server's API key."""
    key = generate_key()
    path = tmp_path / "config.yaml"
    save_config(path, OrgConfig(orgs={ORG: OrgEntry(servers={SERVER: ServerEntry(key_hash=hash_key(key))})}))
    return Tenant(config_path=path, key=key)


@pytest.fixture
def settings(tmp_path: Path, tenant: Tenant) -> ServerSettings:
    return ServerSettings(config_path=tenant.config_path, database_url=f"sqlite:///{tmp_path / 'espk.db'}")


@pytest.fixture
def app(settings: ServerSettings) -> FastAPI:
    return create_app(settings)


@pytest.fixture
def repo(app: FastAPI) -> Repository:
    state: AppState = app.state.espk
    return state.repo


@pytest.fixture
async def api(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:
        yield client


@pytest.fixture
async def fake_rcon() -> AsyncIterator[FakeRconServer]:
    server = FakeRconServer(world=demo_world(n_honest=5, seed=1), options=FakeRconOptions(password="secret"))
    await server.start()
    yield server
    await server.stop()


def rcon_port(server: FakeRconServer) -> int:
    """The port a started fake server is listening on."""
    assert server._server is not None
    return int(server._server.sockets[0].getsockname()[1])
