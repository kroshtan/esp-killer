"""The agent's ``check`` and ``capture`` commands run their own event loop, so the fake server runs on a thread."""

import asyncio
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest
from typer.testing import CliRunner

from agent import __version__
from agent.main import app
from tools.fake_rcon import FakeRconOptions, FakeRconServer
from tools.sim import demo_world

runner = CliRunner()


@pytest.fixture
def rcon_port() -> Iterator[int]:
    loop = asyncio.new_event_loop()
    server = FakeRconServer(world=demo_world(n_honest=3), options=FakeRconOptions(password="secret"))
    port = loop.run_until_complete(server.start())
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    yield port
    asyncio.run_coroutine_threadsafe(server.stop(), loop).result(timeout=5)
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=5)
    loop.close()


def write_config(tmp_path: Path, port: int, password: str = "secret") -> Path:
    path = tmp_path / "agent.toml"
    path.write_text(
        f'[rcon]\nport = {port}\npassword = "{password}"\nidle_timeout_s = 0.05\n'
        # Nothing listens on port 1, so the backend check fails fast.
        '[backend]\nurl = "http://127.0.0.1:1"\napi_key = "k"\ntimeout_s = 1.0\n'
    )
    return path


def test_check_reports_rcon_ok_and_backend_failure(tmp_path: Path, rcon_port: int) -> None:
    result = runner.invoke(app, ["check", "--config", str(write_config(tmp_path, rcon_port))])
    assert "RCON ok: 4 player(s) parsed, 0 unparseable" in result.stdout
    assert "backend FAILED" in result.stderr
    assert result.exit_code == 1


def test_check_reports_wrong_password(tmp_path: Path, rcon_port: int) -> None:
    result = runner.invoke(app, ["check", "--config", str(write_config(tmp_path, rcon_port, "wrong"))])
    assert "RCON FAILED" in result.stderr
    assert result.exit_code == 1


def test_capture_writes_raw_responses_locally(tmp_path: Path, rcon_port: int) -> None:
    out = tmp_path / "captures"
    result = runner.invoke(
        app,
        [
            "capture",
            "--config",
            str(write_config(tmp_path, rcon_port)),
            "--count",
            "2",
            "--interval",
            "0.1",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "personal data" in result.stderr
    names = sorted(p.name.split("-", 1)[1] for p in out.iterdir())
    assert names == ["player_data-0.txt", "player_data-1.txt", "player_list.txt", "server_details.txt"]
    assert "PlayerData" in next(out.glob("*player_data-0.txt")).read_text()


def test_version_flag_prints_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == f"espk-agent {__version__}"
