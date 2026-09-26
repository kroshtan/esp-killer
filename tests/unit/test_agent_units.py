import random
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from agent.backoff import Backoff
from agent.config import load_settings
from agent.poller import build_snapshot
from agent.queue import SnapshotQueue
from shared.playerdata import ParsedPlayer, ParseResult
from tests.helpers import sample, snapshot

REPO = Path(__file__).resolve().parents[2]

AGENT_TOML = """
poll_interval_s = 5.0
[rcon]
password = "from-file"
port = 9999
[backend]
url = "https://espk.example.org"
api_key = "espk_file"
"""


@pytest.fixture
def agent_toml(tmp_path: Path) -> Path:
    path = tmp_path / "agent.toml"
    path.write_text(AGENT_TOML)
    return path


class TestConfig:
    def test_file(self, agent_toml: Path) -> None:
        s = load_settings(agent_toml)
        assert s.poll_interval_s == 5.0
        assert s.rcon.port == 9999
        assert s.rcon.password.get_secret_value() == "from-file"
        assert s.backend.ingest_url == "https://espk.example.org/v1/ingest"

    def test_env_overrides_file(self, agent_toml: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ESPK_RCON__PASSWORD", "from-env")
        monkeypatch.setenv("ESPK_POLL_INTERVAL_S", "7")
        s = load_settings(agent_toml)
        assert s.rcon.password.get_secret_value() == "from-env"
        assert s.rcon.port == 9999
        assert s.poll_interval_s == 7.0

    def test_env_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ESPK_RCON__PASSWORD", "p")
        monkeypatch.setenv("ESPK_BACKEND__API_KEY", "k")
        monkeypatch.setenv("ESPK_BACKEND__URL", "https://b.example.org/")
        assert load_settings(None).backend.ingest_url == "https://b.example.org/v1/ingest"

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_settings(tmp_path / "nope.toml")

    def test_secrets_are_not_printed(self, agent_toml: Path) -> None:
        assert "from-file" not in repr(load_settings(agent_toml))

    @pytest.mark.parametrize(
        ("url", "insecure", "ok"),
        [
            ("http://espk.example.org", False, False),
            ("http://espk.example.org", True, True),
            ("http://localhost:8000", False, True),
            ("http://127.0.0.1:8000", False, True),
        ],
    )
    def test_https_required_for_remote_backends(
        self, monkeypatch: pytest.MonkeyPatch, url: str, insecure: bool, ok: bool
    ) -> None:
        monkeypatch.setenv("ESPK_RCON__PASSWORD", "p")
        monkeypatch.setenv("ESPK_BACKEND__API_KEY", "k")
        monkeypatch.setenv("ESPK_BACKEND__URL", url)
        monkeypatch.setenv("ESPK_BACKEND__ALLOW_INSECURE_HTTP", str(insecure).lower())
        if ok:
            load_settings(None)
        else:
            with pytest.raises(ValidationError, match="https"):
                load_settings(None)


class TestQueue:
    def test_fifo_peek_and_ack(self, tmp_path: Path) -> None:
        q = SnapshotQueue(tmp_path / "q.db", max_bytes=10**7)
        snaps = [snapshot() for _ in range(5)]
        for s in snaps:
            q.put(s)
        items = q.peek(3)
        assert [s.snapshot_id for _, s in items] == [s.snapshot_id for s in snaps[:3]]
        q.ack([i for i, _ in items])
        assert [s.snapshot_id for _, s in q.peek(10)] == [s.snapshot_id for s in snaps[3:]]
        assert len(q) == 2

    def test_survives_restart(self, tmp_path: Path) -> None:
        q = SnapshotQueue(tmp_path / "q.db", max_bytes=10**7)
        s = snapshot()
        q.put(s)
        size = q.size_bytes
        q.close()
        q2 = SnapshotQueue(tmp_path / "q.db", max_bytes=10**7)
        assert q2.size_bytes == size
        assert q2.peek(1)[0][1] == s

    def test_drops_oldest_past_cap(self, tmp_path: Path) -> None:
        one = len(snapshot().model_dump_json())
        q = SnapshotQueue(tmp_path / "q.db", max_bytes=int(one * 3.5))
        snaps = [snapshot() for _ in range(6)]
        for s in snaps:
            q.put(s)
        assert len(q) == 3
        assert q.dropped == 3
        assert [s.snapshot_id for _, s in q.peek(10)] == [s.snapshot_id for s in snaps[3:]]
        assert q.size_bytes <= q.max_bytes

    def test_ack_nothing(self, tmp_path: Path) -> None:
        SnapshotQueue(tmp_path / "q.db", max_bytes=10**7).ack([])


def test_backoff_grows_to_cap_and_resets() -> None:
    b = Backoff(base_s=1.0, cap_s=8.0, rng=random.Random(0))
    delays = [b.next_delay() for _ in range(6)]
    ceilings = [1, 2, 4, 8, 8, 8]
    assert all(c / 2 <= d <= c for d, c in zip(delays, ceilings, strict=True))
    b.reset()
    assert b.next_delay() <= 1.0


def test_build_snapshot_skips_invalid_players() -> None:
    result = ParseResult(
        players=[
            ParsedPlayer(player_id="1", player_name="ok", x=1, y=2, z=3),
            ParsedPlayer(player_id="2", player_name="far", x=1e12, y=0, z=0),
        ]
    )
    snap, invalid = build_snapshot(result, datetime.now(UTC))
    assert [p.player_id for p in snap.players] == ["1"]
    assert invalid == 1


def test_snapshot_rejects_duplicate_players() -> None:
    with pytest.raises(ValidationError, match="duplicate"):
        snapshot([sample("1"), sample("1")])


def test_agent_does_not_import_server_side_code() -> None:
    """The agent binary must stay small and must not contain the server or analysis stack."""
    code = (
        "import sys, agent.main\n"
        "bad = sorted(m for m in sys.modules if m.split('.')[0] in "
        "{'server', 'tools', 'numpy', 'pandas', 'matplotlib', 'fastapi'})\n"
        "print(','.join(bad))"
    )
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == ""
