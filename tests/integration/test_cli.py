from datetime import UTC, datetime
from pathlib import Path

from typer.testing import CliRunner

from server.cli import app
from server.db.database import Database
from server.db.repository import Repository
from server.keys import hash_key, looks_like_key
from server.orgconfig import load_config
from shared.models import ParseHealth

runner = CliRunner()


def cli(config: Path, *args: str) -> tuple[int, str]:
    result = runner.invoke(app, [*args, "--config", str(config)])
    return result.exit_code, result.stdout


def test_org_and_server_lifecycle(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    assert cli(config, "add-org", "acme", "--email", "admins@acme.test")[0] == 0

    code, out = cli(config, "add-server", "acme", "main-1")
    assert code == 0
    key = out.strip().splitlines()[-1]
    assert looks_like_key(key)
    stored = load_config(config)
    assert stored.orgs["acme"].servers["main-1"].key_hash == hash_key(key)
    assert key not in config.read_text()

    assert cli(config, "add-server", "acme", "main-1")[0] == 1  # exists
    code, out = cli(config, "add-server", "acme", "main-1", "--rotate")
    assert code == 0
    assert out.strip().splitlines()[-1] != key

    assert cli(config, "revoke-key", "acme", "main-1")[0] == 0
    assert load_config(config).orgs["acme"].servers["main-1"].key_hash is None


def test_add_org_keeps_alert_settings_not_given(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    cli(config, "add-org", "acme", "--email", "admins@acme.test")
    cli(config, "add-server", "acme", "s1")
    cli(config, "add-org", "acme", "--discord-webhook", "https://discord.com/api/webhooks/1/x")
    org = load_config(config).orgs["acme"]
    assert org.alert.email == "admins@acme.test"
    assert str(org.alert.discord_webhook) == "https://discord.com/api/webhooks/1/x"
    assert "s1" in org.servers


def test_errors(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    assert cli(config, "add-org", "Bad Org")[0] == 1
    assert cli(config, "add-org", "acme", "--email", "not-an-email")[0] == 1
    assert cli(config, "add-server", "nope", "s1")[0] == 1
    assert cli(config, "revoke-key", "nope", "s1")[0] == 1


def test_servers_lists_key_status_and_uploads(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    db = tmp_path / "espk.db"
    cli(config, "add-org", "acme")
    cli(config, "add-server", "acme", "s1")
    cli(config, "add-server", "acme", "s2")
    health = ParseHealth(
        polls=4, lines=8, players=7, unparsed=1, errors={"no Location field": 1}, unknown_keys=["Diet"]
    )
    Repository(Database(db)).record_agent_status("acme", "s1", "0.1.1", health, datetime.now(UTC))

    result = runner.invoke(app, ["servers", "--config", str(config), "--database", str(db)])
    assert result.exit_code == 0
    assert "acme/s1  key active  agent 0.1.1" in result.stdout
    assert "1.8 players/poll" in result.stdout
    assert "unknown fields: Diet" in result.stdout
    assert "acme/s2  key active  never uploaded" in result.stdout
