"""
Operator CLI: ``python -m server.cli --help``.

Key management edits config.yaml; the running API picks changes up automatically. A new key is printed once and
only its hash is stored, so a lost key cannot be recovered, only replaced with ``add-server --rotate``.
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer
from pydantic import TypeAdapter, ValidationError

from server.db.database import Database
from server.db.scoring import FLAG_STATUSES, OPEN, ScoringRepository
from server.keys import generate_key, hash_key
from server.orgconfig import AlertDestinations, OrgEntry, ServerEntry, Slug, load_config, save_config
from server.settings import ServerSettings

app = typer.Typer(help="Manage orgs, servers and API keys for the ESP detector backend.", no_args_is_help=True)

ConfigOption = Annotated[
    Path | None, typer.Option("--config", help="config.yaml path (default: $ESPK_CONFIG_PATH or ./config.yaml)")
]
DatabaseOption = Annotated[
    Path | None, typer.Option("--database", help="SQLite database path (default: $ESPK_DATABASE_PATH)")
]


def _config_path(config: Path | None) -> Path:
    return config if config is not None else ServerSettings().config_path


def _scoring_repo(database: Path | None) -> ScoringRepository:
    return ScoringRepository(Database(database or ServerSettings().database_path))


_SLUG = TypeAdapter(Slug)


def _check_slug(kind: str, value: str) -> None:
    try:
        _SLUG.validate_python(value)
    except ValidationError as e:
        raise _fail(f"invalid {kind} {value!r}: use lowercase letters, digits and dashes") from e


def _fail(message: str) -> typer.Exit:
    typer.echo(f"error: {message}", err=True)
    return typer.Exit(code=1)


@app.command("add-org")
def add_org(
    org_id: Annotated[str, typer.Argument(help="org slug, e.g. example-org")],
    discord_webhook: Annotated[str | None, typer.Option(help="Discord webhook URL for alerts")] = None,
    email: Annotated[str | None, typer.Option(help="email address for alerts")] = None,
    config: ConfigOption = None,
) -> None:
    """Add an organisation, or update its alert destinations if it already exists."""
    _check_slug("org id", org_id)
    path = _config_path(config)
    cfg = load_config(path)
    existing = cfg.orgs.get(org_id)
    # Options that are not given keep their current value.
    current = existing.alert if existing else AlertDestinations()
    try:
        alert = AlertDestinations.model_validate(
            {
                "discord_webhook": discord_webhook if discord_webhook is not None else current.discord_webhook,
                "email": email if email is not None else current.email,
            }
        )
    except ValidationError as e:
        raise _fail(str(e)) from e
    cfg.orgs[org_id] = OrgEntry(alert=alert, servers=existing.servers if existing else {})
    save_config(path, cfg)
    typer.echo(f"{'updated' if existing else 'added'} org {org_id}")


@app.command("add-server")
def add_server(
    org_id: Annotated[str, typer.Argument(help="org slug")],
    server_id: Annotated[str, typer.Argument(help="server slug, e.g. main-1")],
    rotate: Annotated[bool, typer.Option(help="replace the key of an existing server")] = False,
    config: ConfigOption = None,
) -> None:
    """Add a game server and print its new API key (shown only this once)."""
    _check_slug("server id", server_id)
    path = _config_path(config)
    cfg = load_config(path)
    org = cfg.orgs.get(org_id)
    if org is None:
        raise _fail(f"unknown org {org_id!r}; add it first with add-org")
    if server_id in org.servers and not rotate:
        raise _fail(f"server {org_id}/{server_id} exists; use --rotate to replace its key")
    key = generate_key()
    org.servers[server_id] = ServerEntry(key_hash=hash_key(key))
    save_config(path, cfg)
    typer.echo(f"API key for {org_id}/{server_id} (store it now, it is not shown again):", err=True)
    typer.echo(key)


@app.command("revoke-key")
def revoke_key(
    org_id: Annotated[str, typer.Argument(help="org slug")],
    server_id: Annotated[str, typer.Argument(help="server slug")],
    config: ConfigOption = None,
) -> None:
    """Revoke a server's API key. Its agent is rejected from the next request on."""
    path = _config_path(config)
    cfg = load_config(path)
    org = cfg.orgs.get(org_id)
    if org is None or server_id not in org.servers:
        raise _fail(f"unknown server {org_id}/{server_id}")
    org.servers[server_id] = ServerEntry(key_hash=None)
    save_config(path, cfg)
    typer.echo(f"revoked key for {org_id}/{server_id}")


@app.command("list-flags")
def list_flags(
    org: Annotated[str | None, typer.Option(help="only this org")] = None,
    status: Annotated[str, typer.Option(help="open, false_positive or all")] = OPEN,
    database: DatabaseOption = None,
) -> None:
    """List flagged players with their scores and the behaviours behind them."""
    if status != "all" and status not in FLAG_STATUSES:
        raise _fail(f"status must be one of {', '.join(FLAG_STATUSES)} or all")
    found = _scoring_repo(database).list_flags(org_id=org, status=None if status == "all" else status)
    if not found:
        typer.echo("no flags")
        return
    for f in found:
        behaviours = ", ".join(
            f"{name} {sub['score']:.2f}" for name, sub in f.details.items() if isinstance(sub, dict) and sub["score"]
        )
        servers = ",".join(f.details.get("servers", []))
        typer.echo(
            f"#{f.id:<5} {f.status:<14} {f.org_id}/{servers}  {f.player_name} ({f.player_id})  "
            f"score {f.score:.2f} (max {f.max_score:.2f})  [{behaviours}]  "
            f"flagged {f.created_at:%Y-%m-%d %H:%M}Z"
        )


@app.command("mark-false-positive")
def mark_false_positive(
    flag_id: Annotated[int, typer.Argument(help="flag id from list-flags")],
    note: Annotated[str | None, typer.Option(help="why it was a false positive")] = None,
    database: DatabaseOption = None,
) -> None:
    """Mark a flag as a false positive. The player is not flagged again for the configured suppression period."""
    flag = _scoring_repo(database).mark_false_positive(flag_id, datetime.now(UTC), note)
    if flag is None:
        raise _fail(f"no flag #{flag_id}")
    typer.echo(f"flag #{flag.id} ({flag.player_name}) marked as false positive")


if __name__ == "__main__":
    app()
