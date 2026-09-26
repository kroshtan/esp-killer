"""
Operator CLI: ``python -m server.cli --help``.

Key management edits config.yaml; the running API picks changes up automatically. A new key is printed once and
only its hash is stored, so a lost key cannot be recovered, only replaced with ``add-server --rotate``.
"""

from pathlib import Path
from typing import Annotated

import typer
from pydantic import TypeAdapter, ValidationError

from server.keys import generate_key, hash_key
from server.orgconfig import AlertDestinations, OrgEntry, ServerEntry, Slug, load_config, save_config
from server.settings import ServerSettings

app = typer.Typer(help="Manage orgs, servers and API keys for the ESP detector backend.", no_args_is_help=True)

ConfigOption = Annotated[
    Path | None, typer.Option("--config", help="config.yaml path (default: $ESPK_CONFIG_PATH or ./config.yaml)")
]


def _config_path(config: Path | None) -> Path:
    return config if config is not None else ServerSettings().config_path


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


if __name__ == "__main__":
    app()
