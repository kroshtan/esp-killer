"""
The agent's command line: ``run``, ``check`` and ``capture``.

``capture`` exists to verify the RCON response format against a real server (see NOTES.md). Its output contains
player names, ids and positions, i.e. personal data: it is written to a local file only and never uploaded.
"""

import asyncio
import contextlib
import logging
import signal
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import httpx
import typer

from agent import __version__
from agent.config import AgentSettings, load_settings
from agent.poller import Poller
from agent.queue import SnapshotQueue
from agent.rcon import EvrimaRconClient, RconError
from agent.uploader import Uploader
from shared.playerdata import parse_player_data
from shared.rcon_protocol import ReadOnlyCommand

app = typer.Typer(help="ESP detector agent: polls Evrima RCON (read-only) and uploads player positions.")
logger = logging.getLogger("agent")

ConfigOption = Annotated[Path | None, typer.Option("--config", "-c", help="agent TOML config file")]


def make_client(settings: AgentSettings) -> EvrimaRconClient:
    """
    Build the RCON client from settings.

    :param settings: agent settings
    :return: an unconnected client
    """
    rcon = settings.rcon
    return EvrimaRconClient(
        rcon.host,
        rcon.port,
        rcon.password.get_secret_value(),
        connect_timeout_s=rcon.connect_timeout_s,
        response_timeout_s=rcon.response_timeout_s,
        idle_timeout_s=rcon.idle_timeout_s,
        max_response_bytes=rcon.max_response_bytes,
    )


async def run_agent(settings: AgentSettings, stop: asyncio.Event, http: httpx.AsyncClient | None = None) -> None:
    """
    Poll and upload until ``stop`` is set.

    :param settings: agent settings
    :param stop: set to shut down (the uploader makes one last attempt first)
    :param http: HTTP client to use; tests inject one wired to the app
    """
    queue = SnapshotQueue(settings.queue_path, settings.queue_max_bytes)
    owns_http = http is None
    if http is None:
        http = httpx.AsyncClient(
            timeout=settings.backend.timeout_s, headers={"User-Agent": f"espk-agent/{__version__}"}
        )
    poller = Poller(make_client(settings), queue, settings.poll_interval_s)
    uploader = Uploader(
        queue,
        http,
        settings.backend.ingest_url,
        settings.backend.api_key.get_secret_value(),
        max_batch=settings.max_snapshots_per_batch,
        interval_s=settings.upload_interval_s,
    )
    try:
        await asyncio.gather(poller.run(stop), uploader.run(stop))
    finally:
        if owns_http:
            await http.aclose()
        queue.close()


def _setup_logging(level: str) -> None:
    logging.basicConfig(level=level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # httpx logs every request URL at INFO; keep the agent's own log readable.
    logging.getLogger("httpx").setLevel(logging.WARNING)


@app.command()
def run(config: ConfigOption = None) -> None:
    """Poll the game server and upload positions until stopped (Ctrl+C)."""
    settings = load_settings(config)
    _setup_logging(settings.log_level)
    logger.info("espk agent %s starting (poll every %.1fs)", __version__, settings.poll_interval_s)

    async def main() -> None:
        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            # Windows has no add_signal_handler; Ctrl+C then raises KeyboardInterrupt instead.
            with contextlib.suppress(NotImplementedError):
                loop.add_signal_handler(sig, stop.set)
        await run_agent(settings, stop)

    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(main())
    logger.info("espk agent stopped")


@app.command()
def check(config: ConfigOption = None) -> None:
    """Check the RCON login and that the backend is reachable, then exit."""
    settings = load_settings(config)
    _setup_logging(settings.log_level)

    async def main() -> bool:
        ok = True
        try:
            async with make_client(settings) as client:
                result = parse_player_data(await client.request(ReadOnlyCommand.PLAYER_DATA))
            typer.echo(f"RCON ok: {len(result.players)} player(s) parsed, {len(result.errors)} unparseable line(s)")
            ok = not result.errors
        except RconError as e:
            typer.echo(f"RCON FAILED: {e}", err=True)
            ok = False
        health_url = str(settings.backend.url).rstrip("/") + "/healthz"
        try:
            async with httpx.AsyncClient(timeout=settings.backend.timeout_s) as http:
                response = await http.get(health_url)
            response.raise_for_status()
            typer.echo(f"backend ok: {health_url}")
        except httpx.HTTPError as e:
            typer.echo(f"backend FAILED: {health_url}: {e!r}", err=True)
            ok = False
        return ok

    if not asyncio.run(main()):
        raise typer.Exit(code=1)


@app.command()
def capture(
    config: ConfigOption = None,
    count: Annotated[int, typer.Option(help="number of responses to capture")] = 3,
    interval: Annotated[float, typer.Option(help="seconds between captures")] = 3.0,
    out: Annotated[Path, typer.Option(help="output directory")] = Path("captures"),
) -> None:
    """Save raw RCON responses to local files, to check the parser against a real server."""
    settings = load_settings(config)
    _setup_logging(settings.log_level)
    typer.echo(
        "WARNING: captures contain player names, ids and positions (personal data). They stay on this machine; "
        "delete them once you have checked the format.",
        err=True,
    )
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

    async def main() -> None:
        async with make_client(settings) as client:
            for command in (ReadOnlyCommand.SERVER_DETAILS, ReadOnlyCommand.PLAYER_LIST):
                path = out / f"{stamp}-{command.name.lower()}.txt"
                path.write_text(await client.request(command), encoding="utf-8")
            for i in range(count):
                if i:
                    await asyncio.sleep(interval)
                text = await client.request(ReadOnlyCommand.PLAYER_DATA)
                path = out / f"{stamp}-player_data-{i}.txt"
                path.write_text(text, encoding="utf-8")
                result = parse_player_data(text)
                typer.echo(f"{path}: {len(result.players)} parsed, {len(result.errors)} unparseable")
                for error in result.errors:
                    typer.echo(f"  line {error.line_no}: {error.reason} (keys: {error.shape})")

    asyncio.run(main())
