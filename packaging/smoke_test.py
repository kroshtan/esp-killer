"""
Smoke test for the built agent binary.

Run it from the repo root with the project venv::

    uv run python packaging/smoke_test.py dist/espk-agent

It starts the fake Evrima RCON server and the ingest API from source in a temporary directory, registers an
org/server/API key with the server CLI, runs the *binary* ``--version``, ``--help``, ``check`` and ``run`` (for
``--seconds``) against them, and asserts that snapshots and positions reached the database. It exits non-zero
on any failure and prints the output of every process involved.
"""

import argparse
import contextlib
import io
import os
import signal
import socket
import sqlite3
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Iterator
from functools import partial
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parent.parent
RCON_PASSWORD = "devpassword"


def _free_port() -> int:
    """
    Ask the OS for a free TCP port on localhost.

    :return: the port number
    """
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _port_open(port: int) -> bool:
    """
    Check whether something accepts TCP connections on a localhost port.

    :param port: port number
    :return: True once a connection succeeds (raises OSError otherwise)
    """
    with socket.create_connection(("127.0.0.1", port), timeout=1):
        return True


def _http_ok(url: str) -> bool:
    """
    Check whether a URL answers 200.

    :param url: URL to GET
    :return: True on HTTP 200
    """
    return httpx.get(url, timeout=1).status_code == 200


def _wait_until(what: str, ready: Callable[[], bool], timeout_s: float = 30.0) -> None:
    """
    Poll ``ready`` until it returns True.

    :param what: name used in the error message
    :param ready: zero-argument probe; OSError and httpx errors count as "not yet"
    :param timeout_s: give up after this long
    :raises RuntimeError: if ``ready`` never returned True
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        with contextlib.suppress(OSError, httpx.HTTPError):
            if ready():
                return
        time.sleep(0.2)
    raise RuntimeError(f"{what} did not come up within {timeout_s:.0f}s")


@contextlib.contextmanager
def _background(name: str, cmd: list[str], log_dir: Path, env: dict[str, str]) -> Iterator[subprocess.Popen[bytes]]:
    """
    Run a helper process for the duration of the block, then stop it and print its log.

    :param name: log file name
    :param cmd: command line
    :param log_dir: directory for the log file
    :param env: environment
    :yield: the running process
    """
    log_path = log_dir / f"{name}.log"
    with log_path.open("wb") as log:
        proc = subprocess.Popen(cmd, cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            yield proc
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
    print(f"----- {name} log (last 15 lines) -----")
    print("\n".join(log_path.read_text(errors="replace").splitlines()[-15:]))


def _run(cmd: list[str], env: dict[str, str], timeout_s: float = 60.0) -> subprocess.CompletedProcess[str]:
    """
    Run a command to completion, echo its output and fail on a non-zero exit code.

    :param cmd: command line
    :param env: environment
    :param timeout_s: timeout
    :return: the completed process
    :raises RuntimeError: if the command exited non-zero
    """
    print(f"$ {' '.join(cmd)}")
    # Decode leniently: on Windows a piped child may not write UTF-8, and rich draws box characters.
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        timeout=timeout_s,
        check=False,
        encoding="utf-8",
        errors="replace",
    )
    print(result.stdout + result.stderr, end="")
    if result.returncode != 0:
        raise RuntimeError(f"exit code {result.returncode}: {cmd}")
    return result


def smoke(binary: Path, work: Path, seconds: float) -> None:
    """
    Run the whole smoke test in ``work``.

    :param binary: path to the built agent executable
    :param work: empty scratch directory
    :param seconds: how long to let ``run`` poll and upload
    :raises RuntimeError: on any failed check
    """
    rcon_port, api_port = _free_port(), _free_port()
    db_path = work / "espk.db"
    env = {
        **os.environ,
        "ESPK_CONFIG_PATH": str(work / "config.yaml"),
        "ESPK_DATABASE_URL": f"sqlite:///{db_path.as_posix()}",
        "ESPK_DATABASE_PATH": str(db_path),  # for a server build that takes a plain path instead of a URL
        "PYTHONUNBUFFERED": "1",
    }
    python = sys.executable

    # Register an org and a server; add-server prints the API key (only) on stdout.
    _run([python, "-m", "server", "add-org", "smoke"], env)
    api_key = _run([python, "-m", "server", "add-server", "smoke", "gw-1"], env).stdout.strip()

    agent_toml = work / "agent.toml"
    agent_toml.write_text(
        "poll_interval_s = 1.0\n"
        "upload_interval_s = 1.0\n"
        f"queue_path = '{(work / 'queue.db').as_posix()}'\n"
        f"[rcon]\nhost = '127.0.0.1'\nport = {rcon_port}\npassword = '{RCON_PASSWORD}'\nidle_timeout_s = 0.1\n"
        f"[backend]\nurl = 'http://127.0.0.1:{api_port}'\napi_key = '{api_key}'\n",
        encoding="utf-8",
    )

    exe = str(binary.resolve())
    version = _run([exe, "--version"], env).stdout
    if not version.startswith("espk-agent "):
        raise RuntimeError(f"unexpected --version output: {version!r}")
    if "check" not in _run([exe, "--help"], env).stdout:
        raise RuntimeError("--help does not list the commands")

    rcon_cmd = [python, "-m", "tools.fake_rcon", "--port", str(rcon_port), "--password", RCON_PASSWORD]
    api_cmd = [python, "-m", "uvicorn", "server.app:create_app", "--factory", "--port", str(api_port)]
    with _background("fake_rcon", rcon_cmd, work, env), _background("api", api_cmd, work, env):
        _wait_until("fake RCON", partial(_port_open, rcon_port))
        _wait_until("API", partial(_http_ok, f"http://127.0.0.1:{api_port}/healthz"))

        check = _run([exe, "check", "--config", str(agent_toml)], env)
        if "RCON ok" not in check.stdout or "backend ok" not in check.stdout:
            raise RuntimeError("check did not report RCON ok and backend ok")

        print(f"$ {exe} run --config {agent_toml}  (for {seconds:.0f}s)")
        agent_log = work / "agent.log"
        with agent_log.open("wb") as log:
            proc = subprocess.Popen(
                [exe, "run", "--config", str(agent_toml)], cwd=work, env=env, stdout=log, stderr=subprocess.STDOUT
            )
            time.sleep(seconds)
            # SIGINT exercises the graceful shutdown path (final upload). Windows has no SIGINT for a child
            # without a shared console, so there it is simply terminated; uploads every 1s already happened.
            if os.name == "nt":
                proc.terminate()
            else:
                proc.send_signal(signal.SIGINT)
            code = proc.wait(timeout=30)
        log_text = agent_log.read_text(errors="replace")
        print(log_text, end="")
        if "Traceback" in log_text:
            raise RuntimeError("the agent logged a traceback")
        if os.name != "nt" and code != 0:
            raise RuntimeError(f"agent run exited with {code} after SIGINT")

    with contextlib.closing(sqlite3.connect(db_path)) as db:
        snapshots = db.execute("SELECT COUNT(*) FROM ingested_snapshots WHERE org_id = 'smoke'").fetchone()[0]
        positions = db.execute("SELECT COUNT(*) FROM positions WHERE org_id = 'smoke'").fetchone()[0]
        players = db.execute("SELECT COUNT(DISTINCT player_id) FROM positions WHERE org_id = 'smoke'").fetchone()[0]
    print(f"database: {snapshots} snapshot(s), {positions} position row(s), {players} distinct player(s)")
    if snapshots < 3 or positions == 0:
        raise RuntimeError("too little data arrived in the database")


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("binary", type=Path, help="the built agent, e.g. dist/espk-agent")
    parser.add_argument("--seconds", type=float, default=10.0, help="how long to run the agent")
    parser.add_argument("--keep", action="store_true", help="keep the scratch directory for inspection")
    args = parser.parse_args()
    # The Windows runner's piped stdout defaults to cp1252, which cannot print the agent's help boxes.
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if not args.binary.is_file():
        sys.exit(f"binary not found: {args.binary}")

    with tempfile.TemporaryDirectory(prefix="espk-smoke-", delete=not args.keep) as tmp:
        try:
            smoke(args.binary, Path(tmp), args.seconds)
        except (RuntimeError, subprocess.TimeoutExpired) as e:
            sys.exit(f"SMOKE TEST FAILED: {e}")
        if args.keep:
            print(f"scratch directory kept: {tmp}")
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
