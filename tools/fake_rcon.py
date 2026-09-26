"""
A fake Evrima RCON server for local development, demos and end-to-end tests.

It speaks the same wire format as the real server (see shared/rcon_protocol.py) and answers the read-only
commands from a simulated world that advances with wall-clock time (times ``speedup``). It records every opcode it
receives, so tests can assert the agent never sends an admin command.

Run it with ``python -m tools.fake_rcon --port 8888 --password devpassword``.
"""

import argparse
import asyncio
import contextlib
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime

from shared.rcon_protocol import AUTH, AUTH_ACCEPTED, EXEC, TERMINATOR, ReadOnlyCommand
from tools.rcon_format import format_player_data, format_player_list
from tools.sim import World, demo_world

logger = logging.getLogger(__name__)

AUTH_REJECTED = "Password Incorrect"
SIM_STEP_S = 1.0


@dataclass
class FakeRconOptions:
    password: str = "devpassword"
    speedup: float = 1.0
    # Split each response into writes of this many bytes, ``chunk_delay_s`` apart, to exercise the client's
    # read-until-idle logic. 0 sends each response in one write.
    chunk_size: int = 0
    chunk_delay_s: float = 0.0
    # Whether responses end with a NUL byte. Unknown for the real server, so the client must handle both.
    nul_terminate: bool = False


@dataclass
class FakeRconServer:
    world: World
    options: FakeRconOptions = field(default_factory=FakeRconOptions)
    received_opcodes: list[int] = field(default_factory=list)
    auth_attempts: int = 0
    _server: asyncio.Server | None = None
    _last_advance: float = field(default_factory=time.monotonic)
    _connections: set[asyncio.StreamWriter] = field(default_factory=set)

    async def start(self, host: str = "127.0.0.1", port: int = 0) -> int:
        """
        Start listening.

        :param host: interface to bind
        :param port: port to bind; 0 picks a free one
        :return: the bound port
        """
        self._server = await asyncio.start_server(self._handle, host, port)
        self._last_advance = time.monotonic()
        return int(self._server.sockets[0].getsockname()[1])

    async def stop(self) -> None:
        """Stop listening and drop every open connection."""
        if self._server is None:
            return
        self._server.close()
        await self.drop_connections()
        await self._server.wait_closed()
        self._server = None

    async def drop_connections(self) -> None:
        """Close every client connection, as a server restart would."""
        for writer in list(self._connections):
            writer.close()
            with contextlib.suppress(ConnectionError):
                await writer.wait_closed()
        self._connections.clear()

    def advance(self) -> None:
        """Move the simulation forward to the current wall-clock time."""
        now = time.monotonic()
        elapsed = (now - self._last_advance) * self.options.speedup
        steps = int(elapsed // SIM_STEP_S)
        for _ in range(steps):
            self.world.step(SIM_STEP_S)
        self._last_advance += steps * SIM_STEP_S / self.options.speedup

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self._connections.add(writer)
        authed = False
        try:
            while True:
                try:
                    message = await reader.readuntil(TERMINATOR)
                except (asyncio.IncompleteReadError, ConnectionError):
                    return
                message = message[:-1]
                if not message:
                    continue
                kind = message[0]
                if kind == AUTH:
                    self.auth_attempts += 1
                    authed = message[1:].decode("utf-8", errors="replace") == self.options.password
                    await self._send(writer, AUTH_ACCEPTED if authed else AUTH_REJECTED)
                    if not authed:
                        return
                elif kind == EXEC and authed and len(message) >= 2:
                    opcode = message[1]
                    self.received_opcodes.append(opcode)
                    await self._send(writer, self._respond(opcode))
                else:
                    return
        finally:
            self._connections.discard(writer)
            writer.close()

    def _respond(self, opcode: int) -> str:
        now = datetime.now()  # noqa: DTZ005 - the real server reports naive local time
        if opcode == ReadOnlyCommand.PLAYER_DATA:
            self.advance()
            return format_player_data(self.world.snapshot(), now)
        if opcode == ReadOnlyCommand.PLAYER_LIST:
            self.advance()
            return format_player_list(self.world.snapshot(), now)
        if opcode == ReadOnlyCommand.SERVER_DETAILS:
            return f"[{now:%Y.%m.%d-%H.%M.%S}] ServerDetails\nServerName: Fake Evrima Server, ServerMap: Gateway"
        logger.warning("fake RCON received non-read-only opcode 0x%02x", opcode)
        return "Unsupported command"

    async def _send(self, writer: asyncio.StreamWriter, text: str) -> None:
        data = text.encode("utf-8") + (TERMINATOR if self.options.nul_terminate else b"")
        size = self.options.chunk_size or len(data)
        for i in range(0, len(data), size):
            writer.write(data[i : i + size])
            await writer.drain()
            if self.options.chunk_delay_s and i + size < len(data):
                await asyncio.sleep(self.options.chunk_delay_s)


async def _serve(args: argparse.Namespace) -> None:
    server = FakeRconServer(
        world=demo_world(n_honest=args.players, cheater=not args.no_cheater, seed=args.seed),
        options=FakeRconOptions(password=args.password, speedup=args.speedup),
    )
    port = await server.start(args.host, args.port)
    logger.info("fake Evrima RCON listening on %s:%d", args.host, port)
    try:
        await asyncio.Event().wait()
    finally:
        await server.stop()


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8888)
    parser.add_argument("--password", default="devpassword")
    parser.add_argument("--players", type=int, default=15, help="number of honest players")
    parser.add_argument("--no-cheater", action="store_true", help="leave out the scripted beeline cheater")
    parser.add_argument("--speedup", type=float, default=1.0, help="simulated seconds per wall-clock second")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(_serve(args))


if __name__ == "__main__":
    main()
