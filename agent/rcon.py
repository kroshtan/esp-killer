"""
Async client for the Evrima RCON protocol.

The protocol has no framing, so the hard part is knowing when a response is complete. The reference clients do a
single ``read()`` and treat whatever arrived as the whole response, which truncates large player lists and leaves
the rest on the socket to be mistaken for the next response. This client instead:

1. waits up to ``response_timeout_s`` for the first bytes;
2. keeps reading until the socket is quiet for ``idle_timeout_s``, a NUL terminator arrives, or
   ``max_response_bytes`` is exceeded;
3. before each command, drains anything left over from a previous response and warns about it (that means
   ``idle_timeout_s`` is too short for this server).

The client can only send :class:`~shared.rcon_protocol.ReadOnlyCommand` s; there is no raw-command method.
"""

import asyncio
import contextlib
import logging
from types import TracebackType
from typing import Self

from shared.rcon_protocol import (
    AUTH_ACCEPTED,
    TERMINATOR,
    ReadOnlyCommand,
    decode_response,
    encode_auth,
    encode_command,
)

logger = logging.getLogger(__name__)

_READ_CHUNK = 64 * 1024
_DRAIN_TIMEOUT_S = 0.01


class RconError(Exception):
    """Any failure talking to the RCON server. The connection is closed when this is raised."""


class RconAuthError(RconError):
    """The server rejected the password."""


class EvrimaRconClient:
    def __init__(
        self,
        host: str,
        port: int,
        password: str,
        *,
        connect_timeout_s: float = 5.0,
        response_timeout_s: float = 5.0,
        idle_timeout_s: float = 0.25,
        max_response_bytes: int = 1024 * 1024,
    ) -> None:
        self.host = host
        self.port = port
        self._password = password
        self.connect_timeout_s = connect_timeout_s
        self.response_timeout_s = response_timeout_s
        self.idle_timeout_s = idle_timeout_s
        self.max_response_bytes = max_response_bytes
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._lock = asyncio.Lock()

    @property
    def connected(self) -> bool:
        """
        Whether a logged-in connection is open.

        :return: True if connected
        """
        return self._writer is not None and not self._writer.is_closing()

    async def connect(self) -> None:
        """
        Open the connection and log in.

        :raises RconAuthError: if the password is rejected
        :raises RconError: on any other connection failure
        """
        async with self._lock:
            await self._close()
            try:
                self._reader, self._writer = await asyncio.wait_for(
                    asyncio.open_connection(self.host, self.port), timeout=self.connect_timeout_s
                )
            except (OSError, TimeoutError) as e:
                raise RconError(f"cannot connect to RCON at {self.host}:{self.port}: {e!r}") from e
            response = await self._exchange(encode_auth(self._password))
            if AUTH_ACCEPTED.lower() not in response.lower():
                await self._close()
                raise RconAuthError("RCON password rejected")

    async def request(self, command: ReadOnlyCommand) -> str:
        """
        Send a read-only command and return the complete response.

        :param command: the command to send
        :return: the decoded response text
        :raises RconError: if not connected or the exchange fails; the connection is closed
        """
        async with self._lock:
            if not self.connected:
                raise RconError("not connected")
            await self._drain_stale()
            return await self._exchange(encode_command(command))

    async def close(self) -> None:
        """Close the connection, if open."""
        async with self._lock:
            await self._close()

    async def __aenter__(self) -> Self:
        """Connect and log in."""
        await self.connect()
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        """Close the connection."""
        await self.close()

    async def _exchange(self, packet: bytes) -> str:
        assert self._reader is not None and self._writer is not None  # noqa: S101 - guarded by callers
        try:
            self._writer.write(packet)
            await self._writer.drain()
            raw = await self._read_response(self._reader)
        except RconError:
            await self._close()
            raise
        except (OSError, TimeoutError) as e:
            await self._close()
            raise RconError(f"RCON exchange failed: {e!r}") from e
        return decode_response(raw)

    async def _read_response(self, reader: asyncio.StreamReader) -> bytes:
        try:
            first = await asyncio.wait_for(reader.read(_READ_CHUNK), timeout=self.response_timeout_s)
        except TimeoutError as e:
            raise RconError(f"no response within {self.response_timeout_s}s") from e
        if not first:
            raise RconError("connection closed by server")
        buf = bytearray(first)
        while not buf.endswith(TERMINATOR):
            if len(buf) > self.max_response_bytes:
                raise RconError(f"response exceeds {self.max_response_bytes} bytes")
            try:
                chunk = await asyncio.wait_for(reader.read(_READ_CHUNK), timeout=self.idle_timeout_s)
            except TimeoutError:
                break
            if not chunk:
                break  # server closed after responding; the next request reconnects
            buf += chunk
        if len(buf) > self.max_response_bytes:
            raise RconError(f"response exceeds {self.max_response_bytes} bytes")
        return bytes(buf)

    async def _drain_stale(self) -> None:
        assert self._reader is not None  # noqa: S101 - guarded by request()
        stale = 0
        while True:
            try:
                chunk = await asyncio.wait_for(self._reader.read(_READ_CHUNK), timeout=_DRAIN_TIMEOUT_S)
            except TimeoutError:
                break
            if not chunk:
                await self._close()
                raise RconError("connection closed by server")
            stale += len(chunk)
        if stale:
            logger.warning(
                "discarded %d stale bytes from a previous RCON response; consider raising rcon.idle_timeout_s", stale
            )

    async def _close(self) -> None:
        writer, self._reader, self._writer = self._writer, None, None
        if writer is not None:
            writer.close()
            with contextlib.suppress(OSError):
                await writer.wait_closed()
