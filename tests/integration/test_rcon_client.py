import asyncio

import pytest

from agent.rcon import EvrimaRconClient, RconAuthError, RconError
from shared.playerdata import parse_player_data
from shared.rcon_protocol import ReadOnlyCommand
from tests.conftest import rcon_port
from tools.fake_rcon import FakeRconServer


def client_for(server: FakeRconServer, password: str = "secret", **kw: float) -> EvrimaRconClient:
    return EvrimaRconClient("127.0.0.1", rcon_port(server), password, **kw)  # type: ignore[arg-type]


async def test_login_and_player_data(fake_rcon: FakeRconServer) -> None:
    async with client_for(fake_rcon) as client:
        text = await client.request(ReadOnlyCommand.PLAYER_DATA)
    result = parse_player_data(text)
    assert len(result.players) == 6
    assert result.errors == []
    assert fake_rcon.received_opcodes == [0x77]


async def test_wrong_password(fake_rcon: FakeRconServer) -> None:
    client = client_for(fake_rcon, password="wrong")
    with pytest.raises(RconAuthError):
        await client.connect()
    assert not client.connected


async def test_connection_refused() -> None:
    client = EvrimaRconClient("127.0.0.1", 1, "x", connect_timeout_s=1.0)
    with pytest.raises(RconError, match="cannot connect"):
        await client.connect()


async def test_request_without_connect(fake_rcon: FakeRconServer) -> None:
    with pytest.raises(RconError, match="not connected"):
        await client_for(fake_rcon).request(ReadOnlyCommand.PLAYER_DATA)


@pytest.mark.parametrize("nul", [False, True])
async def test_fragmented_response_is_reassembled(fake_rcon: FakeRconServer, nul: bool) -> None:
    fake_rcon.options.chunk_size = 100
    fake_rcon.options.chunk_delay_s = 0.02
    fake_rcon.options.nul_terminate = nul
    async with client_for(fake_rcon, idle_timeout_s=0.2) as client:
        result = parse_player_data(await client.request(ReadOnlyCommand.PLAYER_DATA))
    assert len(result.players) == 6
    assert result.errors == []


async def test_slow_fragments_leave_stale_bytes_that_are_drained(
    fake_rcon: FakeRconServer, caplog: pytest.LogCaptureFixture
) -> None:
    """If the idle timeout is too short, the tail of a response must not be mistaken for the next response."""
    async with client_for(fake_rcon, idle_timeout_s=0.05) as client:
        fake_rcon.options.chunk_size = 200
        fake_rcon.options.chunk_delay_s = 0.15
        first = await client.request(ReadOnlyCommand.PLAYER_DATA)
        assert len(first) < 1000  # truncated
        await asyncio.sleep(1.5)  # let the rest arrive
        fake_rcon.options.chunk_size = 0
        second = await client.request(ReadOnlyCommand.PLAYER_DATA)
    assert second.startswith("[")
    assert len(parse_player_data(second).players) == 6
    assert "stale bytes" in caplog.text


async def test_oversized_response(fake_rcon: FakeRconServer) -> None:
    async with client_for(fake_rcon, max_response_bytes=1024) as client:
        fake_rcon.options.chunk_size = 256
        fake_rcon.options.chunk_delay_s = 0.01
        with pytest.raises(RconError, match="exceeds"):
            await client.request(ReadOnlyCommand.PLAYER_DATA)
        assert not client.connected


async def test_server_drop_is_detected_and_reconnect_works(fake_rcon: FakeRconServer) -> None:
    client = client_for(fake_rcon)
    await client.connect()
    await fake_rcon.drop_connections()
    with pytest.raises(RconError):
        await client.request(ReadOnlyCommand.PLAYER_DATA)
    assert not client.connected
    await client.connect()
    assert parse_player_data(await client.request(ReadOnlyCommand.PLAYER_DATA)).players
    await client.close()


async def test_no_response_times_out() -> None:
    async def silent(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        await reader.read(100)
        await asyncio.sleep(5)

    server = await asyncio.start_server(silent, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    try:
        client = EvrimaRconClient("127.0.0.1", port, "x", response_timeout_s=0.2)
        with pytest.raises(RconError, match="no response"):
            await client.connect()
    finally:
        server.close()
