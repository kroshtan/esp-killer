import pytest

from shared.rcon_protocol import ReadOnlyCommand, decode_response, encode_auth, encode_command


def test_auth_packet() -> None:
    assert encode_auth("hunter2") == b"\x01hunter2\x00"


@pytest.mark.parametrize("password", ["", "a\x00b"])
def test_auth_rejects_bad_passwords(password: str) -> None:
    with pytest.raises(ValueError):
        encode_auth(password)


@pytest.mark.parametrize(
    ("command", "packet"),
    [
        (ReadOnlyCommand.PLAYER_DATA, b"\x02\x77\x00"),
        (ReadOnlyCommand.PLAYER_LIST, b"\x02\x40\x00"),
        (ReadOnlyCommand.SERVER_DETAILS, b"\x02\x12\x00"),
    ],
)
def test_command_packets(command: ReadOnlyCommand, packet: bytes) -> None:
    assert encode_command(command) == packet


def test_only_read_only_opcodes_exist() -> None:
    # Adding an opcode here is a deliberate decision: the agent must never be able to change game state.
    assert {c.value for c in ReadOnlyCommand} == {0x12, 0x40, 0x77}


@pytest.mark.parametrize("opcode", [0x10, 0x20, 0x30, 0x50, 0x81])  # announce, ban, kick, save, whitelist
def test_admin_opcodes_cannot_be_encoded(opcode: int) -> None:
    with pytest.raises(TypeError):
        encode_command(opcode)  # type: ignore[arg-type]


def test_decode_strips_terminator_and_replaces_invalid_utf8() -> None:
    assert decode_response(b"Password Accepted\x00") == "Password Accepted"
    assert decode_response(b"Name: \xff\xfe") == "Name: ��"
