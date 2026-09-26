"""
The Evrima RCON wire format.

Evrima does not speak Source RCON. It is a plain TCP stream with no length prefix or packet ids:

* auth:    ``0x01`` + password + ``0x00``; the server answers with text containing ``Password Accepted``.
* command: ``0x02`` + one opcode byte + parameters + ``0x00``; the server answers with text.

Reconstructed from three independent clients (smultar-dev/evrima.rcon, gamercon-async, butt4cak3/theislercon);
see NOTES.md for what is still unverified against a real server.

This module deliberately knows only the read-only opcodes. Admin commands (announce, kick, ban, ...) cannot be
encoded with it, which is how the agent guarantees it never changes anything on the game server.
"""

from enum import IntEnum

AUTH = 0x01
EXEC = 0x02
TERMINATOR = b"\x00"
AUTH_ACCEPTED = "Password Accepted"


class ReadOnlyCommand(IntEnum):
    """The only opcodes this project ever sends."""

    SERVER_DETAILS = 0x12
    PLAYER_LIST = 0x40
    PLAYER_DATA = 0x77


def encode_auth(password: str) -> bytes:
    """
    Encode the login packet.

    :param password: the RCON password
    :return: the bytes to send
    :raises ValueError: if the password is empty or contains a NUL byte, which would end the packet early
    """
    if not password:
        raise ValueError("RCON password is empty")
    if "\x00" in password:
        raise ValueError("RCON password must not contain NUL bytes")
    return bytes([AUTH]) + password.encode("utf-8") + TERMINATOR


def encode_command(command: ReadOnlyCommand) -> bytes:
    """
    Encode a read-only command. None of them take parameters.

    :param command: the command to send
    :return: the bytes to send
    :raises TypeError: if ``command`` is not a :class:`ReadOnlyCommand`, e.g. a raw admin opcode
    """
    if not isinstance(command, ReadOnlyCommand):
        raise TypeError(f"only read-only commands can be encoded, got {command!r}")
    return bytes([EXEC, command.value]) + TERMINATOR


def decode_response(raw: bytes) -> str:
    """
    Decode a response to text.

    Player names are arbitrary Unicode, so decode as UTF-8 and replace anything invalid rather than fail.

    :param raw: the bytes read from the socket
    :return: the response text without trailing NUL bytes
    """
    return raw.rstrip(TERMINATOR).decode("utf-8", errors="replace")
