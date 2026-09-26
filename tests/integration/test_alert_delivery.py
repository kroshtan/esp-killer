import base64
import email.parser
import email.policy
import json
import logging
import smtplib
import socketserver
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from email.message import EmailMessage

import httpx
import pytest

from server.alerts.discord import redact_webhook, send_discord
from server.alerts.email import build_email, classify_smtp_error, redact_address, send_email, send_email_async
from server.alerts.errors import DeliveryError
from server.alerts.message import Alert
from server.alerts.settings import SmtpSettings

WEBHOOK = "https://discord.com/api/webhooks/123456/s3cr3t-t0ken"
PNG = b"\x89PNG\r\n\x1a\n" + bytes(range(256))

ALERT = Alert(
    kind="flag",
    org_id="demo",
    player_id="76561198000000001",
    player_name="Rex",
    server_ids=("gateway-1",),
    score=0.87,
    details={"servers": ["gateway-1"], "ambush": {"score": 0.5, "waits": 23, "hits": 8, "null_hits": 1.3, "z": 5.0}},
    created_at=datetime(2026, 9, 26, 12, 30, tzinfo=UTC),
    flag_id=42,
)


# --- Discord -------------------------------------------------------------------------------------------------


class Sleeps:
    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, delay: float) -> None:
        self.delays.append(delay)


def client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def parse_multipart(request: httpx.Request) -> EmailMessage:
    raw = b"Content-Type: " + request.headers["Content-Type"].encode() + b"\r\n\r\n" + request.content
    msg = email.parser.BytesParser(policy=email.policy.default).parsebytes(raw)
    assert isinstance(msg, EmailMessage)
    return msg


async def test_discord_sends_json_without_image() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(204)

    async with client(handler) as http:
        await send_discord(http, WEBHOOK, ALERT)
    (request,) = seen
    assert str(request.url) == WEBHOOK
    assert request.headers["Content-Type"] == "application/json"
    payload = json.loads(request.content)
    assert payload["allowed_mentions"] == {"parse": []}
    assert payload["embeds"][0]["title"] == "Possible ESP user: Rex"


async def test_discord_sends_multipart_with_image() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"id": "1"})

    async with client(handler) as http:
        await send_discord(http, WEBHOOK, replace(ALERT, image_png=PNG))
    msg = parse_multipart(seen[0])
    parts = {part.get_param("name", header="Content-Disposition"): part for part in msg.iter_parts()}
    payload = json.loads(parts["payload_json"].get_content())
    assert payload["embeds"][0]["image"] == {"url": "attachment://evidence.png"}
    assert payload["attachments"] == [{"id": 0, "filename": "evidence.png"}]
    image = parts["files[0]"]
    assert image.get_filename() == "evidence.png"
    assert image.get_content_type() == "image/png"
    assert image.get_payload(decode=True) == PNG


@pytest.mark.parametrize(
    ("response", "expected_wait"),
    [
        (httpx.Response(429, json={"retry_after": 2.5, "global": False}), 2.5),
        (httpx.Response(429, headers={"Retry-After": "3"}), 3.0),
        (httpx.Response(429, text="slow down"), 1.0),  # nothing said: backoff_base_s
    ],
)
async def test_discord_honours_rate_limits(response: httpx.Response, expected_wait: float) -> None:
    responses = iter([response, httpx.Response(204)])
    sleeps = Sleeps()
    async with client(lambda _: next(responses)) as http:
        await send_discord(http, WEBHOOK, ALERT, sleep=sleeps)
    assert sleeps.delays == [expected_wait]


async def test_discord_leaves_long_rate_limits_to_the_outbox() -> None:
    sleeps = Sleeps()
    async with client(lambda _: httpx.Response(429, json={"retry_after": 600})) as http:
        with pytest.raises(DeliveryError) as info:
            await send_discord(http, WEBHOOK, ALERT, sleep=sleeps, max_wait_s=60)
    assert not info.value.permanent
    assert info.value.retry_after_s == 600
    assert sleeps.delays == []


async def test_discord_retries_server_errors_then_gives_up(caplog: pytest.LogCaptureFixture) -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(503)

    sleeps = Sleeps()
    caplog.set_level(logging.DEBUG)
    async with client(handler) as http:
        with pytest.raises(DeliveryError) as info:
            await send_discord(http, WEBHOOK, ALERT, sleep=sleeps, max_attempts=3, backoff_base_s=0.5)
    assert len(calls) == 3
    assert sleeps.delays == [0.5, 1.0]
    assert not info.value.permanent
    assert "503" in str(info.value)
    assert "s3cr3t" not in str(info.value)
    assert "s3cr3t" not in caplog.text


async def test_discord_retries_network_errors() -> None:
    outcomes: list[Exception | httpx.Response] = [httpx.ConnectError("down"), httpx.Response(204)]

    def handler(request: httpx.Request) -> httpx.Response:
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    sleeps = Sleeps()
    async with client(handler) as http:
        await send_discord(http, WEBHOOK, ALERT, sleep=sleeps)
    assert sleeps.delays == [1.0]


@pytest.mark.parametrize("status", [400, 401, 404])
async def test_discord_client_errors_are_permanent(status: int, caplog: pytest.LogCaptureFixture) -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(status, json={"message": "Unknown Webhook", "code": 10015})

    sleeps = Sleeps()
    caplog.set_level(logging.DEBUG)
    async with client(handler) as http:
        with pytest.raises(DeliveryError) as info:
            await send_discord(http, WEBHOOK, ALERT, sleep=sleeps)
    assert info.value.permanent
    assert len(calls) == 1
    assert sleeps.delays == []
    assert "s3cr3t" not in str(info.value)
    assert "s3cr3t" not in caplog.text


def test_redact_webhook() -> None:
    assert redact_webhook(WEBHOOK) == "https://discord.com/api/webhooks/123456/***"
    assert redact_webhook("https://discord.com/api/webhooks/123456/tok?wait=true") == (
        "https://discord.com/api/webhooks/123456/***"
    )
    assert redact_webhook("https://user:pw@example.com/hook/secret") == "https://example.com/***"


# --- email ---------------------------------------------------------------------------------------------------


@dataclass
class Received:
    mail_from: str
    rcpt_to: list[str]
    message: EmailMessage


@dataclass
class SmtpState:
    messages: list[Received] = field(default_factory=list)
    credentials: tuple[str, str] = ("alerts", "hunter2")
    refuse: set[str] = field(default_factory=set)
    logins: list[str] = field(default_factory=list)


class SmtpHandler(socketserver.StreamRequestHandler):
    """Just enough ESMTP for smtplib: EHLO, AUTH PLAIN, MAIL, RCPT, DATA, RSET, QUIT."""

    state: SmtpState

    def reply(self, line: str) -> None:
        self.wfile.write(line.encode() + b"\r\n")

    def handle(self) -> None:
        self.reply("220 test ESMTP")
        self.mail_from = ""
        self.rcpt_to: list[str] = []
        while line := self.rfile.readline().decode().rstrip("\r\n"):
            verb, _, arg = line.partition(" ")
            match verb.upper():
                case "EHLO":
                    self.reply("250-test")
                    self.reply("250 AUTH PLAIN")
                case "AUTH":
                    _, user, password = base64.b64decode(arg.split()[1]).decode().split("\0")
                    self.state.logins.append(user)
                    ok = (user, password) == self.state.credentials
                    self.reply("235 ok" if ok else "535 bad credentials")
                case "MAIL":
                    self.mail_from, self.rcpt_to = arg.removeprefix("FROM:").strip("<>"), []
                    self.reply("250 ok")
                case "RCPT":
                    self.recipient(arg.removeprefix("TO:").strip("<>"))
                case "DATA":
                    self.data()
                case "RSET" | "NOOP":
                    self.reply("250 ok")
                case "QUIT":
                    self.reply("221 bye")
                    return
                case _:
                    self.reply("502 not implemented")

    def recipient(self, address: str) -> None:
        if address in self.state.refuse:
            self.reply("550 no such user")
        else:
            self.rcpt_to.append(address)
            self.reply("250 ok")

    def data(self) -> None:
        self.reply("354 go ahead")
        lines = []
        while (chunk := self.rfile.readline()) not in {b".\r\n", b""}:
            lines.append(chunk.removeprefix(b"."))
        parsed = email.parser.BytesParser(policy=email.policy.default).parsebytes(b"".join(lines))
        assert isinstance(parsed, EmailMessage)
        self.state.messages.append(Received(self.mail_from, self.rcpt_to, parsed))
        self.reply("250 queued")


@dataclass
class SmtpServer:
    port: int
    state: SmtpState


@pytest.fixture
def smtp_server() -> Iterator[SmtpServer]:
    state = SmtpState()
    handler = type("Handler", (SmtpHandler,), {"state": state})
    with socketserver.ThreadingTCPServer(("127.0.0.1", 0), handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        yield SmtpServer(port=server.server_address[1], state=state)
        server.shutdown()


def smtp_settings(port: int, **changes: object) -> SmtpSettings:
    values: dict[str, object] = {
        "host": "127.0.0.1",
        "port": port,
        "username": "alerts",
        "password": "hunter2",
        "from_address": "esp-killer <alerts@espk.test>",
        "starttls": False,
        "timeout_s": 5,
    }
    return SmtpSettings.model_validate({**values, **changes})


def test_email_is_delivered_with_attachment(smtp_server: SmtpServer, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    send_email(smtp_settings(smtp_server.port), "admin@example.com", replace(ALERT, image_png=PNG))
    (received,) = smtp_server.state.messages
    assert smtp_server.state.logins == ["alerts"]
    assert received.mail_from == "alerts@espk.test"
    assert received.rcpt_to == ["admin@example.com"]
    msg = received.message
    assert msg["To"] == "admin@example.com"
    assert msg["From"] == "esp-killer <alerts@espk.test>"
    assert msg["Subject"] == "[esp-killer] Possible ESP user: Rex (demo)"
    assert msg["Auto-Submitted"] == "auto-generated"
    assert msg["Message-ID"].endswith("@espk.test>")
    body = msg.get_body(preferencelist=("plain",))
    assert body is not None
    assert "8 of 23 waits" in body.get_content()
    html = msg.get_body(preferencelist=("html",))
    assert html is not None
    assert "<li>" in html.get_content()
    (attachment,) = msg.iter_attachments()
    assert attachment.get_filename() == "evidence.png"
    assert attachment.get_content_type() == "image/png"
    assert attachment.get_payload(decode=True) == PNG
    assert "hunter2" not in caplog.text
    assert "a***@example.com" in caplog.text


async def test_email_async_without_image_or_login(smtp_server: SmtpServer) -> None:
    await send_email_async(smtp_settings(smtp_server.port, username=None), "admin@example.com", ALERT)
    (received,) = smtp_server.state.messages
    assert smtp_server.state.logins == []
    assert list(received.message.iter_attachments()) == []


def test_email_bad_credentials_are_permanent(smtp_server: SmtpServer, caplog: pytest.LogCaptureFixture) -> None:
    with pytest.raises(DeliveryError) as info:
        send_email(smtp_settings(smtp_server.port, password="wrong"), "admin@example.com", ALERT)
    assert info.value.permanent
    assert "authentication" in str(info.value)
    assert "wrong" not in caplog.text
    assert smtp_server.state.messages == []


def test_email_refused_recipient_is_permanent(smtp_server: SmtpServer) -> None:
    smtp_server.state.refuse.add("nobody@example.com")
    with pytest.raises(DeliveryError) as info:
        send_email(smtp_settings(smtp_server.port), "nobody@example.com", ALERT)
    assert info.value.permanent
    assert "550" in str(info.value)


def test_email_connection_failure_is_transient() -> None:
    with socketserver.TCPServer(("127.0.0.1", 0), socketserver.BaseRequestHandler) as unused:
        port = unused.server_address[1]
    with pytest.raises(DeliveryError) as info:
        send_email(smtp_settings(port), "admin@example.com", ALERT)
    assert not info.value.permanent


def test_email_starttls_unsupported_is_permanent(smtp_server: SmtpServer) -> None:
    with pytest.raises(DeliveryError) as info:
        send_email(smtp_settings(smtp_server.port, starttls=True), "admin@example.com", ALERT)
    assert info.value.permanent


def test_email_not_configured_is_permanent() -> None:
    with pytest.raises(DeliveryError) as info:
        send_email(SmtpSettings(host=None), "admin@example.com", ALERT)
    assert info.value.permanent


def test_build_email_rejoin_subject() -> None:
    msg = build_email(smtp_settings(25), "admin@example.com", replace(ALERT, kind="rejoin"))
    assert msg["Subject"].startswith("[esp-killer] Flagged player rejoined: Rex")


def test_redact_address() -> None:
    assert redact_address("admin@example.com") == "a***@example.com"
    assert redact_address("not-an-address") == "***"


def test_smtp_settings_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("HOST", "PORT", "USERNAME", "PASSWORD", "FROM_ADDRESS", "STARTTLS", "USE_SSL", "TIMEOUT_S"):
        monkeypatch.delenv(f"ESPK_SMTP_{name}", raising=False)
    defaults = SmtpSettings()
    assert not defaults.is_configured
    assert (defaults.port, defaults.starttls, defaults.use_ssl) == (587, True, False)

    monkeypatch.setenv("ESPK_SMTP_HOST", "smtp.example.com")
    assert not SmtpSettings().is_configured
    monkeypatch.setenv("ESPK_SMTP_FROM_ADDRESS", "alerts@example.com")
    monkeypatch.setenv("ESPK_SMTP_PORT", "465")
    monkeypatch.setenv("ESPK_SMTP_USE_SSL", "true")
    monkeypatch.setenv("ESPK_SMTP_PASSWORD", "hunter2")
    monkeypatch.setenv("ESPK_SMTP_TIMEOUT_S", "7.5")
    settings = SmtpSettings()
    assert settings.is_configured
    assert (settings.port, settings.use_ssl, settings.timeout_s) == (465, True, 7.5)
    assert settings.password is not None
    assert settings.password.get_secret_value() == "hunter2"
    assert "hunter2" not in repr(settings)


@pytest.mark.parametrize(
    ("error", "permanent"),
    [
        (smtplib.SMTPSenderRefused(553, b"sender not allowed", "alerts@espk.test"), True),
        (smtplib.SMTPDataError(451, b"try later"), False),
        (smtplib.SMTPRecipientsRefused({"a@example.com": (450, b"mailbox busy")}), False),
        (smtplib.SMTPConnectError(554, b"no service"), False),
        (smtplib.SMTPException("odd"), False),
        (ConnectionRefusedError(), False),
    ],
)
def test_classify_smtp_error(error: smtplib.SMTPException | OSError, permanent: bool) -> None:
    reason, is_permanent = classify_smtp_error(error)
    assert is_permanent is permanent
    assert reason
