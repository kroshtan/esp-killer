"""
Deliver an alert to a Discord webhook.

Retries happen here only for the short term: a rate limit Discord asks us to wait out, or a brief server error.
Anything longer is the alert outbox's job, which is why the final failure is a :class:`DeliveryError` that says
whether retrying later can help. A 4xx other than 429 cannot: the webhook was deleted (404), its token is wrong
(401) or the payload is invalid (400), and resending the same request will fail the same way.

A webhook URL contains its token and lets anyone post to the channel, so it is a secret: it is never logged or
put in an exception message, only its redacted form (:func:`redact_webhook`). httpx itself logs every request
URL at INFO, so :func:`send_discord` also installs a filter on the ``httpx`` logger that redacts webhook URLs.
"""

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from urllib.parse import urlsplit

import httpx

from server.alerts.errors import DeliveryError
from server.alerts.message import IMAGE_FILENAME, Alert, discord_payload

logger = logging.getLogger(__name__)

Sleep = Callable[[float], Awaitable[None]]


def redact_webhook(url: str) -> str:
    """
    A form of a webhook URL that is safe to log: host and webhook id, without the token or query.

    :param url: the webhook URL
    :return: e.g. ``https://discord.com/api/webhooks/123/***``
    """
    try:
        parts = urlsplit(url)
    except ValueError:
        return "<invalid URL>"
    segments = parts.path.split("/")
    if "webhooks" in segments:
        keep = segments.index("webhooks") + 2  # ".../webhooks/<id>"
        path = "/".join(segments[:keep]) + ("/***" if len(segments) > keep else "")
    else:
        path = "/***" if parts.path.strip("/") else ""
    host = parts.hostname or ""
    return f"{parts.scheme}://{host}{path}"


class WebhookUrlFilter(logging.Filter):
    """Redact webhook URLs in log record arguments (httpx logs ``HTTP Request: POST <url> ...``)."""

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Rewrite the record's arguments in place; never drops a record.

        :param record: the log record
        :return: True
        """
        if isinstance(record.args, tuple) and any("/webhooks/" in str(arg) for arg in record.args):
            record.args = tuple(redact_webhook(str(arg)) if "/webhooks/" in str(arg) else arg for arg in record.args)
        return True


def install_log_redaction() -> None:
    """Add :class:`WebhookUrlFilter` to the ``httpx`` logger, once."""
    httpx_logger = logging.getLogger("httpx")
    if not any(isinstance(f, WebhookUrlFilter) for f in httpx_logger.filters):
        httpx_logger.addFilter(WebhookUrlFilter())


async def send_discord(
    client: httpx.AsyncClient,
    webhook_url: str,
    alert: Alert,
    *,
    max_attempts: int = 4,
    backoff_base_s: float = 1.0,
    max_wait_s: float = 60.0,
    sleep: Sleep = asyncio.sleep,
) -> None:
    """
    Post an alert to a Discord webhook, with its image attached if it has one.

    Rate limits (429) are waited out as Discord asks (``retry_after`` in the body, else ``Retry-After``); 5xx
    responses and network errors are retried with exponential backoff. Both count against ``max_attempts``.

    :param client: HTTP client; its timeout applies to each attempt
    :param webhook_url: the webhook URL (secret)
    :param alert: the alert to send
    :param max_attempts: attempts in total before giving up
    :param backoff_base_s: first backoff after a 5xx or network error; doubles each time
    :param max_wait_s: longest single wait. If Discord asks for longer, give up and leave it to the outbox.
    :param sleep: how to wait; tests pass a fake
    :raises DeliveryError: if the alert could not be delivered. ``permanent`` is True for a 4xx other than 429.
    """
    install_log_redaction()
    where = redact_webhook(webhook_url)
    payload = discord_payload(alert)
    reason = "no attempt made"
    retry_after: float | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = await _post(client, webhook_url, payload, alert.image_png)
        except httpx.HTTPError as e:
            # Transport errors don't carry the URL in their message, but name only the type to be safe.
            reason, delay, retry_after = (
                f"network error ({type(e).__name__})",
                backoff_base_s * 2 ** (attempt - 1),
                None,
            )
        else:
            status = response.status_code
            if response.is_success:
                logger.info("sent %s alert for flag #%d to Discord webhook %s", alert.kind, alert.flag_id, where)
                return
            if status == 429:  # noqa: PLR2004
                retry_after = _retry_after(response)
                reason, delay = "rate limited (HTTP 429)", retry_after if retry_after is not None else backoff_base_s
            elif response.is_server_error:
                reason, delay, retry_after = f"HTTP {status}", backoff_base_s * 2 ** (attempt - 1), None
            else:
                logger.error("Discord webhook %s rejected flag #%d with HTTP %d", where, alert.flag_id, status)
                raise DeliveryError(f"Discord webhook {where} rejected the alert: HTTP {status}", permanent=True)
        if attempt == max_attempts:
            break
        if delay > max_wait_s:
            logger.warning("Discord webhook %s: %s, asked to wait %.0fs; leaving it for later", where, reason, delay)
            raise DeliveryError(f"Discord webhook {where}: {reason}", permanent=False, retry_after_s=delay)
        logger.warning(
            "Discord webhook %s: %s (attempt %d/%d), retrying in %.1fs", where, reason, attempt, max_attempts, delay
        )
        await sleep(delay)
    logger.error("giving up on Discord webhook %s for flag #%d: %s", where, alert.flag_id, reason)
    raise DeliveryError(
        f"Discord webhook {where}: {reason} after {max_attempts} attempts", permanent=False, retry_after_s=retry_after
    )


async def _post(
    client: httpx.AsyncClient, webhook_url: str, payload: dict[str, object], image_png: bytes | None
) -> httpx.Response:
    """POST as JSON, or as multipart with ``payload_json`` and ``files[0]`` when there is an image."""
    if image_png is None:
        return await client.post(webhook_url, json=payload)
    return await client.post(
        webhook_url,
        data={"payload_json": json.dumps(payload)},
        files={"files[0]": (IMAGE_FILENAME, image_png, "image/png")},
    )


def _retry_after(response: httpx.Response) -> float | None:
    """How long Discord asks us to wait: ``retry_after`` (seconds) in the JSON body, else the header."""
    try:
        body = response.json()
    except ValueError:
        body = None
    candidates = [body.get("retry_after") if isinstance(body, dict) else None, response.headers.get("Retry-After")]
    for value in candidates:
        if value is None or isinstance(value, bool):
            continue
        try:
            seconds = float(value)
        except (TypeError, ValueError):
            continue
        if seconds >= 0 and seconds != float("inf"):
            return seconds
    return None
