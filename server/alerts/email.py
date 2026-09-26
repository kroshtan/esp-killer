"""
Deliver an alert by email, through the SMTP server in :class:`~server.alerts.settings.SmtpSettings`.

smtplib is synchronous; :func:`send_email_async` runs it in a thread so the worker's event loop keeps going.
There are no retries here: every failure becomes a :class:`DeliveryError` and the alert outbox decides when to
try again. ``permanent`` is set when the mail server said no in a way that retrying cannot fix (bad
credentials, a rejected sender or recipient, any 5xx reply); connection problems and 4xx replies are transient.

The password and message bodies are never logged. The recipient is logged in redacted form only.
"""

import asyncio
import logging
import smtplib
import ssl
from email.message import EmailMessage
from email.utils import formatdate, make_msgid

from server.alerts.errors import DeliveryError
from server.alerts.message import IMAGE_FILENAME, Alert, email_content
from server.alerts.settings import SmtpSettings

logger = logging.getLogger(__name__)


def redact_address(address: str) -> str:
    """
    An email address in a form fit for logs: first character of the local part and the domain.

    :param address: e.g. ``admin@example.com``
    :return: e.g. ``a***@example.com``
    """
    local, at, domain = address.rpartition("@")
    if not at:
        return "***"
    return f"{local[:1]}***@{domain}"


def build_email(settings: SmtpSettings, to: str, alert: Alert) -> EmailMessage:
    """
    Build the email for an alert: plain text and HTML alternatives, plus the image as an attachment if any.

    :param settings: SMTP settings (for the from address)
    :param to: recipient address
    :param alert: the alert
    :return: the message
    """
    content = email_content(alert)
    msg = EmailMessage()
    msg["Subject"] = content.subject
    msg["From"] = settings.from_address or ""
    msg["To"] = to
    msg["Date"] = formatdate(alert.created_at.timestamp(), localtime=False, usegmt=True)
    msg["Message-ID"] = make_msgid(domain=_domain(settings.from_address))
    # Mark it as machine-generated so out-of-office replies don't come back to the alert address.
    msg["Auto-Submitted"] = "auto-generated"
    msg.set_content(content.text)
    msg.add_alternative(content.html, subtype="html")
    if alert.image_png is not None:
        msg.add_attachment(alert.image_png, maintype="image", subtype="png", filename=IMAGE_FILENAME)
    return msg


def send_email(settings: SmtpSettings, to: str, alert: Alert) -> None:
    """
    Send an alert by email. Blocks for up to a few times ``settings.timeout_s``.

    :param settings: SMTP settings
    :param to: recipient address (from the org's alert destinations)
    :param alert: the alert
    :raises DeliveryError: if SMTP is not configured, or the mail server could not be reached or refused the message
    """
    if not settings.is_configured or settings.host is None:
        raise DeliveryError("email alerts are not configured (ESPK_SMTP_HOST, ESPK_SMTP_FROM_ADDRESS)", permanent=True)
    msg = build_email(settings, to, alert)
    where = f"{settings.host}:{settings.port}"
    try:
        _deliver(settings, settings.host, msg)
    except (smtplib.SMTPException, OSError) as e:
        reason, permanent = classify_smtp_error(e)
        logger.log(logging.ERROR if permanent else logging.WARNING, "email alert via %s failed: %s", where, reason)
        raise DeliveryError(f"email via {where}: {reason}", permanent=permanent) from e
    logger.info("sent %s alert for flag #%d by email to %s", alert.kind, alert.flag_id, redact_address(to))


async def send_email_async(settings: SmtpSettings, to: str, alert: Alert) -> None:
    """
    :func:`send_email` in a worker thread.

    :param settings: SMTP settings
    :param to: recipient address
    :param alert: the alert
    """
    await asyncio.to_thread(send_email, settings, to, alert)


def classify_smtp_error(error: smtplib.SMTPException | OSError) -> tuple[str, bool]:
    """
    Describe an SMTP failure and decide whether it is permanent.

    :param error: what smtplib (or the socket underneath) raised
    :return: a reason fit for logs (no secrets, no message content), and whether retrying is pointless
    """
    match error:
        case smtplib.SMTPServerDisconnected() | smtplib.SMTPConnectError() | TimeoutError():
            return f"connection failed ({type(error).__name__})", False
        case smtplib.SMTPAuthenticationError():
            return f"authentication failed (SMTP {error.smtp_code})", True
        case smtplib.SMTPRecipientsRefused():
            codes = [code for code, _ in error.recipients.values()]
            permanent = not codes or any(code >= 500 for code in codes)  # noqa: PLR2004
            return f"recipient refused (SMTP {', '.join(map(str, codes))})", permanent
        case smtplib.SMTPResponseException():
            # Sender refused, data refused, ...: 5xx replies are final, 4xx are "try again later".
            return f"server replied {error.smtp_code}", error.smtp_code >= 500  # noqa: PLR2004
        case smtplib.SMTPNotSupportedError():
            # e.g. our settings require STARTTLS and the server doesn't offer it.
            return f"not supported by the server: {error}", True
        case _:
            # OSError: connection refused, DNS failure, TLS errors. Other SMTPExceptions: unexpected replies.
            kind = "connection failed" if isinstance(error, OSError) else "SMTP error"
            return f"{kind} ({type(error).__name__})", False


def _deliver(settings: SmtpSettings, host: str, msg: EmailMessage) -> None:
    """Connect, secure, authenticate and send."""
    context = ssl.create_default_context()
    smtp: smtplib.SMTP
    if settings.use_ssl:
        smtp = smtplib.SMTP_SSL(host, settings.port, timeout=settings.timeout_s, context=context)
    else:
        smtp = smtplib.SMTP(host, settings.port, timeout=settings.timeout_s)
    with smtp:
        if settings.starttls and not settings.use_ssl:
            smtp.starttls(context=context)
        if settings.username:
            password = settings.password.get_secret_value() if settings.password is not None else ""
            smtp.login(settings.username, password)
        smtp.send_message(msg)


def _domain(address: str | None) -> str | None:
    if not address or "@" not in address:
        return None
    return address.rpartition("@")[2].strip(" >") or None
