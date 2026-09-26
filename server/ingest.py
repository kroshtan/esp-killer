"""
``POST /v1/ingest``.

Checks happen cheapest first, and the body is not read until the caller is authenticated and within its rate
limit:

1. ``Authorization: Bearer <key>`` -> key hash -> org/server (401 if unknown or revoked);
2. per-key rate limit (429 with ``Retry-After``);
3. body size, before and after gzip (413), content encoding (415);
4. schema validation (422). The error details never echo the submitted values, which are personal data;
5. snapshots too far in the future (agent clock skew) or older than the retention period are rejected;
6. deduplicated store (a retried upload is accepted, not stored twice).

Nothing from the payload is logged: only org/server ids and counts.
"""

import logging
import zlib
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.concurrency import run_in_threadpool
from pydantic import ValidationError

from server.db.repository import Repository
from server.keys import hash_key, looks_like_key
from server.orgconfig import ConfigStore, ServerIdentity
from server.ratelimit import TokenBucketLimiter
from server.settings import ServerSettings
from shared.models import IngestBatch, IngestResponse, ParseHealth

logger = logging.getLogger(__name__)

router = APIRouter()


@dataclass
class AppState:
    settings: ServerSettings
    config: ConfigStore
    repo: Repository
    limiter: TokenBucketLimiter


def get_state(request: Request) -> AppState:
    """
    The per-app dependencies, set up in :func:`server.app.create_app`.

    :param request: the current request
    :return: the app state
    """
    state: AppState = request.app.state.espk
    return state


def authenticate(request: Request, state: Annotated[AppState, Depends(get_state)]) -> tuple[ServerIdentity, str]:
    """
    Resolve the bearer key to its server.

    :param request: the current request
    :param state: app state
    :return: the server identity and the key hash
    :raises HTTPException: 401 if the key is missing, malformed, unknown or revoked
    """
    scheme, _, token = request.headers.get("Authorization", "").partition(" ")
    token = token.strip()
    identity = None
    key_hash = ""
    if scheme.lower() == "bearer" and looks_like_key(token):
        key_hash = hash_key(token)
        identity = state.config.identify(key_hash)
    if identity is None:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED, "missing or invalid API key", headers={"WWW-Authenticate": "Bearer"}
        )
    return identity, key_hash


@router.post("/v1/ingest")
async def ingest(
    request: Request,
    auth: Annotated[tuple[ServerIdentity, str], Depends(authenticate)],
    state: Annotated[AppState, Depends(get_state)],
) -> IngestResponse:
    """
    Accept a batch of snapshots from an agent.

    :param request: the current request
    :param auth: the authenticated server and key hash
    :param state: app state
    :return: counts of what was stored
    :raises HTTPException: on rate limiting, oversized or invalid payloads
    """
    identity, key_hash = auth
    wait = state.limiter.acquire(key_hash)
    if wait > 0:
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            "rate limit exceeded",
            headers={"Retry-After": TokenBucketLimiter.retry_after_header(wait)},
        )

    body = await _read_body(request, state.settings)
    try:
        batch = IngestBatch.model_validate_json(body)
    except ValidationError as e:
        errors = e.errors(include_input=False, include_url=False, include_context=False)
        logger.info("rejected invalid batch from %s/%s: %d error(s)", identity.org_id, identity.server_id, len(errors))
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, errors[:20]) from e

    received_at = datetime.now(UTC)
    newest = received_at + timedelta(seconds=state.settings.max_clock_skew_s)
    oldest = received_at - timedelta(days=state.settings.retention_days)
    valid = [s for s in batch.snapshots if oldest <= s.captured_at <= newest]
    rejected = len(batch.snapshots) - len(valid)
    if rejected:
        logger.warning(
            "%s/%s: rejected %d snapshot(s) with timestamps outside [-%dd, +%.0fs]; is the agent's clock right?",
            identity.org_id,
            identity.server_id,
            rejected,
            state.settings.retention_days,
            state.settings.max_clock_skew_s,
        )

    result = await run_in_threadpool(state.repo.ingest, identity.org_id, identity.server_id, valid, received_at)
    before = await run_in_threadpool(
        state.repo.record_agent_status,
        identity.org_id,
        identity.server_id,
        batch.agent_version,
        batch.parse_health,
        received_at,
    )
    _warn_on_new_parse_problems(identity, before.total_health, batch.parse_health)
    logger.info(
        "%s/%s: %d snapshot(s) stored (%d rows), %d duplicate",
        identity.org_id,
        identity.server_id,
        result.accepted_snapshots,
        result.rows,
        result.duplicate_snapshots,
    )
    return IngestResponse(
        accepted_snapshots=result.accepted_snapshots,
        duplicate_snapshots=result.duplicate_snapshots,
        rejected_snapshots=rejected,
        rows=result.rows,
        received_at=received_at,
    )


def _warn_on_new_parse_problems(
    identity: ServerIdentity, before: ParseHealth | None, health: ParseHealth | None
) -> None:
    """Log once when a server's agent starts reporting lines it cannot parse (a game update changed RCON?)."""
    if health is None or health.problems == 0 or (before is not None and before.problems > 0):
        return
    logger.warning(
        "%s/%s: the agent could not parse %d of %d player line(s) (%d salvaged); reasons %s, unknown fields %s."
        " The RCON format may have changed; see `python -m server servers`.",
        identity.org_id,
        identity.server_id,
        health.problems,
        health.lines + health.salvaged,
        health.salvaged,
        health.errors,
        health.unknown_keys,
    )


async def _read_body(request: Request, settings: ServerSettings) -> bytes:
    too_large = HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, "payload too large")
    declared = request.headers.get("Content-Length")
    if declared is not None and declared.isdigit() and int(declared) > settings.max_body_bytes:
        raise too_large
    # Content-Length can be absent (chunked) or wrong, so count while streaming too.
    body = bytearray()
    async for chunk in request.stream():
        body += chunk
        if len(body) > settings.max_body_bytes:
            raise too_large

    encoding = request.headers.get("Content-Encoding", "identity").strip().lower()
    if encoding == "identity":
        return bytes(body)
    if encoding != "gzip":
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "Content-Encoding must be gzip or identity")
    decompressor = zlib.decompressobj(wbits=16 + zlib.MAX_WBITS)
    try:
        # Decompress at most one byte past the cap: enough to know the payload is too big, without
        # materialising a zip bomb.
        out = decompressor.decompress(bytes(body), settings.max_decompressed_bytes + 1)
    except zlib.error as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "invalid gzip body") from e
    if len(out) > settings.max_decompressed_bytes:
        raise too_large
    if not decompressor.eof:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "truncated gzip body")
    return out
