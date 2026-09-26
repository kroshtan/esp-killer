import gzip
import json
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from agent.health import HealthTally
from agent.queue import SnapshotQueue
from agent.uploader import Outcome, Uploader
from shared.playerdata import parse_player_data
from tests.helpers import snapshot

URL = "https://espk.test/v1/ingest"


def make(
    tmp_path: Path, handler: Callable[[httpx.Request], httpx.Response], n: int = 5
) -> tuple[Uploader, SnapshotQueue]:
    queue = SnapshotQueue(tmp_path / "q.db", max_bytes=10**7)
    for _ in range(n):
        queue.put(snapshot())
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return Uploader(queue, http, URL, "espk_key", max_batch=4), queue


async def test_success_acks_and_sends_gzip_with_bearer(tmp_path: Path) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={})

    uploader, queue = make(tmp_path, handler)
    assert await uploader.upload_once() is Outcome.SENT
    assert len(queue) == 1
    assert await uploader.upload_once() is Outcome.SENT
    assert await uploader.upload_once() is Outcome.EMPTY
    assert seen[0].headers["Authorization"] == "Bearer espk_key"
    assert seen[0].headers["Content-Encoding"] == "gzip"


@pytest.mark.parametrize("status", [401, 403, 429, 500, 503])
async def test_transient_failures_keep_the_data(tmp_path: Path, status: int) -> None:
    uploader, queue = make(tmp_path, lambda _: httpx.Response(status, headers={"Retry-After": "30"}))
    assert await uploader.upload_once() is Outcome.RETRY
    assert len(queue) == 5
    assert uploader.retry_after_s > 0
    if status == 429:
        assert uploader.retry_after_s >= 30


async def test_network_error_keeps_the_data(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down", request=request)

    uploader, queue = make(tmp_path, handler)
    assert await uploader.upload_once() is Outcome.RETRY
    assert len(queue) == 5


async def test_permanent_rejection_drops_the_batch(tmp_path: Path) -> None:
    uploader, queue = make(tmp_path, lambda _: httpx.Response(422))
    assert await uploader.upload_once() is Outcome.DROPPED
    assert len(queue) == 1


async def test_payload_too_large_halves_the_batch(tmp_path: Path) -> None:
    sizes: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        n = len(json.loads(gzip.decompress(request.content))["snapshots"])
        sizes.append(n)
        return httpx.Response(413 if n > 1 else 200)

    uploader, queue = make(tmp_path, handler)
    outcomes = [await uploader.upload_once() for _ in range(3)]
    assert outcomes == [Outcome.RETRY, Outcome.RETRY, Outcome.SENT]
    assert sizes == [4, 2, 1]
    assert len(queue) == 4


async def test_parse_health_travels_with_the_batch_and_survives_failures(tmp_path: Path) -> None:
    bodies: list[dict[str, object]] = []
    status = {"code": 503}

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(gzip.decompress(request.content)))
        return httpx.Response(status["code"])

    uploader, _ = make(tmp_path, handler)
    uploader.health = HealthTally()
    uploader.health.record(parse_player_data("Name: A, PlayerID: 1, Location: X=0 Y=0 Z=0"))
    assert await uploader.upload_once() is Outcome.RETRY
    status["code"] = 200
    assert await uploader.upload_once() is Outcome.SENT
    assert bodies[0]["parse_health"]["polls"] == 1  # type: ignore[index]
    assert bodies[1]["parse_health"]["polls"] == 1  # type: ignore[index]  # given back after the failure
    assert uploader.health.take() is None
