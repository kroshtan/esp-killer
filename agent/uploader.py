"""
Upload queued snapshots to the backend.

Snapshots leave the queue only after the backend has them. How each response is handled:

* 2xx: acknowledged and removed from the queue.
* 401/403: the key is wrong or revoked. Keep the data and back off; an operator will fix the config, and the
  queue's size cap bounds what accumulates meanwhile.
* 413: the batch is too big. Halve the batch size and retry.
* 429, 5xx, network errors: back off (honouring ``Retry-After``) and retry.
* any other 4xx: the backend will never accept this batch. Drop it and log, so one bad batch cannot block the
  queue forever.

The backend deduplicates on ``snapshot_id``, so resending after a lost response is harmless.
"""

import asyncio
import contextlib
import gzip
import logging
from enum import Enum

import httpx

from agent import __version__
from agent.backoff import Backoff
from agent.queue import SnapshotQueue
from shared.models import IngestBatch

logger = logging.getLogger(__name__)

# While draining a backlog, pause briefly between batches to stay inside the backend's rate limit.
BACKLOG_PAUSE_S = 1.0


class Outcome(Enum):
    EMPTY = "empty"
    SENT = "sent"
    DROPPED = "dropped"
    RETRY = "retry"


class Uploader:
    def __init__(
        self,
        queue: SnapshotQueue,
        http: httpx.AsyncClient,
        ingest_url: str,
        api_key: str,
        *,
        max_batch: int = 200,
        interval_s: float = 15.0,
    ) -> None:
        self.queue = queue
        self.http = http
        self.ingest_url = ingest_url
        self._api_key = api_key
        self.max_batch = max_batch
        self.batch_size = max_batch
        self.interval_s = interval_s
        self.retry_after_s: float = 0.0
        self._backoff = Backoff(base_s=2.0, cap_s=300.0)

    async def upload_once(self) -> Outcome:
        """
        Send the oldest batch in the queue.

        After a ``RETRY`` outcome, :attr:`retry_after_s` says how long to wait.

        :return: what happened
        """
        items = self.queue.peek(self.batch_size)
        if not items:
            return Outcome.EMPTY
        ids = [queue_id for queue_id, _ in items]
        batch = IngestBatch(agent_version=__version__, snapshots=[snapshot for _, snapshot in items])
        body = gzip.compress(batch.model_dump_json().encode("utf-8"))
        try:
            response = await self.http.post(
                self.ingest_url,
                content=body,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                    "Content-Encoding": "gzip",
                },
            )
        except httpx.HTTPError as e:
            return self._retry(f"backend unreachable: {e!r}")
        return self._handle(response, ids)

    def _handle(self, response: httpx.Response, ids: list[int]) -> Outcome:
        status = response.status_code
        if response.is_success:
            self.queue.ack(ids)
            self._backoff.reset()
            self.batch_size = min(self.max_batch, self.batch_size * 2)
            logger.debug("uploaded %d snapshot(s)", len(ids))
            return Outcome.SENT
        if status in {401, 403}:
            logger.error("backend rejected the API key (HTTP %d); check backend.api_key", status)
            return self._retry(f"HTTP {status}")
        if status == 413 and len(ids) > 1:  # noqa: PLR2004
            self.batch_size = max(1, len(ids) // 2)
            logger.warning("batch too large for the backend; reducing batch size to %d", self.batch_size)
            self.retry_after_s = 0.0
            return Outcome.RETRY
        if status == 429 or response.is_server_error:  # noqa: PLR2004
            return self._retry(f"HTTP {status}", _retry_after(response))
        self.queue.ack(ids)
        logger.error("backend rejected a batch of %d snapshot(s) with HTTP %d; dropped it", len(ids), status)
        return Outcome.DROPPED

    async def run(self, stop: asyncio.Event) -> None:
        """
        Upload every ``interval_s``, draining any backlog, until ``stop`` is set; then make one final attempt.

        :param stop: set to stop uploading
        """
        while not stop.is_set():
            outcome = await self.upload_once()
            if outcome is Outcome.RETRY:
                delay = self.retry_after_s
            elif outcome in {Outcome.SENT, Outcome.DROPPED} and len(self.queue) >= self.batch_size:
                delay = BACKLOG_PAUSE_S
            else:
                delay = self.interval_s
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(stop.wait(), timeout=delay)
        await self.upload_once()

    def _retry(self, reason: str, retry_after: float | None = None) -> Outcome:
        delay = self._backoff.next_delay()
        if retry_after is not None:
            delay = max(delay, retry_after)
        self.retry_after_s = delay
        logger.warning("upload failed (%s); %d snapshot(s) queued, retrying in %.1fs", reason, len(self.queue), delay)
        return Outcome.RETRY


def _retry_after(response: httpx.Response) -> float | None:
    value = response.headers.get("Retry-After")
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None
