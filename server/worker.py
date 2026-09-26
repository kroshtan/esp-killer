"""
The background worker: ``python -m server.worker``.

Runs the scoring job every ``scoring_interval_s``. It is a separate process from the API so a long scoring run
never delays ingest; both use the same database (SQLite in WAL mode allows one writer and concurrent readers).
Run exactly one worker per database.
"""

import logging
import signal
import threading
from datetime import UTC, datetime, timedelta
from types import FrameType

from server.db.database import Database
from server.db.scoring import ScoringRepository
from server.orgconfig import ConfigStore
from server.scoring.job import OrgRunResult, run_scoring
from server.settings import ServerSettings

logger = logging.getLogger("server.worker")


class Worker:
    def __init__(self, settings: ServerSettings, repo: ScoringRepository | None = None) -> None:
        self.settings = settings
        self.config = ConfigStore(settings.config_path)
        self.repo = repo or ScoringRepository(Database(settings.database_path))
        self.stop = threading.Event()

    def run_once(self, now: datetime | None = None) -> list[OrgRunResult]:
        """
        Run every job once.

        :param now: current time; defaults to the wall clock
        :return: scoring results per org
        """
        now = now or datetime.now(UTC)
        return run_scoring(self.repo, self.config.config, now, lag=timedelta(seconds=self.settings.scoring_lag_s))

    def run_forever(self) -> None:
        """Run the jobs every ``scoring_interval_s`` until :attr:`stop` is set. A failing run is logged, not fatal."""
        while not self.stop.is_set():
            try:
                self.run_once()
            except Exception:
                logger.exception("scoring run failed")
            self.stop.wait(self.settings.scoring_interval_s)


def main() -> None:
    """Command-line entry point."""
    settings = ServerSettings()
    logging.basicConfig(level=settings.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    worker = Worker(settings)

    def _stop(signum: int, _frame: FrameType | None) -> None:
        logger.info("received signal %d, stopping", signum)
        worker.stop.set()

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    logger.info("worker started: scoring every %.0fs", settings.scoring_interval_s)
    worker.run_forever()


if __name__ == "__main__":
    main()
