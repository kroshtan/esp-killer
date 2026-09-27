"""
The background worker: ``python -m server.worker``.

Every ``scoring_interval_s`` it scores new windows, scores them with the leakage model in shadow mode, queues
alerts for new flags and for flagged players who rejoined, delivers due alerts, exports new windows to the private
training dataset (the model and the export only when ``ESPK_DATA_URL`` is set), and once a day applies data
retention. It is a separate process from the API so none of this ever delays ingest; both use the same SQLite
database (WAL mode: one writer, concurrent readers).
Run exactly one worker per database.
"""

import asyncio
import logging
import signal
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from types import FrameType

import httpx

from server import __version__
from server.alerts.discord import install_log_redaction
from server.alerts.dispatch import DispatchSettings, deliver_due
from server.alerts.pipeline import detect_rejoins, enqueue_flag_alerts
from server.alerts.settings import SmtpSettings
from server.db.alerts import AlertRepository
from server.db.database import Database
from server.db.leakage import LeakageRepository
from server.db.scoring import ScoringRepository
from server.orgconfig import ConfigStore
from server.retention import RetentionPolicy, run_retention
from server.scoring.job import OrgRunResult, run_scoring
from server.settings import ServerSettings
from server.training.export import ExportResult, export_pending
from server.training.schema import SCORING_CONFIG, Pseudonymiser
from server.training.shadow import ShadowResult, ShadowScorer, scoring_config_json
from server.training.store import ObjectStore, open_store

logger = logging.getLogger("server.worker")

RETENTION_EVERY = timedelta(days=1)


@dataclass
class RunReport:
    scoring: list[OrgRunResult] = field(default_factory=list)
    queued: int = 0
    delivery: dict[str, int] = field(default_factory=dict)
    retention: dict[str, int] | None = None
    export: ExportResult | None = None
    shadow: ShadowResult | None = None


class Worker:
    def __init__(self, settings: ServerSettings, smtp: SmtpSettings | None = None) -> None:
        self.settings = settings
        self.smtp = smtp or SmtpSettings()
        self.config = ConfigStore(settings.config_path)
        self.db = Database(settings.database_path)
        self.scoring = ScoringRepository(self.db)
        self.alerts = AlertRepository(self.db)
        self.stop = threading.Event()
        self._last_retention: datetime | None = None
        self.training_store: ObjectStore | None = None
        self.pseudonymiser: Pseudonymiser | None = None
        self.shadow: ShadowScorer | None = None
        self._synced_scoring_config: bytes | None = None
        if settings.data_url:
            self.training_store = open_store(settings.data_url)
            self.shadow = ShadowScorer(self.training_store, LeakageRepository(self.db), self.scoring)
            if settings.export_key:
                self.pseudonymiser = Pseudonymiser(settings.export_key.get_secret_value())

    def run_once(self, now: datetime | None = None, http: httpx.AsyncClient | None = None) -> RunReport:
        """
        Run every job once.

        :param now: current time; defaults to the wall clock
        :param http: HTTP client for Discord; tests inject one, otherwise a fresh client is used
        :return: what happened
        """
        now = now or datetime.now(UTC)
        config = self.config.config
        s = self.settings
        report = RunReport()
        report.scoring = run_scoring(self.scoring, config, now, lag=timedelta(seconds=s.scoring_lag_s))
        # Before alerts, so a new flag's alert shows the model's opinion on the same windows. The model is a second
        # opinion: whatever goes wrong with it must not stop scoring, alerts or retention.
        if self.shadow is not None:
            try:
                report.shadow = self.shadow.run(config, now)
            except Exception:
                logger.exception("leakage model (shadow mode) failed; retrying next run")

        email_enabled = self.smtp.is_configured
        new_flags = [flag_id for r in report.scoring for flag_id in r.new_flag_ids]
        report.queued = len(
            enqueue_flag_alerts(
                self.alerts,
                config,
                new_flags,
                now,
                email_enabled=email_enabled,
                cooldown=timedelta(hours=s.alert_cooldown_h),
            )
        )
        report.queued += len(
            detect_rejoins(
                self.alerts, config, now, email_enabled=email_enabled, rejoin_gap=timedelta(seconds=s.rejoin_gap_s)
            )
        )
        report.delivery = asyncio.run(self._deliver(now, http))

        # Export before retention, so windows reach the training dataset before their positions are deleted.
        if self.training_store is not None and self.pseudonymiser is not None:
            report.export = export_pending(
                self.scoring, self.training_store, self.pseudonymiser, config.scoring_config, now=now
            )
        if self.training_store is not None:
            self._sync_scoring_config(self.training_store)

        if self._last_retention is None or now - self._last_retention >= RETENTION_EVERY:
            report.retention = run_retention(self.db, self.retention_policy(), now)
            self._last_retention = now
        return report

    def _sync_scoring_config(self, store: ObjectStore) -> None:
        """
        Write the scoring config to the store when it changed, for the trainer (which cannot read config.yaml).

        :param store: the private training store
        """
        data = scoring_config_json(self.config.config)
        if data == self._synced_scoring_config:
            return
        try:
            store.put(SCORING_CONFIG, data)
        except Exception:
            logger.exception("could not write the scoring config to the training store; retrying next run")
            return
        self._synced_scoring_config = data

    def retention_policy(self) -> RetentionPolicy:
        """
        Retention periods from the settings and the scoring horizon.

        :return: the policy
        """
        s = self.settings
        return RetentionPolicy(
            positions_days=s.retention_days,
            # Evidence must outlive the scoring horizon, or scores would silently lose their oldest windows.
            evidence_days=self.config.config.scoring_config.horizon_days + 1,
            scores_days=s.score_retention_days,
            alerts_days=s.alert_retention_days,
            flags_days=s.flag_retention_days,
        )

    async def _deliver(self, now: datetime, http: httpx.AsyncClient | None) -> dict[str, int]:
        settings = DispatchSettings(
            max_attempts=self.settings.alert_max_attempts, image_minutes=self.settings.alert_image_minutes
        )
        if http is not None:
            return await deliver_due(
                self.alerts, self.scoring, self.config.config, smtp=self.smtp, http=http, now=now, settings=settings
            )
        async with httpx.AsyncClient(timeout=30.0, headers={"User-Agent": f"espk/{__version__}"}) as client:
            return await deliver_due(
                self.alerts, self.scoring, self.config.config, smtp=self.smtp, http=client, now=now, settings=settings
            )

    def run_forever(self) -> None:
        """Run the jobs every ``scoring_interval_s`` until :attr:`stop` is set. A failing run is logged, not fatal."""
        while not self.stop.is_set():
            try:
                report = self.run_once()
                if report.queued or any(report.delivery.values()):
                    logger.info("alerts: %d queued, delivery %s", report.queued, report.delivery)
            except Exception:
                logger.exception("worker run failed")
            self.stop.wait(self.settings.scoring_interval_s)


def main() -> None:
    """Command-line entry point."""
    settings = ServerSettings()
    logging.basicConfig(level=settings.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # httpx logs request URLs, and a Discord webhook URL is a secret.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    install_log_redaction()
    worker = Worker(settings)

    def _stop(signum: int, _frame: FrameType | None) -> None:
        logger.info("received signal %d, stopping", signum)
        worker.stop.set()

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    logger.info(
        "worker started: every %.0fs; email alerts %s",
        settings.scoring_interval_s,
        "enabled" if worker.smtp.is_configured else "disabled (ESPK_SMTP_* not set)",
    )
    worker.run_forever()


if __name__ == "__main__":
    main()
