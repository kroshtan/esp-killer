"""
The leakage model in shadow mode: scored and shown, never used to flag.

The worker calls :meth:`ShadowScorer.run` after each scoring run. It follows the promoted model in the private
store (``models/current.json``), scores every scoring window within the evidence horizon that the current model
has not scored yet, and updates the latest score of the players in those windows. The score appears in alerts and
in ``list-flags`` next to the rule-based score, so admins can compare the two on real cases before the model is
given any say.

Failures here (the bucket is down, a bad model) are logged by the worker and never affect scoring or alerts.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

from server.db.leakage import LeakageRepository
from server.db.scoring import ScoringRepository
from server.orgconfig import OrgConfig
from server.scoring.job import to_trajectories
from server.scoring.teams import infer_teams
from server.training.leakage import LeakageModel, current_version
from server.training.store import ObjectStore

logger = logging.getLogger(__name__)

# Scoring a window takes seconds for a busy server; a backlog (a newly promoted model) is worked off over several
# runs rather than one very long one.
MAX_WINDOWS_PER_RUN = 6


@dataclass(frozen=True)
class ShadowResult:
    model_version: str | None = None
    windows: int = 0
    scored: int = 0


class ShadowScorer:
    def __init__(self, store: ObjectStore, repo: LeakageRepository, scoring: ScoringRepository) -> None:
        self.store = store
        self.repo = repo
        self.scoring = scoring
        self.model: LeakageModel | None = None
        self.version: str | None = None

    def refresh(self) -> None:
        """Load the promoted model if it changed since the last call."""
        version = current_version(self.store)
        if version is None or version == self.version:
            return
        self.model = LeakageModel.load(self.store, version)
        self.version = version
        logger.info("leakage model %s loaded (shadow mode)", version)

    def run(self, config: OrgConfig, now: datetime, limit: int = MAX_WINDOWS_PER_RUN) -> ShadowResult:
        """
        Score pending windows with the current model and update the scores of the players in them.

        :param config: org config (scoring thresholds, game profiles)
        :param now: current time
        :param limit: at most this many windows
        :return: what was done
        """
        self.refresh()
        if self.model is None or self.version is None:
            return ShadowResult()
        model, version = self.model, self.version
        cfg = config.scoring_config
        horizon = timedelta(days=cfg.horizon_days)
        window = timedelta(minutes=cfg.window_minutes)
        touched: dict[str, set[str]] = {}
        pending = self.repo.pending_windows(version, since=now - horizon, limit=limit)
        for org_id, end in pending:
            start = end - window
            frame = self.scoring.load_positions(org_id, start - timedelta(minutes=cfg.context_minutes), end)
            trajectories = to_trajectories(frame, cfg, config.game_profiles(org_id))
            teams = infer_teams(self.scoring.load_pair_evidence(org_id, since=end - horizon), cfg)
            stats = model.player_stats(trajectories, cfg, count_from=start.timestamp(), teams=teams)
            self.repo.save_window(org_id, version, end, stats, now)
            touched.setdefault(org_id, set()).update(stats)

        scored = 0
        for org_id, players in touched.items():
            scored += self._rescore(model, version, org_id, players, now - horizon, now)
        if pending:
            logger.info("leakage model %s: %d window(s), %d player(s) rescored", version, len(pending), scored)
        return ShadowResult(version, len(pending), scored)

    def _rescore(
        self, model: LeakageModel, version: str, org_id: str, players: set[str], since: datetime, now: datetime
    ) -> int:
        totals = self.repo.totals(org_id, version, since, players)
        scores = {pid: model.rescore(stats) for pid, stats in totals.items()}
        self.repo.save_scores(org_id, version, now, scores, model.spec.stride_s)
        return len(scores)


def scoring_config_json(config: OrgConfig) -> bytes:
    """
    The scoring config as the trainer reads it from the store (``SCORING_CONFIG``).

    :param config: org config
    :return: JSON
    """
    return config.scoring_config.model_dump_json(indent=2).encode()
