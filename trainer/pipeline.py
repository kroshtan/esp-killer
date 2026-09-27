"""
One training run, start to finish: read the dataset, train, gate, and promote on success.

Meant to run from a scheduler (weekly cron) and exit. A rejected candidate is not an error: the current model stays,
and the candidate's metrics are kept under ``models/candidates/`` for whoever looks into why.
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import Any

import lightgbm as lgb

from server.scoring.config import ScoringConfig
from server.scoring.game import GameProfile, load_profile
from server.training.leakage import FeatureSpec, LeakageModel, promote
from server.training.schema import MODELS
from server.training.store import ObjectStore
from trainer.dataset import chunks, count_rows, load_positions
from trainer.gate import Benchmark, GateThresholds, evaluate, run_benchmark, summary
from trainer.train import TrainParams, build_data, train

logger = logging.getLogger(__name__)

CANDIDATES = f"{MODELS}/candidates"


@dataclass
class RunOutcome:
    status: str  # "promoted", "rejected" or "skipped"
    reason: str = ""
    version: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)


def new_version(now: datetime) -> str:
    """
    A model version name.

    :param now: the current time
    :return: e.g. ``20260927T140000Z``
    """
    return f"{now.astimezone(UTC):%Y%m%dT%H%M%SZ}"


def _load_current(store: ObjectStore) -> LeakageModel | None:
    """The promoted model, or None if there is none or it can no longer be read (e.g. an older feature layout)."""
    try:
        return LeakageModel.load_current(store)
    except (KeyError, ValueError, lgb.basic.LightGBMError) as e:  # pydantic's ValidationError is a ValueError
        logger.warning("cannot load the current model, comparing against none: %s", e)
        return None


def run_training(
    store: ObjectStore,
    *,
    config: ScoringConfig | None = None,
    params: TrainParams | None = None,
    thresholds: GateThresholds | None = None,
    bench: Benchmark | None = None,
    spec: FeatureSpec | None = None,
    profile: GameProfile | None = None,
    days: float | None = 28.0,
    min_new_rows: int = 0,
    now: datetime | None = None,
) -> RunOutcome:
    """
    Train a candidate on recent data, gate it, and promote it if it passes.

    :param store: the object store with the dataset
    :param config: scoring config (what counts as known, windows, null shifts); defaults if None
    :param params: training parameters
    :param thresholds: gate thresholds
    :param bench: simulator benchmark
    :param spec: feature spec (the class vocabulary is filled in from the data)
    :param profile: game profile for awareness ranges; the default profile if None
    :param days: train on the date partitions of the last this many days; None for all
    :param min_new_rows: skip unless at least this many position rows were exported since the promoted model was
        trained (0: always train)
    :param now: the current time
    :return: what happened
    """
    config = config or ScoringConfig()
    params = params or TrainParams()
    thresholds = thresholds or GateThresholds(flag_budget=params.flag_budget, flag_score=params.flag_score)
    bench = bench or Benchmark()
    now = now or datetime.now(UTC)
    rows = count_rows(store)
    current = _load_current(store)
    if current is not None and min_new_rows:
        new_rows = rows - int(current.metadata.get("dataset_rows", 0))
        if new_rows < min_new_rows:
            reason = f"{new_rows} new position rows since the current model, fewer than {min_new_rows}"
            logger.info("skipping: %s", reason)
            return RunOutcome("skipped", reason)
    since = (now - timedelta(days=days)).date() if days is not None else None
    frame = load_positions(store, since=since, until=now.date() if days is not None else None)
    if frame.empty:
        logger.info("skipping: no positions in the dataset")
        return RunOutcome("skipped", "no positions in the dataset")
    classes = tuple(sorted(str(c) for c in frame["dino_class"].dropna().unique()))
    spec = (spec or FeatureSpec()).model_copy(update={"classes": classes})
    logger.info("building samples from %d position rows", len(frame))
    data = build_data(chunks(frame, config, profile=profile or load_profile()), spec, config, params)
    del frame
    logger.info("training on %d samples (%d synthetic)", len(data.real), len(data.synthetic))
    result = train(data, params)
    model = result.model
    logger.info("benchmarking on simulated servers")
    benchmark = run_benchmark(model, bench, config, thresholds.flag_score)
    current_benchmark = run_benchmark(current, bench, config, thresholds.flag_score) if current else None
    gate = evaluate(
        result.metrics,
        benchmark,
        thresholds,
        current_metrics=current.metadata.get("metrics") if current else None,
        current_benchmark=current_benchmark,
    )
    for line in summary(gate, benchmark, result.metrics):
        logger.info(line)
    version = new_version(now)
    model.metadata.update(
        {
            "trained_at": now.isoformat(),
            "dataset_rows": rows,
            "data_since": since.isoformat() if isinstance(since, date) else None,
            "scoring_config": config.model_dump(mode="json"),
            "benchmark": benchmark,
            "benchmark_spec": {"kinds": list(bench.kinds), "seeds": list(bench.seeds), "hours": bench.hours},
            "gate": {"passed": gate.passed, "failures": gate.failures, "thresholds": asdict(thresholds)},
            "previous_version": current.metadata.get("version") if current else None,
            "previous_benchmark": current_benchmark,
        }
    )
    record = dict(model.metadata)
    store.put(f"{CANDIDATES}/{version}.json", json.dumps(record, indent=2, sort_keys=True, default=str).encode())
    if not gate.passed:
        return RunOutcome("rejected", "; ".join(gate.failures), version, record)
    model.save(store, version)
    promote(store, version)
    logger.info("promoted %s", version)
    return RunOutcome("promoted", "", version, record)
