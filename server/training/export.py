"""
Export scored windows to the private training dataset (see ``schema.py`` for the layout).

The worker calls :func:`export_pending` after each scoring run. Every scoring window that has not been exported
yet is written as two Parquet files (positions, per-player evidence) plus a small JSON manifest the trainer uses to
count new data. Identifiers are pseudonymised and player names are left out.

Export is best-effort and never blocks scoring: a failure (the bucket is down) is logged and the window is retried
on the next run, as long as its positions have not been deleted by retention yet.
"""

import io
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from server.db.scoring import ScoringRepository
from server.scoring.config import ScoringConfig
from server.scoring.teams import infer_teams
from server.training.schema import DATASET, EVIDENCE, POSITIONS, Pseudonymiser, partition, window_name
from server.training.store import ObjectStore

logger = logging.getLogger(__name__)

MAX_WINDOWS_PER_RUN = 24


@dataclass(frozen=True)
class ExportResult:
    windows: int = 0
    rows: int = 0
    failed: int = 0


def export_pending(
    repo: ScoringRepository,
    store: ObjectStore,
    pseudo: Pseudonymiser,
    config: ScoringConfig,
    *,
    now: datetime,
    limit: int = MAX_WINDOWS_PER_RUN,
) -> ExportResult:
    """
    Export windows that were scored but not exported yet.

    :param repo: scoring repository
    :param store: the private dataset store
    :param pseudo: pseudonymiser (keyed with ``ESPK_EXPORT_KEY``)
    :param config: scoring config (window length, horizon)
    :param now: current time
    :param limit: at most this many windows per call
    :return: what was exported
    """
    windows = rows = failed = 0
    for org_id, window_end in repo.unexported_windows(limit):
        try:
            n = export_window(repo, store, pseudo, config, org_id=org_id, window_end=window_end)
        except Exception:
            logger.exception("training export of %s window ending %s failed; will retry", org_id, window_end)
            failed += 1
            continue
        repo.mark_exported(org_id, window_end, n, now)
        windows += 1
        rows += n
    if windows or failed:
        logger.info("training export: %d window(s), %d position row(s), %d failed", windows, rows, failed)
    return ExportResult(windows, rows, failed)


def export_window(
    repo: ScoringRepository,
    store: ObjectStore,
    pseudo: Pseudonymiser,
    config: ScoringConfig,
    *,
    org_id: str,
    window_end: datetime,
) -> int:
    """
    Export one window: positions, per-player evidence (with inferred clans) and a manifest.

    :param repo: scoring repository
    :param store: the private dataset store
    :param pseudo: pseudonymiser
    :param config: scoring config
    :param org_id: the org
    :param window_end: end of the window
    :return: position rows written
    """
    start = window_end - timedelta(minutes=config.window_minutes)
    org = pseudo("org", org_id)
    name = window_name(org, window_end)
    part = partition(window_end)

    frame = repo.load_positions(org_id, start, window_end)
    positions = pd.DataFrame(
        {
            "org": org,
            "server": frame["server_id"].map(lambda s: pseudo("server", s)),
            "player": frame["player_id"].map(lambda p: pseudo("player", p)),
            "t": (pd.to_datetime(frame["server_ts"], utc=True) - pd.Timestamp(0, tz="UTC")).dt.total_seconds(),
            "x": frame["x"],
            "y": frame["y"],
            "z": frame["z"],
            "dino_class": frame["dino_class"],
            "growth": frame["growth"],
        }
    )
    store.put(f"{DATASET}/positions/{part}/{name}.parquet", _parquet(positions, POSITIONS))

    clans = infer_teams(
        repo.load_pair_evidence(org_id, since=window_end - timedelta(days=config.horizon_days)), config
    )
    ev = repo.window_rows(org_id, window_end)
    evidence = pd.DataFrame(
        {
            "org": org,
            "player": ev["player_id"].map(lambda p: pseudo("player", p)),
            "window_end": window_end.timestamp(),
            "moving_s": ev["moving_s"],
            "beeline_episodes": ev["beeline_episodes"],
            "beeline_null": ev["beeline_null"],
            "ambush_waits": ev["ambush_waits"],
            "ambush_hits": ev["ambush_hits"],
            "ambush_null": ev["ambush_null"],
            "spawn_episodes": ev["spawn_episodes"],
            "clan": ev["player_id"].map(lambda p: clans.get(p, -1)),
        }
    )
    store.put(f"{DATASET}/evidence/{part}/{name}.parquet", _parquet(evidence, EVIDENCE))

    manifest = {
        "org": org,
        "window_end": window_end.timestamp(),
        "rows": len(positions),
        "players": int(positions["player"].nunique()),
        "servers": int(positions["server"].nunique()),
    }
    store.put(f"{DATASET}/windows/{name}.json", json.dumps(manifest).encode("utf-8"))
    return len(positions)


def _parquet(frame: pd.DataFrame, schema: pa.Schema) -> bytes:
    table = pa.Table.from_pandas(frame, schema=schema, preserve_index=False)
    buffer = io.BytesIO()
    pq.write_table(table, buffer, compression="zstd")
    return buffer.getvalue()
