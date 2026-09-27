"""
Reading the exported dataset (``server/training/schema.py``) back into per-server trajectory windows.

The export is one Parquet file per org per scoring window, in date partitions. Training cuts it the same way the
backend scores it: per server, into windows of ``window_minutes`` with some leading context, so features look
exactly like they will at inference time and memory stays bounded by one window at a time.
"""

import io
import json
import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from server.scoring.config import ScoringConfig
from server.scoring.game import GameProfile, load_profile
from server.scoring.trajectories import Trajectories, from_frame
from server.training.schema import DATASET
from server.training.store import ObjectStore

logger = logging.getLogger(__name__)

POSITIONS_PREFIX = f"{DATASET}/positions/"
WINDOWS_PREFIX = f"{DATASET}/windows/"
_PARTITION = re.compile(r"/date=(\d{4}-\d{2}-\d{2})/")
# Leading context per window: enough for heading lags, "seen recently" (10 min) and time since spawn.
DEFAULT_CONTEXT_S = 900.0


@dataclass(frozen=True)
class Chunk:
    """One server's trajectories for one window; samples count from ``count_from``."""

    org: str
    trajectories: Trajectories
    count_from: float


def count_rows(store: ObjectStore) -> int:
    """
    Total position rows exported so far, from the window manifests (cheap: no Parquet is read).

    :param store: the object store
    :return: the sum of ``rows`` over ``dataset/v1/windows/*.json``
    """
    total = 0
    for key in store.list(WINDOWS_PREFIX):
        if key.endswith(".json"):
            total += int(json.loads(store.get(key)).get("rows", 0))
    return total


def position_keys(store: ObjectStore, since: date | None = None, until: date | None = None) -> list[str]:
    """
    Positions files in the date partitions ``since`` .. ``until`` (inclusive).

    :param store: the object store
    :param since: first date, or None for all
    :param until: last date, or None for all
    :return: object keys, sorted
    """
    keys = []
    for key in store.list(POSITIONS_PREFIX):
        match = _PARTITION.search(key)
        if not key.endswith(".parquet") or match is None:
            continue
        day = date.fromisoformat(match.group(1))
        if (since is None or day >= since) and (until is None or day <= until):
            keys.append(key)
    return keys


def load_positions(store: ObjectStore, since: date | None = None, until: date | None = None) -> pd.DataFrame:
    """
    Read positions into one frame, converted to what the scoring code expects (seconds, player ids).

    :param store: the object store
    :param since: first date partition, or None for all
    :param until: last date partition, or None for all
    :return: columns ``org``, ``server``, ``player_id``, ``t``, ``x``, ``y`` (game units) and ``dino_class``
    """
    columns = ["org", "server", "player", "t", "x", "y", "dino_class"]
    frames = [
        pq.read_table(io.BytesIO(store.get(key)), columns=columns).to_pandas()
        for key in position_keys(store, since, until)
    ]
    if not frames:
        return pd.DataFrame({c: pd.Series(dtype=object) for c in ("org", "server", "player_id", "dino_class")})
    frame = pd.concat(frames, ignore_index=True).rename(columns={"player": "player_id"})
    frame["dino_class"] = frame["dino_class"].astype(object).where(frame["dino_class"].notna(), None)
    # A window's rows can appear in two exports if it was re-exported; keep one.
    deduplicated: pd.DataFrame = frame.drop_duplicates(["server", "player_id", "t"]).reset_index(drop=True)
    return deduplicated


def chunks(
    frame: pd.DataFrame,
    config: ScoringConfig,
    *,
    profile: GameProfile | None = None,
    context_s: float = DEFAULT_CONTEXT_S,
) -> Iterator[Chunk]:
    """
    Cut positions into per-server windows (aligned to multiples of ``window_minutes``), each with leading context.

    :param frame: output of :func:`load_positions`
    :param config: scoring config (units, grid, window length)
    :param profile: game profile for awareness ranges; the default profile if None
    :param context_s: leading context per window, in seconds
    :yield: one chunk per server and window with data
    """
    if frame.empty:
        return
    profile = profile or load_profile()
    window = config.window_minutes * 60
    for (org, server), group in frame.groupby(["org", "server"], sort=True):
        samples = pd.DataFrame(
            {
                "t": group["t"].to_numpy(float),
                "player_id": group["player_id"].to_numpy(),
                "x": group["x"].to_numpy(float) / config.units_per_metre,
                "y": group["y"].to_numpy(float) / config.units_per_metre,
                "dino_class": group["dino_class"].to_numpy(),
            }
        ).sort_values("t")
        t = samples["t"].to_numpy()
        for start in np.arange(np.floor(t[0] / window) * window, t[-1], window):
            lo, hi = np.searchsorted(t, [start - context_s, start + window])
            part = samples.iloc[lo:hi]
            if (part["t"] >= start).sum() == 0:
                continue
            yield Chunk(str(org), from_frame(part, str(server), config, profile), float(start))
