"""
A development dataset from the simulator, in exactly the export format (``server/training/schema.py``).

It lets the trainer be developed and tested without a backend or real data: positions in game units, identifiers
pseudonymised with a dev key, one Parquet file per org per window in date partitions, and a window manifest each.
The simulator's hidden archetypes are written next to it (``dev/labels.json``, outside ``dataset/``) for looking at
results; training never reads them.
"""

import io
import json
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from server.training.schema import DATASET, POSITIONS, Pseudonymiser, partition, window_name
from server.training.store import ObjectStore
from trainer.synthetic import sim_frames

DEV_KEY = "espk-dev-key-not-secret"
DEV_ORG = "dev-org"
LABELS = "dev/labels.json"
BASE_TIME = datetime(2026, 9, 1, tzinfo=UTC)
UNITS_PER_METRE = 100.0


def generate(
    store: ObjectStore,
    *,
    seeds: Sequence[int],
    hours: float,
    kinds: Sequence[str] = ("mixed", "clan"),
    window_minutes: float = 120.0,
    key: str = DEV_KEY,
    cheaters: tuple[str, ...] | None = None,
) -> int:
    """
    Simulate one server per kind and seed and export it window by window.

    :param store: where to write
    :param seeds: rng seeds
    :param hours: simulated time per server
    :param kinds: world kinds (``mixed``, ``clan``)
    :param window_minutes: export window length
    :param key: pseudonymisation key
    :param cheaters: cheater archetypes per mixed world (see :func:`trainer.synthetic.sim_frames`)
    :return: number of position rows written
    """
    pseudo = Pseudonymiser(key)
    org = pseudo("org", DEV_ORG)
    labels: dict[str, str] = json.loads(store.get(LABELS)) if store.exists(LABELS) else {}
    rows = 0
    window = timedelta(minutes=window_minutes)
    for kind in kinds:
        for seed in seeds:
            frame, arch = sim_frames(kind, seed, hours, cheaters=cheaters)
            server_name = f"sim-{kind}-{seed}"
            player_map = {pid: pseudo("player", f"{server_name}:{pid}") for pid in arch}
            labels.update({player_map[pid]: a for pid, a in arch.items()})
            frame = frame.assign(
                org=org,
                server=pseudo("server", server_name),
                player=frame["player_id"].map(player_map),
                t=frame["t"] + BASE_TIME.timestamp(),
            )
            end = BASE_TIME + window
            while (end - window).timestamp() <= frame["t"].max():
                part = frame[(frame["t"] >= (end - window).timestamp()) & (frame["t"] < end.timestamp())]
                if len(part):
                    rows += _write_window(store, org, end, part, tag=f"{kind}{seed}")
                end += window
    store.put(LABELS, json.dumps(labels, indent=1, sort_keys=True).encode())
    return rows


def _write_window(store: ObjectStore, org: str, end: datetime, part: pd.DataFrame, *, tag: str) -> int:
    name = f"{window_name(org, end)}-{tag}"  # one org, many servers: keep their files apart
    table = pa.Table.from_pydict(
        {
            "org": part["org"].tolist(),
            "server": part["server"].tolist(),
            "player": part["player"].tolist(),
            "t": part["t"].to_numpy(float),
            "x": (part["x"].to_numpy() * UNITS_PER_METRE).astype(np.float32),
            "y": (part["y"].to_numpy() * UNITS_PER_METRE).astype(np.float32),
            "z": np.zeros(len(part), np.float32),
            "dino_class": pa.array(part["dino_class"].tolist()).dictionary_encode().cast(POSITIONS.field(7).type),
            "growth": np.ones(len(part), np.float32),
        },
        schema=POSITIONS,
    )
    buf = io.BytesIO()
    pq.write_table(table, buf)
    store.put(f"{DATASET}/positions/{partition(end)}/{name}.parquet", buf.getvalue())
    manifest = {"rows": len(part), "org": org, "window_end": end.isoformat(), "players": part["player"].nunique()}
    store.put(f"{DATASET}/windows/{name}.json", json.dumps(manifest).encode())
    return len(part)
