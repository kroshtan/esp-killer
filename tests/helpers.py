import gzip
import uuid
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

from server.db.repository import Repository
from shared.models import IngestBatch, PlayerSample, Snapshot


def sample(player_id: str = "76561198000000001", x: float = 0.0, y: float = 0.0, **kw: object) -> PlayerSample:
    return PlayerSample.model_validate(
        {"player_id": player_id, "player_name": f"name-{player_id}", "x": x, "y": y, "z": 0.0, **kw}
    )


def snapshot(players: list[PlayerSample] | None = None, captured_at: datetime | None = None) -> Snapshot:
    return Snapshot(
        snapshot_id=uuid.uuid4(),
        captured_at=captured_at or datetime.now(UTC),
        players=players if players is not None else [sample()],
    )


def batch_body(*snapshots: Snapshot, compress: bool = True) -> bytes:
    raw = IngestBatch(agent_version="test", snapshots=list(snapshots)).model_dump_json().encode()
    return gzip.compress(raw) if compress else raw


def auth(key: str, *, gzip_body: bool = True) -> dict[str, str]:
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    if gzip_body:
        headers["Content-Encoding"] = "gzip"
    return headers


def ingest_frame(repo: Repository, frame: pd.DataFrame, org_id: str, server_id: str, start: datetime) -> None:
    """Store simulated samples (metres, t in seconds from ``start``) as the agent would upload them (game units)."""
    snapshots = [
        Snapshot(
            snapshot_id=uuid.uuid4(),
            captured_at=start + timedelta(seconds=float(group["t"].to_numpy(dtype=float)[0])),
            players=[
                PlayerSample(
                    player_id=pid, player_name=f"name {pid[-3:]}", dino_class=cls, x=x * 100, y=y * 100, z=0.0
                )
                for pid, cls, x, y in zip(group["player_id"], group["dino_class"], group["x"], group["y"], strict=True)
            ],
        )
        for _, group in frame.groupby("t")
    ]
    for i in range(0, len(snapshots), 2000):
        repo.ingest(org_id, server_id, snapshots[i : i + 2000], start)


DT = 5.0


def track(player_id: str, points: list[tuple[float, float, float]], dino_class: str = "Troodon") -> pd.DataFrame:
    """Samples every DT seconds along straight segments between (t, x, y) waypoints."""
    ts = np.arange(points[0][0], points[-1][0] + 1e-9, DT)
    xs = np.interp(ts, [p[0] for p in points], [p[1] for p in points])
    ys = np.interp(ts, [p[0] for p in points], [p[2] for p in points])
    return pd.DataFrame({"t": ts, "player_id": player_id, "x": xs, "y": ys, "dino_class": dino_class})


def killed(frame: pd.DataFrame, at: float) -> pd.DataFrame:
    """A track that ends at ``at``: the player was killed (and is gone for the rest of the test)."""
    return frame[frame["t"] <= at]
