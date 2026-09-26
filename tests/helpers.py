import gzip
import uuid
from datetime import UTC, datetime

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
