"""Export of scored windows to the private training dataset."""

import io
import json
import shutil
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
import pytest

from server.db.database import Database
from server.db.repository import Repository
from server.orgconfig import OrgConfig, OrgEntry, ServerEntry, save_config
from server.settings import ServerSettings
from server.training.schema import DATASET, POSITIONS, Pseudonymiser
from server.training.store import LocalStore, S3Store, open_store
from server.worker import Worker
from tests.helpers import ingest_frame
from tools.sim.scenarios import mixed_world, record

ORG = "sim-org"
START = datetime(2026, 1, 1, tzinfo=UTC)
KEY = "a-long-enough-test-export-key"


@pytest.fixture(scope="module")
def template(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("template") / "espk.db"
    db = Database(path)
    ingest_frame(Repository(db), record(mixed_world(900), 4.5 * 3600), ORG, "s1", START)
    with db.connect() as conn:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    return path


@pytest.fixture
def worker(tmp_path: Path, template: Path) -> Iterator[Worker]:
    save_config(tmp_path / "config.yaml", OrgConfig(orgs={ORG: OrgEntry(servers={"s1": ServerEntry()})}))
    shutil.copy(template, tmp_path / "espk.db")
    settings = ServerSettings.model_validate(
        {
            "config_path": tmp_path / "config.yaml",
            "database_path": tmp_path / "espk.db",
            "data_url": str(tmp_path / "dataset-store"),
            "export_key": KEY,
        }
    )
    yield Worker(settings)


def read(store: LocalStore, key: str) -> pd.DataFrame:
    frame: pd.DataFrame = pq.read_table(io.BytesIO(store.get(key))).to_pandas()
    return frame


def test_scored_windows_are_exported_pseudonymised(worker: Worker, tmp_path: Path) -> None:
    now = START + timedelta(hours=4.5, minutes=10)
    report = worker.run_once(now, http=None)
    assert report.export is not None
    assert (report.export.windows, report.export.failed) == (2, 0)  # two complete 2-hour windows

    store = LocalStore(tmp_path / "dataset-store")
    keys = store.list(DATASET)
    positions = [k for k in keys if "/positions/" in k]
    evidence = [k for k in keys if "/evidence/" in k]
    manifests = [k for k in keys if "/windows/" in k]
    assert len(positions) == len(evidence) == len(manifests) == 2
    assert all("date=2026-01-01" in k for k in positions)

    frame = read(store, positions[0])
    assert list(frame.columns) == POSITIONS.names
    # Each manifest counts its own window's rows, and together they are what the worker reported.
    for manifest_key in manifests:
        stem = manifest_key.rsplit("/", 1)[1].removesuffix(".json")
        (positions_key,) = [k for k in positions if k.endswith(f"/{stem}.parquet")]
        assert json.loads(store.get(manifest_key))["rows"] == len(read(store, positions_key))
    assert sum(json.loads(store.get(k))["rows"] for k in manifests) == report.export.rows
    pseudo = Pseudonymiser(KEY)
    real_ids = {p.player_id for p in mixed_world(900).players}
    assert set(frame["player"]) <= {pseudo("player", p) for p in real_ids}
    raw = b"".join(store.get(k) for k in keys)
    assert b"7656119" not in raw  # no Steam ids
    assert b"name " not in raw  # no player names (the simulator names players "name <n>")

    ev = read(store, evidence[0])
    assert set(ev["player"]) <= set(frame["player"])
    assert (ev["clan"] >= -1).all()

    # Nothing new: nothing is exported twice.
    again = worker.run_once(now + timedelta(minutes=5), http=None)
    assert again.export is not None
    assert again.export.windows == 0


def test_a_failing_store_is_retried_next_run(worker: Worker) -> None:
    class Broken:
        def put(self, key: str, data: bytes) -> None:
            raise OSError("bucket unreachable")

    good = worker.training_store
    worker.training_store = Broken()  # type: ignore[assignment]
    now = START + timedelta(hours=4.5, minutes=10)
    report = worker.run_once(now, http=None)
    assert report.export is not None
    assert (report.export.windows, report.export.failed) == (0, 2)
    worker.training_store = good
    later = worker.run_once(now + timedelta(minutes=5), http=None)
    assert later.export is not None
    assert later.export.windows == 2


def test_local_store(tmp_path: Path) -> None:
    store = open_store(str(tmp_path))
    assert isinstance(store, LocalStore)
    store.put("a/b.bin", b"x")
    store.put("a/c.bin", b"y")
    assert store.get("a/b.bin") == b"x"
    assert store.list("a/") == ["a/b.bin", "a/c.bin"]
    assert store.exists("a/b.bin")
    assert not store.exists("a/nope")
    with pytest.raises(KeyError):
        store.get("a/nope")
    with pytest.raises(ValueError, match="escapes"):
        store.put("../outside", b"z")


class FakeS3:
    """Just enough of a boto3 S3 client for S3Store."""

    class exceptions:  # noqa: N801 - mirrors boto3's attribute
        class NoSuchKey(Exception):  # noqa: N818 - boto3's name
            pass

        class ClientError(Exception):
            pass

    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}

    def put_object(self, Bucket: str, Key: str, Body: bytes) -> None:  # noqa: N803
        self.objects[(Bucket, Key)] = Body

    def get_object(self, Bucket: str, Key: str) -> dict[str, Any]:  # noqa: N803
        if (Bucket, Key) not in self.objects:
            raise self.exceptions.NoSuchKey
        return {"Body": io.BytesIO(self.objects[(Bucket, Key)])}

    def head_object(self, Bucket: str, Key: str) -> None:  # noqa: N803
        if (Bucket, Key) not in self.objects:
            raise self.exceptions.ClientError

    def get_paginator(self, _name: str) -> "FakeS3":
        return self

    def paginate(self, Bucket: str, Prefix: str) -> list[dict[str, Any]]:  # noqa: N803
        keys = sorted(k for b, k in self.objects if b == Bucket and k.startswith(Prefix))
        return [{"Contents": [{"Key": k} for k in keys]}]


def test_s3_store_with_a_prefix() -> None:
    client = FakeS3()
    store = S3Store("bucket", "/espk/", client=client)
    store.put("dataset/x", b"1")
    assert ("bucket", "espk/dataset/x") in client.objects
    assert store.get("dataset/x") == b"1"
    assert store.list("dataset/") == ["dataset/x"]
    assert store.exists("dataset/x")
    assert not store.exists("dataset/y")
    with pytest.raises(KeyError):
        store.get("dataset/y")
