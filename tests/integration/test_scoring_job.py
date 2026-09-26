"""The scoring job end to end on the database: simulated positions in, windows, scores and flags out."""

import uuid
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import func, select
from typer.testing import CliRunner

from server.cli import app as cli_app
from server.db.engine import make_engine
from server.db.repository import Repository
from server.db.scoring import FALSE_POSITIVE, OPEN, ScoringRepository
from server.db.tables import evidence, player_scores
from server.orgconfig import OrgConfig, OrgEntry, ServerEntry, save_config
from server.scoring.combine import PlayerScore
from server.scoring.config import ScoringConfig
from server.scoring.job import apply_flags, run_scoring
from server.settings import ServerSettings
from server.worker import Worker
from shared.models import PlayerSample, Snapshot
from tools.sim.scenarios import archetypes, mixed_world, record

ORG = "sim-org"
START = datetime(2026, 1, 1, tzinfo=UTC)
HOURS = 4.5
LAG = timedelta(minutes=5)
CONFIG = OrgConfig(orgs={ORG: OrgEntry(servers={"s1": ServerEntry()})})


def ingest_frame(repo: Repository, frame: pd.DataFrame, server_id: str) -> None:
    """Store simulated samples (metres) as the agent would have uploaded them (game units)."""
    snapshots = [
        Snapshot(
            snapshot_id=uuid.uuid4(),
            captured_at=START + timedelta(seconds=float(group["t"].to_numpy(dtype=float)[0])),
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
        repo.ingest(ORG, server_id, snapshots[i : i + 2000], START)


@pytest.fixture(scope="module")
def database(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[str, dict[str, str]]]:
    url = f"sqlite:///{tmp_path_factory.mktemp('db') / 'espk.db'}"
    world = mixed_world(900)
    ingest_frame(Repository(make_engine(url)), record(world, HOURS * 3600), "s1")
    yield url, archetypes(world)


@pytest.fixture
def repo(database: tuple[str, dict[str, str]]) -> ScoringRepository:
    return ScoringRepository(make_engine(database[0]))


def test_job_processes_windows_scores_and_flags_cheaters(
    repo: ScoringRepository, database: tuple[str, dict[str, str]]
) -> None:
    arch = database[1]
    now = START + timedelta(hours=HOURS) + LAG
    (result,) = run_scoring(repo, CONFIG, now, lag=LAG)

    assert result.windows == 2  # 4.5 hours of data: two complete 2-hour windows
    assert result.scored == len(arch)
    assert repo.processed_until(ORG) == START + timedelta(hours=4)
    flagged = {f.player_id: f for f in repo.list_flags(ORG)}
    assert {arch[p] for p in flagged} <= {"beeline_cheater", "ambush_cheater", "subtle_cheater"}
    assert {"beeline_cheater", "ambush_cheater"} <= {arch[p] for p in flagged}
    flag = next(f for f in flagged.values() if arch[f.player_id] == "beeline_cheater")
    assert flag.status == OPEN
    assert flag.player_name == f"name {flag.player_id[-3:]}"
    assert flag.details["beeline"]["episodes"] > 0
    assert flag.details["servers"] == ["s1"]

    # Running again at the same time processes nothing, and flags are not duplicated.
    (again,) = run_scoring(repo, CONFIG, now, lag=LAG)
    assert again.windows == 0
    assert len(repo.list_flags(ORG)) == len(flagged)
    with repo.engine.connect() as conn:
        windows = conn.scalar(select(func.count(func.distinct(evidence.c.window_end))))
        score_rows = conn.scalar(select(func.count()).select_from(player_scores))
    assert windows == 2
    assert score_rows == len(arch)


def test_accumulated_evidence_matches_the_windows(repo: ScoringRepository) -> None:
    run_scoring(repo, CONFIG, START + timedelta(hours=HOURS) + LAG, lag=LAG)  # idempotent: a no-op if already run
    total = repo.load_evidence(ORG, since=START - timedelta(days=1))
    assert total.player_ids
    assert all(ev.moving_s > 0 for ev in total.beeline.values())
    assert repo.load_evidence(ORG, since=START + timedelta(hours=5)).player_ids == []


def _score(player: str, score: float) -> PlayerScore:
    return PlayerScore(player, score, None, None, None, servers=("s1",))


def test_flag_lifecycle(tmp_path: Path) -> None:
    repo = ScoringRepository(make_engine(f"sqlite:///{tmp_path / 'f.db'}"))
    cfg = ScoringConfig()
    now = START

    (first,) = apply_flags(repo, ORG, [_score("p", 0.7), _score("q", 0.3)], cfg, now)
    assert [f.player_id for f in repo.list_flags(ORG)] == ["p"]

    assert apply_flags(repo, ORG, [_score("p", 0.9)], cfg, now + timedelta(hours=1)) == []
    (flag,) = repo.list_flags(ORG)
    assert (flag.id, flag.score, flag.max_score) == (first, 0.9, 0.9)

    repo.mark_false_positive(first, now + timedelta(hours=2), note="streamer, verified")
    assert apply_flags(repo, ORG, [_score("p", 0.95)], cfg, now + timedelta(days=1)) == []
    later = now + timedelta(days=cfg.false_positive_suppress_days + 1)
    assert len(apply_flags(repo, ORG, [_score("p", 0.95)], cfg, later)) == 1
    assert [f.status for f in repo.list_flags(ORG)] == [OPEN, FALSE_POSITIVE]
    assert repo.list_flags(ORG, status=FALSE_POSITIVE)[0].note == "streamer, verified"
    assert repo.mark_false_positive(9999, now) is None


def test_cli_lists_flags_and_marks_false_positives(tmp_path: Path) -> None:
    url = f"sqlite:///{tmp_path / 'c.db'}"
    repo = ScoringRepository(make_engine(url))
    (flag_id,) = apply_flags(repo, ORG, [_score("76561198000000042", 0.8)], ScoringConfig(), START)
    runner = CliRunner()

    listed = runner.invoke(cli_app, ["list-flags", "--database-url", url])
    assert listed.exit_code == 0
    assert f"#{flag_id}" in listed.stdout
    assert "76561198000000042" in listed.stdout

    marked = runner.invoke(cli_app, ["mark-false-positive", str(flag_id), "--note", "x", "--database-url", url])
    assert marked.exit_code == 0
    assert "no flags" in runner.invoke(cli_app, ["list-flags", "--database-url", url]).stdout
    assert "false_positive" in runner.invoke(cli_app, ["list-flags", "--status", "all", "--database-url", url]).stdout
    assert runner.invoke(cli_app, ["mark-false-positive", "999", "--database-url", url]).exit_code == 1
    assert runner.invoke(cli_app, ["list-flags", "--status", "bogus", "--database-url", url]).exit_code == 1


def test_worker_runs_the_job_for_orgs_in_the_config(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    save_config(config_path, CONFIG)
    settings = ServerSettings(config_path=config_path, database_url=f"sqlite:///{tmp_path / 'w.db'}")
    worker = Worker(settings)
    (result,) = worker.run_once(START)
    assert (result.org_id, result.windows) == (ORG, 0)  # no data yet
