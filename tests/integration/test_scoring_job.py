"""The scoring job end to end on the database: simulated positions in, windows, scores and flags out."""

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from typer.testing import CliRunner

from server.cli import app as cli_app
from server.db.database import Database
from server.db.repository import Repository
from server.db.scoring import FALSE_POSITIVE, OPEN, ScoringRepository
from server.orgconfig import OrgConfig, OrgEntry, ServerEntry, save_config
from server.scoring.combine import PlayerScore
from server.scoring.config import ScoringConfig
from server.scoring.job import apply_flags, run_scoring
from server.settings import ServerSettings
from server.worker import Worker
from tests.helpers import ingest_frame
from tools.sim.scenarios import archetypes, mixed_world, record

ORG = "sim-org"
START = datetime(2026, 1, 1, tzinfo=UTC)
HOURS = 4.5
LAG = timedelta(minutes=5)
CONFIG = OrgConfig(orgs={ORG: OrgEntry(servers={"s1": ServerEntry()})})


@pytest.fixture(scope="module")
def database(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[Path, dict[str, str]]]:
    path = tmp_path_factory.mktemp("db") / "espk.db"
    world = mixed_world(900)
    ingest_frame(Repository(Database(path)), record(world, HOURS * 3600), ORG, "s1", START)
    yield path, archetypes(world)


@pytest.fixture
def repo(database: tuple[Path, dict[str, str]]) -> ScoringRepository:
    return ScoringRepository(Database(database[0]))


def test_job_processes_windows_scores_and_flags_cheaters(
    repo: ScoringRepository, database: tuple[Path, dict[str, str]]
) -> None:
    arch = database[1]
    now = START + timedelta(hours=HOURS) + LAG
    (result,) = run_scoring(repo, CONFIG, now, lag=LAG)

    assert result.windows == 2  # 4.5 hours of data: two complete 2-hour windows
    assert result.scored == len(arch)
    assert repo.processed_until(ORG) == START + timedelta(hours=4)
    flagged = {f.player_id: f for f in repo.list_flags(ORG)}
    # Mechanics, not detection power (tests/integration/test_scoring_separation.py covers that): with two
    # windows of data, only cheaters are flagged, and at least one is.
    assert flagged
    assert {arch[p] for p in flagged} <= {"beeline_cheater", "ambush_cheater", "subtle_cheater"}
    flag = next(iter(flagged.values()))
    assert flag.status == OPEN
    assert flag.player_name == f"name {flag.player_id[-3:]}"
    assert {"beeline", "ambush"} & set(flag.details)  # the behaviours behind the flag
    assert flag.details["servers"] == ["s1"]

    # Running again at the same time processes nothing, and flags are not duplicated.
    (again,) = run_scoring(repo, CONFIG, now, lag=LAG)
    assert again.windows == 0
    assert len(repo.list_flags(ORG)) == len(flagged)
    with repo.db.connect() as conn:
        windows = conn.execute("SELECT COUNT(DISTINCT window_end) FROM evidence").fetchone()[0]
        score_rows = conn.execute("SELECT COUNT(*) FROM player_scores").fetchone()[0]
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
    repo = ScoringRepository(Database(tmp_path / "f.db"))
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
    path = tmp_path / "c.db"
    repo = ScoringRepository(Database(path))
    (flag_id,) = apply_flags(repo, ORG, [_score("76561198000000042", 0.8)], ScoringConfig(), START)
    runner = CliRunner()

    listed = runner.invoke(cli_app, ["list-flags", "--database", str(path)])
    assert listed.exit_code == 0
    assert f"#{flag_id}" in listed.stdout
    assert "76561198000000042" in listed.stdout

    marked = runner.invoke(cli_app, ["mark-false-positive", str(flag_id), "--note", "x", "--database", str(path)])
    assert marked.exit_code == 0
    assert "no flags" in runner.invoke(cli_app, ["list-flags", "--database", str(path)]).stdout
    assert (
        "false_positive" in runner.invoke(cli_app, ["list-flags", "--status", "all", "--database", str(path)]).stdout
    )
    assert runner.invoke(cli_app, ["mark-false-positive", "999", "--database", str(path)]).exit_code == 1
    assert runner.invoke(cli_app, ["list-flags", "--status", "bogus", "--database", str(path)]).exit_code == 1


def test_worker_runs_the_job_for_orgs_in_the_config(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    save_config(config_path, CONFIG)
    settings = ServerSettings(config_path=config_path, database_path=tmp_path / "w.db")
    report = Worker(settings).run_once(START)
    assert [(r.org_id, r.windows) for r in report.scoring] == [(ORG, 0)]  # no data yet
    assert (report.queued, report.delivery) == (0, {"sent": 0, "retry": 0, "failed": 0})
