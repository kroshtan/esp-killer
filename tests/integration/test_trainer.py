import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

import trainer.train as train_module
from server.alerts.message import Alert, discord_payload, email_content
from server.db.database import Database
from server.db.leakage import LeakageRepository
from server.db.repository import Repository
from server.orgconfig import OrgConfig, OrgEntry, ServerEntry, save_config
from server.scoring.config import ScoringConfig
from server.settings import ServerSettings
from server.training.leakage import FeatureSpec, LeakageModel, Samples, current_version, promote
from server.training.schema import CURRENT_MODEL, SCORING_CONFIG
from server.training.store import LocalStore
from server.worker import Worker
from tests.helpers import ingest_frame
from tools.sim.scenarios import mixed_world, record
from trainer.__main__ import _scoring_config, app
from trainer.dataset import chunks, count_rows, load_positions
from trainer.devdata import LABELS, generate
from trainer.gate import Benchmark, GateThresholds, evaluate, run_benchmark
from trainer.pipeline import run_training
from trainer.train import TrainingData, TrainParams, auc, build_data, source_player, train

CONFIG = ScoringConfig()
PARAMS = TrainParams(folds=3, max_rounds=150, early_stopping_rounds=10, min_samples=60)
NOW = datetime(2026, 9, 2, tzinfo=UTC)  # the dev dataset starts on 2026-09-01
BENCH = Benchmark(kinds=("mixed",), seeds=(9001,), hours=2.0)
QUICK_BENCH = Benchmark(kinds=("mixed",), seeds=(9001,), hours=0.75)
ANYTHING = GateThresholds(synthetic_auc=0.0, sim_auc=0.0)
HONEST = ("roamer", "waterhole", "camper", "hunter", "group", "clan_member", "clan_hunter", "clan_spotter")


@pytest.fixture(scope="module")
def dev_store(tmp_path_factory: pytest.TempPathFactory) -> LocalStore:
    store = LocalStore(tmp_path_factory.mktemp("espk-data"))
    # Three beeline cheaters per server, so that there are enough of them to compare with honest players.
    generate(store, seeds=(11, 12, 13), hours=5.0, kinds=("mixed",), cheaters=("beeline_cheater",) * 3)
    return store


@pytest.fixture(scope="module")
def data(dev_store: LocalStore) -> TrainingData:
    frame = load_positions(dev_store)
    spec = FeatureSpec(classes=tuple(sorted(frame["dino_class"].dropna().unique())))
    return build_data(chunks(frame, CONFIG), spec, CONFIG, PARAMS)


@pytest.fixture(scope="module")
def trained(data: TrainingData) -> train_module.TrainResult:
    return train(data, PARAMS)


def test_devdata_is_in_the_export_format(dev_store: LocalStore) -> None:
    frame = load_positions(dev_store)
    assert count_rows(dev_store) == len(frame) > 0
    assert set(frame.columns) >= {"org", "server", "player_id", "t", "x", "y", "dino_class"}
    assert frame["server"].nunique() == 3
    assert not frame["player_id"].str.startswith("7656").any()  # pseudonymised
    assert frame["x"].abs().max() > 1000  # game units, not metres


def test_crossfitting_never_scores_a_player_with_a_model_that_saw_them(
    data: TrainingData, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[set[str]] = []
    scored: list[tuple[int, set[str]]] = []
    real_fit, real_predict = train_module.fit_pair, train_module.predict_terms

    def spy_fit(samples: Samples, *args: object, **kwargs: object) -> object:
        players = {source_player(p) for p in samples.player}
        synthetic = kwargs.get("synthetic")
        if isinstance(synthetic, Samples):
            players |= {source_player(p) for p in synthetic.player}
        if kwargs.get("rounds") is None:  # the final refit on everything is not used for measuring
            seen.append(players)
        return real_fit(samples, *args, **kwargs)  # type: ignore[arg-type]

    def spy_predict(a: object, b: object, samples: Samples, spec: FeatureSpec) -> np.ndarray:
        scored.append((len(seen) - 1, {source_player(p) for p in samples.player}))
        return real_predict(a, b, samples, spec)  # type: ignore[arg-type]

    monkeypatch.setattr(train_module, "fit_pair", spy_fit)
    monkeypatch.setattr(train_module, "predict_terms", spy_predict)
    train(_thinned(data, 4), TrainParams(folds=3, max_rounds=3, min_samples=60))
    assert len(seen) == 3
    assert all(players for _, players in scored)
    for fold, players in scored:
        assert not players & seen[fold]


def test_beeline_cheaters_leak_more_than_honest_players(
    dev_store: LocalStore, trained: train_module.TrainResult
) -> None:
    labels = json.loads(dev_store.get(LABELS))
    z = {p: s.z for p, s in trained.real_stats.items() if s.n >= PARAMS.min_samples}
    cheaters = [v for p, v in z.items() if labels[p] == "beeline_cheater"]
    honest = [v for p, v in z.items() if labels[p] in HONEST]
    assert len(cheaters) == 9 and len(honest) > 60
    assert auc(cheaters, honest) > 0.75
    assert np.mean(cheaters) > np.mean(honest) + 1.0
    assert trained.metrics["synthetic_auc"] > 0.6


def test_gate_accepts_a_good_model_and_rejects_one_that_learned_nothing(
    data: TrainingData, trained: train_module.TrainResult
) -> None:
    thresholds = GateThresholds(synthetic_auc=0.6, sim_auc=0.7)
    bench = run_benchmark(trained.model, BENCH, CONFIG, thresholds.flag_score)
    good = evaluate(trained.metrics, bench, thresholds)
    assert good.passed, good.failures
    assert bench["honest_flagged"] == 0

    # Break B: shuffle the hidden players' columns between samples, so they carry no information about the move.
    rng = np.random.default_rng(0)
    thin = _thinned(data, 2)
    broken = TrainingData(data.spec, _shuffled_hidden(thin.real, rng), _shuffled_hidden(thin.synthetic, rng))
    result = train(broken, TrainParams(folds=3, max_rounds=30, early_stopping_rounds=5, min_samples=60))
    assert result.metrics["synthetic_auc"] < 0.56
    bad = evaluate(result.metrics, bench, thresholds)
    assert not bad.passed
    assert any("synthetic ESP AUC" in f for f in bad.failures)
    # And against a promoted good model, a worse candidate is a regression.
    regression = evaluate(
        {**trained.metrics, "synthetic_auc": trained.metrics["synthetic_auc"] - 0.1},
        bench,
        thresholds,
        current_metrics=trained.metrics,
        current_benchmark=bench,
    )
    assert any("worse than the current model" in f for f in regression.failures)


def _thinned(data: TrainingData, every: int) -> TrainingData:
    """Every ``every``-th sample (all players keep some), for tests that do not need the model to be good."""
    return TrainingData(
        data.spec,
        data.real.subset(np.arange(0, len(data.real), every)),
        data.synthetic.subset(np.arange(0, len(data.synthetic), every)),
    )


def _shuffled_hidden(samples: Samples, rng: np.random.Generator) -> Samples:
    order = rng.permutation(len(samples))
    x = samples.x.copy()
    x[:, samples.n_a :] = x[order, samples.n_a :]
    return Samples(x, samples.y, samples.player, samples.t, samples.n_a, samples.x_null[:, order])


def test_model_store_round_trip(tmp_path: Path, trained: train_module.TrainResult, data: TrainingData) -> None:
    store = LocalStore(tmp_path)
    assert LeakageModel.load_current(store) is None
    trained.model.save(store, "v1")
    promote(store, "v1")
    assert current_version(store) == "v1"
    loaded = LeakageModel.load_current(store)
    assert loaded is not None
    assert loaded.spec == trained.model.spec and loaded.calibration == trained.model.calibration
    sample = data.real.subset(np.arange(200))
    np.testing.assert_allclose(loaded.terms(sample), trained.model.terms(sample))
    assert loaded.metadata["metrics"]["synthetic_auc"] == trained.metrics["synthetic_auc"]


def test_score_reports_every_player_with_a_summary(dev_store: LocalStore, trained: train_module.TrainResult) -> None:
    chunk = next(chunks(load_positions(dev_store), CONFIG))
    scores = trained.model.score([chunk.trajectories], CONFIG, count_from=chunk.count_from)
    assert set(scores) <= set(chunk.trajectories.player_ids) and len(scores) > 10
    one = next(iter(scores.values()))
    assert 0.0 <= one.score_0_1 <= 1.0
    assert "leakage z" in one.summary and f"{one.n} moves" in one.summary


def test_pipeline_promotes_then_skips_until_enough_new_rows(tmp_path: Path) -> None:
    store = LocalStore(tmp_path)
    generate(store, seeds=(21, 22), hours=1.0, kinds=("mixed",))
    params = TrainParams(folds=3, max_rounds=10, min_samples=60)
    # Mechanics only: the thresholds let any model through (quality is tested above).
    kwargs = {"params": params, "thresholds": ANYTHING, "bench": QUICK_BENCH, "now": NOW}

    first = run_training(store, **kwargs)  # type: ignore[arg-type]
    assert first.status == "promoted", first.reason
    assert store.exists(CURRENT_MODEL) and current_version(store) == first.version
    assert store.exists(f"models/{first.version}/model_a.txt")
    meta = json.loads(store.get(f"models/{first.version}/metadata.json"))
    assert meta["dataset_rows"] == count_rows(store)
    assert meta["gate"]["passed"] and "benchmark" in meta

    again = run_training(store, min_new_rows=1, **kwargs)  # type: ignore[arg-type]
    assert again.status == "skipped"

    generate(store, seeds=(24,), hours=0.5, kinds=("mixed",))
    later = datetime(2026, 9, 3, tzinfo=UTC)
    third = run_training(store, min_new_rows=1, **{**kwargs, "now": later})  # type: ignore[arg-type]
    assert third.status in ("promoted", "rejected")
    assert store.exists(f"models/candidates/{third.version}.json")


def test_rejected_candidate_keeps_the_current_model(tmp_path: Path) -> None:
    store = LocalStore(tmp_path)
    generate(store, seeds=(31, 32), hours=1.0, kinds=("mixed",))
    impossible = GateThresholds(synthetic_auc=1.01)
    params = TrainParams(folds=3, max_rounds=5, min_samples=60)
    outcome = run_training(store, params=params, thresholds=impossible, bench=QUICK_BENCH, now=NOW)
    assert outcome.status == "rejected" and "synthetic ESP AUC" in outcome.reason
    assert not store.exists(CURRENT_MODEL)
    assert not store.exists(f"models/{outcome.version}/model_a.txt")


def test_pipeline_skips_an_empty_store(tmp_path: Path) -> None:
    assert run_training(LocalStore(tmp_path), now=NOW).status == "skipped"


def test_cli_devdata_and_help(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        app, ["devdata", "--store", str(tmp_path), "--hours", "0.3", "--seeds", "5", "--kinds", "mixed"]
    )
    assert result.exit_code == 0, result.output
    assert count_rows(LocalStore(tmp_path)) > 0
    assert runner.invoke(app, ["train", "--help"]).exit_code == 0


# --- shadow mode in the backend ---


@pytest.fixture
def shadow_worker(tmp_path: Path, trained: train_module.TrainResult) -> tuple[Worker, LocalStore]:
    store = LocalStore(tmp_path / "store")
    trained.model.save(store, "v1")
    promote(store, "v1")
    save_config(tmp_path / "config.yaml", OrgConfig(orgs={"org": OrgEntry(servers={"s1": ServerEntry()})}))
    db = Database(tmp_path / "espk.db")
    ingest_frame(Repository(db), record(mixed_world(905), 4.5 * 3600), "org", "s1", SHADOW_START)
    settings = ServerSettings.model_validate(
        {"config_path": tmp_path / "config.yaml", "database_path": tmp_path / "espk.db", "data_url": str(store.root)}
    )
    return Worker(settings), store


SHADOW_START = datetime(2026, 1, 1, tzinfo=UTC)
SHADOW_NOW = SHADOW_START + timedelta(hours=4.5, minutes=10)


def test_worker_scores_windows_with_the_promoted_model(shadow_worker: tuple[Worker, LocalStore]) -> None:
    worker, store = shadow_worker
    report = worker.run_once(SHADOW_NOW, http=None)
    assert report.shadow is not None
    assert (report.shadow.model_version, report.shadow.windows) == ("v1", 2)
    assert report.shadow.scored > 10
    assert report.export is None  # no export key: the model runs, the export does not

    leakage = LeakageRepository(worker.db)
    player = next(p.player_id for p in mixed_world(905).players)
    shadow = leakage.latest("org", player)
    assert shadow is not None and shadow.model_version == "v1" and shadow.moves > 0
    assert "shadow" in shadow.line()

    # Nothing is scored twice; a newly promoted model scores the horizon's windows again.
    again = worker.run_once(SHADOW_NOW + timedelta(minutes=5), http=None)
    assert again.shadow is not None and again.shadow.windows == 0
    LeakageModel.load(store, "v1").save(store, "v2")
    promote(store, "v2")
    third = worker.run_once(SHADOW_NOW + timedelta(minutes=10), http=None)
    assert third.shadow is not None and (third.shadow.model_version, third.shadow.windows) == ("v2", 2)

    # The trainer reads the scoring config the worker wrote to the store.
    assert ScoringConfig.model_validate_json(store.get(SCORING_CONFIG)) == ScoringConfig()


def test_a_broken_model_never_stops_the_worker(shadow_worker: tuple[Worker, LocalStore]) -> None:
    worker, store = shadow_worker
    store.put(CURRENT_MODEL, json.dumps({"version": "missing"}).encode())
    report = worker.run_once(SHADOW_NOW, http=None)
    assert report.shadow is None
    assert report.scoring and report.scoring[0].windows == 2


def test_alerts_show_the_model_line_but_the_score_is_the_rules() -> None:
    alert = Alert(
        kind="flag",
        org_id="org",
        player_id="76561198000000001",
        player_name="Someone",
        server_ids=("s1",),
        score=0.7,
        details={},
        created_at=SHADOW_START,
        flag_id=1,
        model_line="Model (shadow, not part of the score): leakage z 3.1 over 5.0 h of movement (model v1)",
    )
    payload = discord_payload(alert)
    behaviours = next(f["value"] for f in payload["embeds"][0]["fields"] if f["name"] == "Behaviours")
    assert "leakage z 3.1" in behaviours
    assert "leakage z 3.1" in email_content(alert).text


def test_trainer_uses_the_scoring_config_from_the_store(tmp_path: Path) -> None:
    store = LocalStore(tmp_path)
    assert _scoring_config(store, None) == ScoringConfig()
    custom = ScoringConfig(awareness_m=250.0)
    store.put(SCORING_CONFIG, custom.model_dump_json().encode())
    assert _scoring_config(store, None) == custom
