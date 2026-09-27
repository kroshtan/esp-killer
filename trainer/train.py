"""
Training the leakage models without labels, and measuring them honestly.

Every number that decides anything is **out of sample**: players are split into folds by a hash of their id (the same
player lands in the same fold on every server and every run), and each fold's terms come from models A and B trained
on the other folds only. The model the backend uses is refit on everything afterwards, with the number of boosting
rounds the folds found. A player is therefore never measured by a model that has seen that player, which matters: B
has more to go on than A and would "explain" a memorised track better, for any player.

B's residual may also be shown synthetic ESP users (``synthetic.py``: real tracks with hunting phases injected) from
the training folds, so that it learns what following hidden players looks like even if the real data has few ESP
users. That is safe because the statistic is measured against the time-shifted null (see
``server/training/leakage.py``): for a player whose moves do not depend on where hidden players are now, real and
phantom players are interchangeable, whatever B learned. B's training only decides how much power the test has.

Calibration maps a player's z to the 0-1 sub-score so that, on the out-of-sample real data, at most ``flag_budget``
of players with enough samples reach the flag level. Real data may contain cheaters, so this caps the flag rate
rather than promising that everyone above it is honest; the floor on the flag z (``z_flag_min``) keeps a model
trained on clean data from flagging the noisiest honest players just to fill the budget.

Recall is measured on synthetic ESP users from held-out folds.
"""

import logging
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import lightgbm as lgb
import numpy as np

from server.scoring.config import ScoringConfig
from server.training.leakage import (
    Calibration,
    FeatureSpec,
    LeakageModel,
    LeakageStats,
    Samples,
    aggregate,
    build_samples,
    candidate_rows,
    feature_names,
    model_size_bytes,
    predict_terms,
    row_names,
    stable_fold,
)
from trainer.dataset import Chunk
from trainer.synthetic import ESP_SUFFIX, Injection, inject_pursuits

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainParams:
    folds: int = 3
    max_rounds: int = 300
    early_stopping_rounds: int = 25
    learning_rate: float = 0.1
    num_leaves: int = 31
    min_data_in_leaf: int = 100
    max_train_samples: int = 1_000_000  # per fit; random samples beyond this are left out of fitting (not scoring)
    validation_share: int = 7  # one in this many training players is held out for early stopping
    threads: int = 0  # 0: all cores
    seed: int = 0
    # Calibration: at most this share of players with at least ``min_samples`` may reach ``flag_score``...
    flag_budget: float = 0.005
    flag_score: float = 0.6
    min_samples: int = 120
    # ...and reaching it takes at least this z. The sub-score ramps over ``ramp_width`` of z.
    z_flag_min: float = 5.0
    ramp_width: float = 4.0
    # One in this many players also gets a synthetic ESP copy (1: everyone), to measure recall on held-out folds...
    synthetic_share: int = 1
    # ...and, from the training folds, to show B's residual what following hidden players looks like. Without it B
    # learns only from the ESP users in the data, which is too little on a small or clean dataset.
    augment: bool = True
    injection: Injection = field(default_factory=Injection)


@dataclass
class TrainingData:
    spec: FeatureSpec
    real: Samples
    synthetic: Samples  # player ids end in ``ESP_SUFFIX``


@dataclass
class TrainResult:
    model: LeakageModel
    metrics: dict[str, Any]
    real_stats: dict[str, LeakageStats]  # cross-fitted, per player
    synthetic_stats: dict[str, LeakageStats]
    terms: np.ndarray  # cross-fitted per-sample terms of the real samples


def is_synthetic_source(player_id: str, params: TrainParams) -> bool:
    """
    Whether a player gets a synthetic ESP copy.

    :param player_id: player id
    :param params: training parameters
    :return: True for about one in ``synthetic_share`` players
    """
    return stable_fold(player_id, params.synthetic_share, salt="synthetic") == 0


def build_data(chunks: Iterable[Chunk], spec: FeatureSpec, config: ScoringConfig, params: TrainParams) -> TrainingData:
    """
    Samples from every chunk, plus synthetic ESP samples for a stable subset of players.

    :param chunks: trajectory windows
    :param spec: feature spec (with the class vocabulary)
    :param config: scoring config
    :param params: training parameters
    :return: the samples
    """
    names_a, names_b = feature_names(spec)
    shape = (len(names_a), len(names_b), len(config.null_shifts_s))
    rng = np.random.default_rng(params.seed)
    real, synthetic = [], []
    for chunk in chunks:
        tr = chunk.trajectories
        real.append(build_samples(tr, spec, config, count_from=chunk.count_from))
        chosen = [i for i, pid in enumerate(tr.player_ids) if is_synthetic_source(pid, params)]
        if chosen:
            injected = inject_pursuits(tr, chosen, rng, spec, config, injection=params.injection)
            s = build_samples(injected, spec, config, count_from=chunk.count_from, players=chosen)
            s.player = np.array([f"{p}{ESP_SUFFIX}" for p in s.player], dtype=object)
            synthetic.append(s)
    return TrainingData(spec, Samples.concat(real, *shape), Samples.concat(synthetic, *shape))


def source_player(player_id: str) -> str:
    """
    The real player a (possibly synthetic) player id was made from.

    :param player_id: player id
    :return: the id without ``ESP_SUFFIX``
    """
    return player_id.removesuffix(ESP_SUFFIX)


def folds_of(players: np.ndarray, folds: int) -> np.ndarray:
    """
    Cross-fitting fold of every sample, by (source) player.

    :param players: (n,) player ids
    :param folds: number of folds
    :return: (n,) fold numbers
    """
    unique, inverse = np.unique(players.astype(str), return_inverse=True)
    per_player = np.array([stable_fold(source_player(p), folds) for p in unique], dtype=int)
    return per_player[inverse] if len(unique) else np.empty(0, int)


def _lgb_params(params: TrainParams) -> dict[str, Any]:
    return {
        "objective": "binary",
        "learning_rate": params.learning_rate,
        "num_leaves": params.num_leaves,
        "min_data_in_leaf": params.min_data_in_leaf,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "max_bin": 63,
        "num_threads": params.threads,
        "seed": params.seed,
        "deterministic": True,
        "verbose": -1,
    }


def _labels(y: np.ndarray, spec: FeatureSpec) -> np.ndarray:
    """For each candidate row: whether it was the move taken."""
    return np.asarray((np.arange(spec.n_classes)[None, :] == y[:, None]).ravel(), dtype=np.float32)


def _dataset(
    rows: np.ndarray,
    labels: np.ndarray,
    init_score: np.ndarray | None,
    names: list[str],
    reference: lgb.Dataset | None = None,
) -> lgb.Dataset:
    categorical = [names.index("dino_class")] if "dino_class" in names else []
    return lgb.Dataset(
        rows,
        labels,
        feature_name=names,
        categorical_feature=categorical,
        init_score=init_score,
        reference=reference,
        free_raw_data=True,
    )


def _fit(
    train: tuple[np.ndarray, np.ndarray, np.ndarray | None],
    valid: tuple[np.ndarray, np.ndarray, np.ndarray | None] | None,
    names: list[str],
    params: TrainParams,
    rounds: int,
) -> lgb.Booster:
    train_set = _dataset(*train, names)
    callbacks: list[Any] = []
    valid_sets = []
    if valid is not None and len(valid[1]):
        valid_sets = [_dataset(*valid, names, reference=train_set)]
        callbacks.append(lgb.early_stopping(params.early_stopping_rounds, verbose=False))
    return lgb.train(
        _lgb_params(params), train_set, num_boost_round=rounds, valid_sets=valid_sets, callbacks=callbacks
    )


def _split(samples: Samples, params: TrainParams, validate: bool) -> list[Samples]:
    """Training and (if ``validate``) player-grouped validation parts."""
    if not validate:
        return [samples]
    is_valid = np.array(
        [stable_fold(source_player(str(p)), params.validation_share, salt="valid") == 0 for p in samples.player],
        dtype=bool,
    )
    return [samples.subset(~is_valid), samples.subset(is_valid)]


def _cap(samples: Samples, limit: int, rng: np.random.Generator) -> Samples:
    if len(samples) <= limit:
        return samples
    return samples.subset(np.sort(rng.choice(len(samples), limit, replace=False)))


def _rows(parts: Sequence[Samples], spec: FeatureSpec) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Per part: (A's candidate rows, B's residual rows, labels)."""
    return [(*candidate_rows(part, spec), _labels(part.y, spec)) for part in parts]


def fit_pair(
    samples: Samples,
    spec: FeatureSpec,
    params: TrainParams,
    *,
    synthetic: Samples | None = None,
    rounds: tuple[int, int] | None = None,
) -> tuple[lgb.Booster, lgb.Booster]:
    """
    Train model A, then B's residual on top of A's scores (B = A + residual on hidden players' columns).

    :param samples: training samples (real)
    :param spec: feature spec
    :param params: training parameters
    :param synthetic: synthetic ESP samples added to B's residual training (never to A's)
    :param rounds: fixed boosting rounds for (A, B), without early stopping; None to early-stop on a player-grouped
        validation split
    :return: (booster A, B's residual booster)
    """
    names_a, names_b = row_names(spec)
    rng = np.random.default_rng(params.seed)
    parts = _split(_cap(samples, params.max_train_samples, rng), params, rounds is None)
    rows = _rows(parts, spec)
    a = _fit(
        (rows[0][0], rows[0][2], None),
        (rows[1][0], rows[1][2], None) if len(rows) > 1 else None,
        names_a,
        params,
        params.max_rounds if rounds is None else max(1, rounds[0]),
    )
    if synthetic is not None and len(synthetic):
        extra = _rows(_split(_cap(synthetic, params.max_train_samples // 2, rng), params, rounds is None), spec)
        rows = [
            (np.concatenate([r[0], e[0]]), np.concatenate([r[1], e[1]]), np.concatenate([r[2], e[2]]))
            for r, e in zip(rows, extra, strict=True)
        ]
    offsets = [np.asarray(a.predict(r[0], raw_score=True, num_threads=params.threads)) for r in rows]
    b = _fit(
        (rows[0][1], rows[0][2], offsets[0]),
        (rows[1][1], rows[1][2], offsets[1]) if len(rows) > 1 else None,
        names_b,
        params,
        params.max_rounds if rounds is None else max(1, rounds[1]),
    )
    return a, b


def calibrate(stats: dict[str, LeakageStats], params: TrainParams) -> Calibration:
    """
    The z ramp that keeps the real-data flag rate within budget.

    :param stats: cross-fitted per-player statistics on real data
    :param params: training parameters (budget, floors)
    :return: the calibration
    """
    z = np.array([s.z for s in stats.values() if s.n >= params.min_samples])
    z_flag = params.z_flag_min
    if len(z):
        z_flag = max(z_flag, float(np.quantile(z, 1 - params.flag_budget, method="higher")) + 1e-3)
    z_floor = z_flag - params.flag_score * params.ramp_width
    return Calibration(z_floor=z_floor, z_full=z_floor + params.ramp_width, min_samples=params.min_samples)


def auc(positives: Sequence[float] | np.ndarray, negatives: Sequence[float] | np.ndarray) -> float:
    """
    Probability that a random positive ranks above a random negative (ties count half).

    :param positives: scores of the positive class
    :param negatives: scores of the negative class
    :return: the AUC, NaN if either side is empty
    """
    if len(positives) == 0 or len(negatives) == 0:
        return float("nan")
    p = np.asarray(positives, float)[:, None]
    n = np.asarray(negatives, float)[None, :]
    return float(np.mean((p > n) + 0.5 * (p == n)))


def train(data: TrainingData, params: TrainParams) -> TrainResult:
    """
    Cross-fit, calibrate, and refit the final models on all data.

    :param data: samples
    :param params: training parameters
    :return: the model (not yet gated) with its metrics
    :raises ValueError: if there are too few players to cross-fit
    """
    started = time.monotonic()
    real, syn, spec = data.real, data.synthetic, data.spec
    fold = folds_of(real.player, params.folds)
    syn_fold = folds_of(syn.player, params.folds)
    if len(np.unique(fold)) < params.folds:
        raise ValueError(f"need players in all {params.folds} folds to cross-fit, have {len(np.unique(fold))}")
    gain, syn_gain = np.zeros(len(real)), np.zeros(len(syn))
    iterations: list[tuple[int, int]] = []
    for f in range(params.folds):
        augment = syn.subset(syn_fold != f) if params.augment else None
        a, b = fit_pair(real.subset(fold != f), spec, params, synthetic=augment)
        iterations.append((a.best_iteration or a.current_iteration(), b.best_iteration or b.current_iteration()))
        held = fold == f
        gain[held] = predict_terms(a, b, real.subset(held), spec)
        syn_gain[syn_fold == f] = predict_terms(a, b, syn.subset(syn_fold == f), spec)
        logger.info("fold %d: rounds A %d, B %d", f, *iterations[-1])
    crossfit_s = time.monotonic() - started
    rounds = (int(np.median([i[0] for i in iterations])), int(np.median([i[1] for i in iterations])))
    final_a, final_b = fit_pair(real, spec, params, synthetic=syn if params.augment else None, rounds=rounds)

    block = Calibration().block_samples
    real_stats = aggregate(real.player, real.t, gain, block)
    syn_stats = aggregate(syn.player, syn.t, syn_gain, block)
    calibration = calibrate(real_stats, params)
    model = LeakageModel(spec, calibration, final_a, final_b)
    metrics = _metrics(real_stats, syn_stats, calibration, params, gain)
    metrics.update(
        {
            "samples": len(real),
            "synthetic_samples": len(syn),
            "rounds": list(rounds),
            "crossfit_seconds": round(crossfit_s, 1),
            "train_seconds": round(time.monotonic() - started, 1),
            "model_bytes": model_size_bytes(model),
        }
    )
    model.metadata["metrics"] = metrics
    return TrainResult(model, metrics, real_stats, syn_stats, gain)


def _metrics(
    real: dict[str, LeakageStats],
    syn: dict[str, LeakageStats],
    calibration: Calibration,
    params: TrainParams,
    gain: np.ndarray,
) -> dict[str, Any]:
    eligible = {p: s for p, s in real.items() if s.n >= params.min_samples}
    syn_eligible = {p: s for p, s in syn.items() if s.n >= params.min_samples}
    z_real = np.array([s.z for s in eligible.values()])
    z_syn = np.array([s.z for s in syn_eligible.values()])
    score_real = np.array([calibration.score(s.z, s.n) for s in eligible.values()])
    score_syn = np.array([calibration.score(s.z, s.n) for s in syn_eligible.values()])

    def q(values: np.ndarray, level: float) -> float:
        return round(float(np.quantile(values, level)), 3) if len(values) else float("nan")

    return {
        "players": len(real),
        "eligible_players": len(eligible),
        "synthetic_players": len(syn_eligible),
        "mean_term": round(float(gain.mean()), 5) if len(gain) else float("nan"),
        "real_z": {"p50": q(z_real, 0.5), "p90": q(z_real, 0.9), "p99": q(z_real, 0.99), "max": q(z_real, 1.0)},
        "synthetic_z": {"p10": q(z_syn, 0.1), "p50": q(z_syn, 0.5), "p90": q(z_syn, 0.9)},
        "synthetic_auc": round(auc(z_syn, z_real), 4),
        "synthetic_recall": round(float(np.mean(score_syn >= params.flag_score)), 4) if len(score_syn) else 0.0,
        "real_flag_rate": round(float(np.mean(score_real >= params.flag_score)), 5) if len(score_real) else 0.0,
        "flag_score": params.flag_score,
        "flag_budget": params.flag_budget,
    }
