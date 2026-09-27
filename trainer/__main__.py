"""
Trainer CLI: ``python -m trainer --help``.

``train`` is one run that exits: 0 whether or not the candidate was promoted, non-zero only on errors. Schedule it
weekly (a Render Cron Job, or crontab; see NOTES.md, "Leakage model"). ``devdata`` writes a simulated dataset for
development.
"""

import logging
from pathlib import Path
from typing import Annotated

import typer

from server.orgconfig import load_config
from server.scoring.config import ScoringConfig
from server.scoring.game import load_profile
from server.training.store import LocalStore, open_store
from trainer.devdata import generate
from trainer.gate import Benchmark
from trainer.pipeline import run_training
from trainer.train import TrainParams

app = typer.Typer(help="Train the information-leakage model (no labels needed).", no_args_is_help=True)
logger = logging.getLogger("trainer")

StoreOption = Annotated[
    str, typer.Option("--store", envvar="ESPK_DATA_URL", help="Dataset/model store: s3://bucket/prefix or a path")
]


def _seeds(text: str) -> tuple[int, ...]:
    """``1,2,5`` or a range ``1-6`` (inclusive)."""
    if "-" in text:
        lo, hi = text.split("-", 1)
        return tuple(range(int(lo), int(hi) + 1))
    return tuple(int(s) for s in text.split(",") if s.strip())


@app.command()
def train(  # noqa: PLR0917 - typer options
    store: StoreOption,
    days: Annotated[float, typer.Option(help="Train on the last this many days of data (0: all)")] = 28.0,
    min_new_rows: Annotated[
        int, typer.Option(help="Exit early unless this many position rows arrived since the current model")
    ] = 0,
    config: Annotated[
        Path | None, typer.Option(help="The backend's config.yaml, for its scoring: section (default: defaults)")
    ] = None,
    profile: Annotated[str, typer.Option(help="Game profile for awareness ranges")] = "evrima",
    folds: Annotated[int, typer.Option(help="Cross-fitting folds")] = 3,
    threads: Annotated[int, typer.Option(help="LightGBM threads (0: all cores)")] = 0,
    bench_seeds: Annotated[str, typer.Option(help="Simulator benchmark seeds, e.g. 9001,9002 or 9001-9004")] = (
        "9001,9002"
    ),
    bench_hours: Annotated[float, typer.Option(help="Simulated hours per benchmark server")] = 4.0,
) -> None:
    """Train a candidate on recent data, gate it, and promote it if it passes."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    scoring = load_config(config).scoring_config if config is not None else ScoringConfig()
    outcome = run_training(
        open_store(store),
        config=scoring,
        params=TrainParams(folds=folds, threads=threads),
        bench=Benchmark(seeds=_seeds(bench_seeds), hours=bench_hours),
        profile=load_profile(profile),
        days=days or None,
        min_new_rows=min_new_rows,
    )
    logger.info("%s%s", outcome.status, f": {outcome.reason}" if outcome.reason else "")


@app.command()
def devdata(
    store: Annotated[Path, typer.Option(help="Directory to write the dataset to")],
    hours: Annotated[float, typer.Option(help="Simulated hours per server")] = 4.0,
    seeds: Annotated[str, typer.Option(help="Seeds, e.g. 1,2,3 or 1-6; one server per kind and seed")] = "1-6",
    kinds: Annotated[str, typer.Option(help="World kinds: mixed, clan")] = "mixed,clan",
) -> None:
    """Write a simulated dataset in the export format (pseudonymised with a dev key), for development."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    rows = generate(LocalStore(store), seeds=_seeds(seeds), hours=hours, kinds=tuple(kinds.split(",")))
    logger.info("wrote %d position rows to %s", rows, store)


if __name__ == "__main__":
    app(prog_name="python -m trainer")
