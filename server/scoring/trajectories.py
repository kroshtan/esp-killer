"""
Player trajectories on one server, resampled onto a regular time grid.

Scoring works on a dense array ``pos[t, p] = (x, y)`` in metres, with NaN where the player was not on the server.
Polls arrive every few seconds but not exactly regularly, and players come and go, so the raw samples are
linearly interpolated onto a grid of ``resample_s``. Gaps longer than ``max_gap_s`` are left as NaN rather than
bridged: a player who vanished for a minute may have died and respawned elsewhere.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from server.scoring.config import ScoringConfig

REQUIRED_COLUMNS = ("t", "player_id", "x", "y")


@dataclass(frozen=True)
class Trajectories:
    server_id: str
    t: np.ndarray  # (T,) seconds (epoch or relative; only differences matter)
    player_ids: list[str]
    pos: np.ndarray  # (T, P, 2) metres, NaN when absent
    classes: np.ndarray  # (T, P) object: class name or None

    @property
    def dt(self) -> float:
        """
        Grid step in seconds.

        :return: the step
        """
        return float(self.t[1] - self.t[0]) if len(self.t) > 1 else 0.0

    @property
    def present(self) -> np.ndarray:
        """
        Whether each player is on the server at each grid time.

        :return: (T, P) bool
        """
        return ~np.isnan(self.pos[..., 0])

    def steps(self, seconds: float) -> int:
        """
        Convert a duration to a whole number of grid steps (at least 1).

        :param seconds: the duration
        :return: number of steps
        """
        return max(1, round(seconds / self.dt)) if self.dt > 0 else 1


def from_frame(frame: pd.DataFrame, server_id: str, config: ScoringConfig) -> Trajectories:
    """
    Resample one server's position samples onto a regular grid.

    :param frame: long-format samples with columns ``t`` (seconds), ``player_id``, ``x``, ``y`` (metres) and
        optionally ``dino_class``
    :param server_id: the server these samples belong to
    :param config: scoring config (grid step and maximum gap)
    :return: the trajectories; empty if there are fewer than two distinct sample times
    :raises ValueError: if a required column is missing
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    player_ids = sorted(frame["player_id"].unique().tolist())
    t0, t1 = float(frame["t"].min()) if len(frame) else 0.0, float(frame["t"].max()) if len(frame) else 0.0
    if len(frame) == 0 or t1 <= t0:
        return Trajectories(server_id, np.zeros(0), player_ids, np.zeros((0, len(player_ids), 2)), np.empty((0, 0)))

    grid = np.arange(t0, t1 + 1e-9, config.resample_s)
    pos = np.full((len(grid), len(player_ids), 2), np.nan)
    classes = np.full((len(grid), len(player_ids)), None, dtype=object)
    has_class = "dino_class" in frame.columns
    for p, (_, samples) in enumerate(frame.sort_values("t").groupby("player_id", sort=True)):
        ts = samples["t"].to_numpy(dtype=float)
        ts, first = np.unique(ts, return_index=True)
        xs = samples["x"].to_numpy(dtype=float)[first]
        ys = samples["y"].to_numpy(dtype=float)[first]
        # For each grid time, the samples either side of it; present only if both exist and are close enough.
        after = np.searchsorted(ts, grid, side="left")
        before = np.searchsorted(ts, grid, side="right") - 1
        exact = (after < len(ts)) & (ts[np.minimum(after, len(ts) - 1)] == grid)
        inside = (before >= 0) & (after < len(ts))
        gap = np.where(inside, ts[np.minimum(after, len(ts) - 1)] - ts[np.maximum(before, 0)], np.inf)
        ok = exact | (inside & (gap <= config.max_gap_s))
        pos[ok, p, 0] = np.interp(grid[ok], ts, xs)
        pos[ok, p, 1] = np.interp(grid[ok], ts, ys)
        if has_class:
            cls = samples["dino_class"].to_numpy(dtype=object)[first]
            classes[ok, p] = cls[np.maximum(before[ok], 0)]
    return Trajectories(server_id=server_id, t=grid, player_ids=player_ids, pos=pos, classes=classes)
