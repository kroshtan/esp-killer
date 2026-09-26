"""
The PNG attached to an alert: the flagged player's path over the last minutes, and the players around them.

A score says *that* a player looks suspicious; the picture lets an admin judge *whether* in a few seconds. It
shows the flagged path with time running light to dark, the beeline episodes that produced the evidence (the
player's path while lined up, where the target was when it started, and the awareness radius that target was
outside of), and the other players who came near, so an honest explanation (a group, a trail, a waterhole) is
visible too.

Everything here is pure: tracks in, PNG bytes out, no database, no network, no clock. Rendering uses the
object-oriented :class:`matplotlib.figure.Figure` API with the Agg canvas, never pyplot, so no global figure state
is touched and images can be rendered from worker threads. Names and positions are personal data: they appear in
the image, which goes to the org's admins, and never in logs.
"""

import logging
import math
import re
from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from io import BytesIO
from typing import Any

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from pydantic import BaseModel, ConfigDict, Field

from server.alerts.transform import MapTransform, game_to_plot

log = logging.getLogger(__name__)

# Light chart surface and text inks; the flagged path is a one-hue sequential ramp (time), beelines take the
# first contrasting categorical hue, and everyone else stays a recessive grey.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e6e5e0"
OTHERS = "#b5b4ae"
TARGET_PATH = "#6f6e69"
FLAGGED_RAMP = ("#86b6ef", "#3987e5", "#1c5cab", "#0d366b")  # early -> late
FLAGGED_START = FLAGGED_RAMP[0]
FLAGGED_END = FLAGGED_RAMP[-1]
HIGHLIGHT = "#eb6834"

_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")


@dataclass(frozen=True)
class Track:
    """One player's positions. Coordinates are game units; NaN marks samples where the player was absent."""

    player_id: str
    label: str
    t: np.ndarray  # (N,) seconds, increasing
    x: np.ndarray  # (N,) game units
    y: np.ndarray  # (N,) game units

    def __post_init__(self) -> None:
        """
        Coerce the arrays to float and check their shapes.

        :raises ValueError: if the arrays are not one-dimensional or differ in length
        """
        arrays = [np.asarray(a, dtype=float) for a in (self.t, self.x, self.y)]
        if any(a.ndim != 1 for a in arrays) or len({len(a) for a in arrays}) != 1:
            raise ValueError("t, x and y must be one-dimensional and of equal length")
        for name, a in zip(("t", "x", "y"), arrays, strict=True):
            object.__setattr__(self, name, a)

    def clipped(self, start_t: float, end_t: float) -> "Track":
        """
        The part of the track inside a time window.

        :param start_t: window start (inclusive)
        :param end_t: window end (inclusive)
        :return: a new track with only the samples in the window
        """
        keep = (self.t >= start_t) & (self.t <= end_t)
        return replace(self, t=self.t[keep], x=self.x[keep], y=self.y[keep])


@dataclass(frozen=True)
class Highlight:
    """An interval of the flagged track to draw distinctly, e.g. a beeline episode and the player it targeted."""

    start_t: float
    end_t: float
    target_player_id: str | None = None


@dataclass(frozen=True)
class PathImage:
    """Everything one alert image shows. ``tracks`` includes the flagged player's own track."""

    flagged_player_id: str
    flagged_name: str
    tracks: Sequence[Track]
    window_start_t: float  # epoch seconds
    window_end_t: float
    highlights: Sequence[Highlight] = ()
    highlight_label: str = "beeline"
    awareness_m: float | None = None  # drawn around the flagged player at each highlight start
    subtitle: str | None = None  # e.g. the server name or the score; shown under the title

    def __post_init__(self) -> None:
        """
        Check the window and that the flagged player has a track.

        :raises ValueError: if the window is empty or the flagged player has no track
        """
        if not self.window_end_t > self.window_start_t:
            raise ValueError("window_end_t must be after window_start_t")
        if not any(tr.player_id == self.flagged_player_id for tr in self.tracks):
            raise ValueError("the flagged player has no track")


class RenderOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    width_px: int = Field(default=1000, ge=200, le=4000)
    height_px: int = Field(default=800, ge=200, le=4000)
    dpi: int = Field(default=100, ge=50, le=300)
    units_per_metre: float = Field(default=100.0, gt=0)  # Unreal units are centimetres
    transform: MapTransform = MapTransform()
    # Samples further apart than this are not joined by a line (a death, a disconnect). None: only NaN breaks.
    max_gap_s: float | None = Field(default=30.0, gt=0)
    max_label_chars: int = Field(default=28, ge=4)
    # The view is fitted to the flagged path and highlights plus this margin, so a far-roaming neighbour does not
    # shrink the path that matters. Metres; the awareness radius is used if it is larger.
    margin_m: float = Field(default=150.0, ge=0)


def split_segments(
    t: np.ndarray, x: np.ndarray, y: np.ndarray, max_gap_s: float | None = None
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Split a track into runs of consecutive valid samples, so gaps are drawn as gaps and never bridged.

    A sample is invalid if any of t, x or y is NaN or infinite. A run also ends where the time between two
    samples exceeds ``max_gap_s``.

    :param t: times in seconds
    :param x: x coordinates
    :param y: y coordinates
    :param max_gap_s: largest time step still drawn as a line; None to split on invalid samples only
    :return: (t, x, y) per run, in order; single-sample runs included
    """
    t, x, y = (np.asarray(a, dtype=float) for a in (t, x, y))
    valid = np.isfinite(t) & np.isfinite(x) & np.isfinite(y)
    idx = np.flatnonzero(valid)
    if len(idx) == 0:
        return []
    # A new run starts where the previous valid sample is not the previous sample, or is too long ago.
    breaks = np.diff(idx) > 1
    if max_gap_s is not None:
        breaks |= np.diff(t[idx]) > max_gap_s
    cuts = np.flatnonzero(breaks) + 1
    return [(t[run], x[run], y[run]) for run in np.split(idx, cuts)]


def _min_distance(ax: np.ndarray, ay: np.ndarray, bx: np.ndarray, by: np.ndarray, chunk: int = 1024) -> float:
    """Smallest distance between any point of A and any point of B (inf if either is empty)."""
    if len(ax) == 0 or len(bx) == 0:
        return math.inf
    best = math.inf
    for s in range(0, len(ax), chunk):
        dx = ax[s : s + chunk, None] - bx[None, :]
        dy = ay[s : s + chunk, None] - by[None, :]
        best = min(best, float(np.sqrt(np.min(dx * dx + dy * dy))))
    return best


def _finite_xy(track: Track) -> tuple[np.ndarray, np.ndarray]:
    ok = np.isfinite(track.x) & np.isfinite(track.y) & np.isfinite(track.t)
    return track.x[ok], track.y[ok]


def select_nearby(
    tracks: Iterable[Track],
    flagged_player_id: str,
    radius_m: float,
    max_players: int,
    *,
    units_per_metre: float = 100.0,
    always_include: Collection[str] = (),
) -> list[Track]:
    """
    The other players worth drawing: those who came within ``radius_m`` of anywhere on the flagged player's path.

    Distance is to the path, not to where the flagged player was at the same moment, so someone who crossed the
    route a few minutes earlier or later (a trail, a waterhole) is shown too. Pass tracks already clipped to the
    window. Players in ``always_include`` (e.g. beeline targets) are kept however far away they stayed and take
    their slots first; the rest follow closest first, up to ``max_players`` in total.

    :param tracks: every track, including the flagged player's
    :param flagged_player_id: the flagged player
    :param radius_m: how close to the path counts as nearby, in metres
    :param max_players: at most this many tracks are returned
    :param units_per_metre: game units per metre
    :param always_include: player ids kept regardless of distance
    :return: the selected tracks, closest first (``always_include`` ones before the rest)
    :raises ValueError: if the flagged player has no track, or the radius or cap is invalid
    """
    if radius_m <= 0 or max_players < 0 or units_per_metre <= 0:
        raise ValueError("radius_m and units_per_metre must be positive and max_players non-negative")
    tracks = list(tracks)
    flagged = next((tr for tr in tracks if tr.player_id == flagged_player_id), None)
    if flagged is None:
        raise ValueError("the flagged player has no track")
    fx, fy = _finite_xy(flagged)
    ranked: list[tuple[bool, float, Track]] = []
    for tr in tracks:
        if tr.player_id == flagged_player_id:
            continue
        ox, oy = _finite_xy(tr)
        d = _min_distance(fx, fy, ox, oy) / units_per_metre
        pinned = tr.player_id in always_include
        if pinned or d <= radius_m:
            ranked.append((not pinned, d, tr))
    ranked.sort(key=lambda r: (r[0], r[1]))
    return [tr for _, _, tr in ranked[:max_players]]


def _safe_text(text: str, max_chars: int) -> str:
    """Player-chosen text as a plain matplotlib label: no control characters, no mathtext, bounded length."""
    text = _CONTROL_CHARS.sub("", text).strip() or "?"
    if len(text) > max_chars:
        text = text[: max_chars - 1] + "…"
    return text.replace("$", r"\$")


def _interp_at(track: Track, t: float, max_gap_s: float | None) -> tuple[float, float] | None:
    """Position at time ``t``, interpolated inside a run of valid samples; None if the player was absent."""
    for ts, xs, ys in split_segments(track.t, track.x, track.y, max_gap_s):
        if ts[0] <= t <= ts[-1]:
            return float(np.interp(t, ts, xs)), float(np.interp(t, ts, ys))
    return None


def _format_window(start_t: float, end_t: float) -> str:
    start = datetime.fromtimestamp(start_t, UTC)
    end = datetime.fromtimestamp(end_t, UTC)
    end_fmt = "%H:%M" if end.date() == start.date() else "%Y-%m-%d %H:%M"
    return f"{start:%Y-%m-%d %H:%M}–{end.strftime(end_fmt)} UTC"


def _to_plot(track: Track, options: RenderOptions) -> Track:
    u, v = game_to_plot(track.x, track.y, options.transform, options.units_per_metre)
    return replace(track, x=u, y=v)


class _Bounds:
    """Running bounding box of what the view must show."""

    def __init__(self) -> None:
        self.lo = np.array([math.inf, math.inf])
        self.hi = np.array([-math.inf, -math.inf])

    def add(self, x: Sequence[float] | np.ndarray | float, y: Sequence[float] | np.ndarray | float) -> None:
        xs, ys = np.broadcast_arrays(
            np.atleast_1d(np.asarray(x, dtype=float)), np.atleast_1d(np.asarray(y, dtype=float))
        )
        ok = np.isfinite(xs) & np.isfinite(ys)
        if ok.any():
            self.lo = np.minimum(self.lo, [xs[ok].min(), ys[ok].min()])
            self.hi = np.maximum(self.hi, [xs[ok].max(), ys[ok].max()])

    def limits(self, margin: float) -> tuple[float, float, float, float]:
        if not np.all(np.isfinite(self.lo)):
            return -margin, margin, -margin, margin
        return self.lo[0] - margin, self.hi[0] + margin, self.lo[1] - margin, self.hi[1] + margin


def _key(label: str, **style: Any) -> tuple[Line2D, str]:
    """A legend entry: an empty line carrying the style of the marks it stands for."""
    return Line2D([], [], **style), label


class _Drawing:
    """One alert image being drawn; each method adds one layer, in stacking order."""

    def __init__(self, image: PathImage, options: RenderOptions) -> None:
        self.image = image
        self.options = options
        self.gap = options.max_gap_s
        self.scale = options.transform.scale  # plot units per metre
        self.minutes = (image.window_end_t - image.window_start_t) / 60.0
        window = (image.window_start_t, image.window_end_t)
        self.tracks = {tr.player_id: _to_plot(tr.clipped(*window), options) for tr in image.tracks}
        self.flagged = self.tracks[image.flagged_player_id]
        self.others = [tr for pid, tr in self.tracks.items() if pid != image.flagged_player_id]
        self.fig = Figure(
            figsize=(options.width_px / options.dpi, options.height_px / options.dpi),
            dpi=options.dpi,
            facecolor=SURFACE,
            layout="constrained",
        )
        FigureCanvasAgg(self.fig)
        self.ax = self.fig.add_subplot()
        self.ax.set_facecolor(SURFACE)
        self.bounds = _Bounds()
        self.bounds.add(self.flagged.x, self.flagged.y)
        self.legend: list[tuple[Line2D, str]] = []

    def _plot_during(self, track: Track, start_t: float, end_t: float, **style: Any) -> None:
        """Draw the part of ``track`` between two times, respecting gaps."""
        for ts, xs, ys in split_segments(track.t, track.x, track.y, self.gap):
            inside = (ts >= start_t) & (ts <= end_t)
            if inside.sum() >= 2:  # noqa: PLR2004
                self.ax.plot(xs[inside], ys[inside], solid_capstyle="round", **style)

    def others_paths(self) -> None:
        """Everyone else: thin, recessive grey."""
        for tr in self.others:
            for _, xs, ys in split_segments(tr.t, tr.x, tr.y, self.gap):
                self.ax.plot(xs, ys, color=OTHERS, lw=1.2, solid_capstyle="round", zorder=1)
        if self.others:
            self.legend.append(_key("other players nearby", color=OTHERS, lw=1.2))

    def highlights(self) -> None:
        """The path while lined up, the target's path meanwhile, and where the target was when it started."""
        image = self.image
        targets_marked = radius_drawn = False
        for h in image.highlights:
            target = self.tracks.get(h.target_player_id) if h.target_player_id else None
            self._plot_during(self.flagged, h.start_t, h.end_t, color=HIGHLIGHT, lw=6, alpha=0.9, zorder=3)
            if target is not None:
                self._plot_during(target, h.start_t, h.end_t, color=TARGET_PATH, lw=2, zorder=2)
            here = _interp_at(self.flagged, h.start_t, self.gap)
            if here is None:
                continue
            self.bounds.add(*here)
            if image.awareness_m:
                r = image.awareness_m * self.scale
                self.ax.add_patch(Circle(here, r, fill=False, ec=HIGHLIGHT, lw=1.2, ls=(0, (2, 2)), zorder=2))
                self.bounds.add([here[0] - r, here[0] + r], [here[1] - r, here[1] + r])
                radius_drawn = True
            there = _interp_at(target, h.start_t, self.gap) if target is not None else None
            if there is not None:
                self._target_at_start(here, there)
                targets_marked = True
        label = image.highlight_label
        if image.highlights:
            self.legend.append(_key(label, color=HIGHLIGHT, lw=6, alpha=0.9))
        if targets_marked:
            target_style = {"color": HIGHLIGHT, "lw": 1.2, "ls": (0, (5, 3)), "marker": "D", "ms": 7, "mec": SURFACE}
            self.legend.append(_key(f"target at {label} start", **target_style))
            self.legend.append(_key("target's path meanwhile", color=TARGET_PATH, lw=2))
        if radius_drawn:
            radius = f"awareness radius ({image.awareness_m:.0f} m)"
            self.legend.append(_key(radius, color=HIGHLIGHT, lw=1.2, ls=(0, (2, 2))))

    def _target_at_start(self, here: tuple[float, float], there: tuple[float, float]) -> None:
        """A dashed line from the player to their target, labelled with the distance between them."""
        self.ax.plot(*zip(here, there, strict=True), color=HIGHLIGHT, lw=1.2, ls=(0, (5, 3)), zorder=3)
        self.ax.plot(*there, marker="D", ms=8, mfc=HIGHLIGHT, mec=SURFACE, mew=2, ls="none", zorder=6)
        dist_m = math.dist(here, there) / self.scale
        self.ax.annotate(
            f"{dist_m / 1000:.1f} km" if dist_m >= 1000 else f"{dist_m:.0f} m",  # noqa: PLR2004
            ((here[0] + there[0]) / 2, (here[1] + there[1]) / 2),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
            color=INK_SECONDARY,
            zorder=7,
            bbox={"boxstyle": "round,pad=0.15", "fc": SURFACE, "ec": "none", "alpha": 0.85},
        )
        self.bounds.add(*there)

    def flagged_path(self) -> None:
        """The flagged path coloured by time, light (early) to dark (late), with a colour bar as its key."""
        cmap = LinearSegmentedColormap.from_list("flagged", FLAGGED_RAMP)
        norm = Normalize(0.0, self.minutes)
        f = self.flagged
        for ts, xs, ys in split_segments(f.t, f.x, f.y, self.gap):
            if len(ts) < 2:  # noqa: PLR2004
                continue
            points = np.column_stack([xs, ys])
            segments = list(np.stack([points[:-1], points[1:]], axis=1))
            lines = LineCollection(segments, cmap=cmap, norm=norm, linewidths=2.5, capstyle="round", zorder=4)
            lines.set_array(((ts[:-1] + ts[1:]) / 2 - self.image.window_start_t) / 60.0)
            self.ax.add_collection(lines)
        colorbar = self.fig.colorbar(
            ScalarMappable(norm=norm, cmap=cmap), ax=self.ax, fraction=0.035, pad=0.01, aspect=40
        )
        name = _safe_text(self.image.flagged_name, self.options.max_label_chars)
        colorbar.set_label(f"{name}: minutes into window", color=INK)
        colorbar.outline.set_visible(False)
        colorbar.ax.tick_params(colors=INK_SECONDARY, labelsize=8, length=0)
        self.legend.insert(0, _key("flagged player (light→dark = time)", color=FLAGGED_RAMP[2], lw=2.5))

        valid = np.isfinite(f.x) & np.isfinite(f.y)
        if valid.any():
            fx, fy = f.x[valid], f.y[valid]
            start: dict[str, Any] = {"marker": "o", "mfc": SURFACE, "mec": FLAGGED_START, "mew": 3, "ls": "none"}
            end: dict[str, Any] = {"marker": "s", "mfc": FLAGGED_END, "mec": SURFACE, "mew": 2, "ls": "none"}
            self.ax.plot(fx[0], fy[0], ms=11, zorder=8, **start)
            self.ax.plot(fx[-1], fy[-1], ms=10, zorder=8, **end)
            self.legend += [_key("start", ms=9, **start), _key("end", ms=9, **end)]

    def fit_view(self) -> tuple[float, float, float, float]:
        """
        Fit the view to the flagged path and highlights, so a far-roaming neighbour does not shrink them.

        :return: (x0, x1, y0, y1) of the fitted box (the equal aspect may widen one axis further)
        """
        margin = max(self.options.margin_m, self.image.awareness_m or 0.0) * self.scale * 0.5
        x0, x1, y0, y1 = self.bounds.limits(margin)
        # Replace the data limits (which include every neighbour's whole path) and let the equal aspect widen one
        # axis to fill the figure. Setting both limits directly would fight the aspect.
        self.ax.dataLim.set_points(np.array([[x0, y0], [x1, y1]]))
        self.ax.margins(0)
        self.ax.set_aspect("equal", adjustable="datalim")
        self.ax.autoscale_view()
        return x0, x1, y0, y1

    def others_labels(self, view: tuple[float, float, float, float]) -> None:
        """
        Each other player's name at the last point where they were in view.

        A label that would sit on top of an earlier one goes below its dot instead.

        :param view: (x0, x1, y0, y1) from :meth:`fit_view`
        """
        x0, x1, y0, y1 = view
        placed: list[tuple[float, float]] = []
        near = 0.03 * max(x1 - x0, y1 - y0)
        for tr in self.others:
            inside = np.isfinite(tr.x) & np.isfinite(tr.y) & (tr.x >= x0) & (tr.x <= x1) & (tr.y >= y0) & (tr.y <= y1)
            if not inside.any():
                continue
            k = int(np.flatnonzero(inside)[-1])
            px, py = float(tr.x[k]), float(tr.y[k])
            crowded = any(abs(px - qx) < near and abs(py - qy) < near for qx, qy in placed)
            placed.append((px, py))
            self.ax.plot(px, py, marker="o", ms=5, mfc=OTHERS, mec=SURFACE, mew=1.5, ls="none", zorder=5)
            self.ax.annotate(
                _safe_text(tr.label, self.options.max_label_chars),
                (px, py),
                xytext=(5, -11 if crowded else 3),
                textcoords="offset points",
                fontsize=8,
                color=INK_SECONDARY,
                zorder=7,
                annotation_clip=True,
            )

    def frame(self) -> None:
        """Recessive axes and grid, the title and details line, and the legend below the plot."""
        ax, image = self.ax, self.image
        unit = self.options.transform.unit
        ax.set_xlabel(f"x ({unit})", color=INK_SECONDARY, fontsize=9)
        ax.set_ylabel(f"y ({unit})", color=INK_SECONDARY, fontsize=9)
        ax.tick_params(colors=INK_SECONDARY, labelsize=8, length=0)
        ax.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_visible(False)

        name = _safe_text(image.flagged_name, self.options.max_label_chars)
        title = f"{name}: last {self.minutes:.0f} min"
        ax.set_title(title, loc="left", fontsize=14, color=INK, fontweight="bold", pad=22)
        n_others, n_highlights = len(self.others), len(image.highlights)
        details = [
            _format_window(image.window_start_t, image.window_end_t),
            f"{n_others} other player{'s' * (n_others != 1)} shown",
        ]
        if n_highlights:
            details.append(f"{n_highlights} {image.highlight_label} episode{'s' * (n_highlights != 1)}")
        if image.subtitle:
            details.insert(0, _safe_text(image.subtitle, 80))
        ax.text(0, 1.015, "  ·  ".join(details), transform=ax.transAxes, fontsize=9, color=INK_SECONDARY, va="bottom")
        self.fig.legend(
            [h for h, _ in self.legend],
            [label for _, label in self.legend],
            loc="outside lower center",
            ncols=min(4, len(self.legend)),
            frameon=False,
            fontsize=9,
            labelcolor=INK,
            handlelength=2.4,
        )

    def png(self) -> bytes:
        """
        Encode the figure.

        :return: PNG bytes
        """
        buf = BytesIO()
        self.fig.savefig(buf, format="png", facecolor=SURFACE)
        return buf.getvalue()


def render_path_png(image: PathImage, options: RenderOptions | None = None) -> bytes:
    """
    Render an alert image.

    :param image: what to draw; tracks are clipped to the image's window
    :param options: size, coordinate mapping and styling; defaults to :class:`RenderOptions`
    :return: PNG bytes
    """
    drawing = _Drawing(image, options or RenderOptions())
    drawing.others_paths()
    drawing.highlights()
    drawing.flagged_path()
    drawing.others_labels(drawing.fit_view())
    drawing.frame()
    png = drawing.png()
    log.debug(
        "rendered alert image: %d tracks, %d highlights, %d bytes",
        len(drawing.tracks),
        len(image.highlights),
        len(png),
    )
    return png
