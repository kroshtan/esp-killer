"""Alert image rendering: coordinate transform, nearby-player selection, gap handling and the PNG itself."""

from io import BytesIO

import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

from server.alerts.render import (
    Highlight,
    PathImage,
    RenderOptions,
    Track,
    render_path_png,
    select_nearby,
    split_segments,
)
from server.alerts.transform import MapTransform, game_to_plot

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
T0 = 1_790_000_000.0  # epoch seconds
CM = 100.0  # game units per metre


def line(player_id: str, start_m: tuple[float, float], end_m: tuple[float, float], n: int = 61) -> Track:
    """A straight walk over 10 minutes, in game units."""
    t = T0 + np.linspace(0, 600, n)
    return Track(
        player_id,
        f"name of {player_id}",
        t,
        np.linspace(start_m[0], end_m[0], n) * CM,
        np.linspace(start_m[1], end_m[1], n) * CM,
    )


def decode(png: bytes) -> Image.Image:
    assert png.startswith(PNG_SIGNATURE)
    img = Image.open(BytesIO(png))
    img.load()
    return img


# --- transform ---


def test_identity_transform_converts_game_units_to_metres() -> None:
    u, v = game_to_plot(np.array([100.0, -250.0]), np.array([0.0, 1000.0]), MapTransform())
    np.testing.assert_allclose(u, [1.0, -2.5])
    np.testing.assert_allclose(v, [0.0, 10.0])


def test_scale_and_translate_transform() -> None:
    transform = MapTransform(matrix=((0.5, 0.0, 10.0), (0.0, -2.0, 3.0)))
    u, v = transform.apply(np.array([4.0, np.nan]), np.array([1.0, 2.0]))
    np.testing.assert_allclose(u, [12.0, np.nan])
    np.testing.assert_allclose(v, [1.0, np.nan])  # an absent sample stays absent in both coordinates
    assert transform.scale == pytest.approx(1.0)  # sqrt(|0.5 * -2|)
    u, v = game_to_plot(np.array([400.0]), np.array([100.0]), transform, units_per_metre=100.0)
    np.testing.assert_allclose([u[0], v[0]], [12.0, 1.0])


@pytest.mark.parametrize(
    "matrix",
    [((1.0, 0.0), (0.0, 1.0)), ((1.0, 0.0, 0.0),), ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))],
)
def test_transform_rejects_wrong_shape(matrix: object) -> None:
    with pytest.raises(ValidationError):
        MapTransform(matrix=matrix)


def test_non_positive_units_per_metre_rejected() -> None:
    with pytest.raises(ValueError, match="units_per_metre"):
        game_to_plot(np.zeros(1), np.zeros(1), MapTransform(), units_per_metre=0)


# --- segments ---


def test_nan_gap_splits_segments_and_is_not_bridged() -> None:
    t = np.arange(6.0)
    x = np.array([0.0, 1.0, np.nan, np.nan, 4.0, 5.0])
    y = np.zeros(6)
    runs = split_segments(t, x, y)
    assert [r[0].tolist() for r in runs] == [[0.0, 1.0], [4.0, 5.0]]
    assert not any(np.isnan(r[1]).any() for r in runs)


def test_long_time_gap_splits_segments() -> None:
    t = np.array([0.0, 5.0, 100.0, 105.0])
    runs = split_segments(t, np.arange(4.0), np.arange(4.0), max_gap_s=30.0)
    assert [len(r[0]) for r in runs] == [2, 2]
    assert len(split_segments(t, np.arange(4.0), np.arange(4.0))) == 1


def test_segments_of_degenerate_tracks() -> None:
    assert split_segments(np.array([]), np.array([]), np.array([])) == []
    assert split_segments(np.arange(2.0), np.full(2, np.nan), np.zeros(2)) == []
    runs = split_segments(np.arange(3.0), np.array([np.nan, 1.0, np.nan]), np.zeros(3))
    assert [r[1].tolist() for r in runs] == [[1.0]]


def test_track_rejects_mismatched_arrays() -> None:
    with pytest.raises(ValueError, match="equal length"):
        Track("p", "p", np.arange(3.0), np.arange(2.0), np.arange(3.0))


# --- nearby players ---


def test_nearby_includes_close_excludes_far_closest_first_and_capped() -> None:
    flagged = line("flagged", (0, 0), (1000, 0))
    crosser = line("crosser", (500, 300), (500, -300))  # crosses the path
    close = line("close", (0, 150), (1000, 150))  # parallel, 150 m away
    far = line("far", (0, 2000), (1000, 2000))
    tracks = [flagged, far, close, crosser]

    picked = select_nearby(tracks, "flagged", radius_m=200, max_players=5)
    assert [t.player_id for t in picked] == ["crosser", "close"]

    assert [t.player_id for t in select_nearby(tracks, "flagged", radius_m=200, max_players=1)] == ["crosser"]
    assert select_nearby(tracks, "flagged", radius_m=200, max_players=0) == []


def test_nearby_counts_the_path_not_simultaneous_positions() -> None:
    flagged = line("flagged", (0, 0), (1000, 0))
    # Walks the same route in the opposite direction: far apart most of the time, but on the path.
    reverse = line("reverse", (1000, 10), (0, 10))
    assert [t.player_id for t in select_nearby([flagged, reverse], "flagged", 50, 3)] == ["reverse"]


def test_nearby_always_include_takes_slots_first() -> None:
    flagged = line("flagged", (0, 0), (1000, 0))
    tracks = [flagged, line("near", (0, 10), (1000, 10)), line("target", (0, 3000), (0, 2000))]
    picked = select_nearby(tracks, "flagged", radius_m=100, max_players=1, always_include={"target"})
    assert [t.player_id for t in picked] == ["target"]


def test_nearby_ignores_nan_and_requires_flagged_track() -> None:
    flagged = line("flagged", (0, 0), (1000, 0))
    absent = Track("absent", "a", flagged.t, np.full_like(flagged.x, np.nan), flagged.y)
    assert select_nearby([flagged, absent], "flagged", 100, 3) == []
    with pytest.raises(ValueError, match="no track"):
        select_nearby([absent], "flagged", 100, 3)
    with pytest.raises(ValueError, match="positive"):
        select_nearby([flagged], "flagged", 0, 3)


# --- rendering ---


def test_render_full_image() -> None:
    flagged = line("flagged", (0, 0), (1500, 800))
    flagged.x[20:25] = np.nan  # a gap in the middle
    target = line("target", (1600, 900), (1550, 850))
    other = line("other", (200, -100), (900, 600))
    image = PathImage(
        flagged_player_id="flagged",
        flagged_name="Rex $pecial\n<b>",  # mathtext and control characters must not break rendering
        tracks=[flagged, target, other],
        window_start_t=T0,
        window_end_t=T0 + 600,
        highlights=[Highlight(T0 + 150, T0 + 450, "target")],
        awareness_m=300.0,
        subtitle="Server 1",
    )
    img = decode(render_path_png(image, RenderOptions(width_px=1000, height_px=800)))
    assert img.size == (1000, 800)


def test_render_respects_size_and_transform() -> None:
    image = PathImage("p", "P", [line("p", (0, 0), (100, 100))], T0, T0 + 600)
    options = RenderOptions(
        width_px=640, height_px=480, transform=MapTransform(matrix=((0.01, 0, 5), (0, -0.01, 5)), unit="px")
    )
    assert decode(render_path_png(image, options)).size == (640, 480)


def test_render_single_point_and_no_other_players() -> None:
    single = Track("p", "P", np.array([T0 + 10]), np.array([5000.0]), np.array([-300.0]))
    image = PathImage("p", "P", [single], T0, T0 + 600, highlights=[Highlight(T0, T0 + 60, None)], awareness_m=300)
    assert decode(render_path_png(image)).size == (1000, 800)


def test_render_flagged_player_absent_in_window_and_missing_target() -> None:
    outside = Track("p", "P", np.array([T0 - 100.0, T0 - 50]), np.zeros(2), np.zeros(2))
    image = PathImage("p", "P", [outside], T0, T0 + 600, highlights=[Highlight(T0, T0 + 60, "gone")], awareness_m=300)
    decode(render_path_png(image))


def test_path_image_validation() -> None:
    track = line("p", (0, 0), (1, 1))
    with pytest.raises(ValueError, match="no track"):
        PathImage("q", "Q", [track], T0, T0 + 600)
    with pytest.raises(ValueError, match="after"):
        PathImage("p", "P", [track], T0, T0)
