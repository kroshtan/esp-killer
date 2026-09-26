"""
World-to-image coordinates for alert images.

The game reports positions in Unreal units (centimetres) on axes whose orientation relative to the in-game map is
not yet known (see NOTES.md, "Verify on a real server"). Until the map is calibrated, alert images use plain axes
in metres. Calibration will be an affine fit from a few known landmarks to pixel positions on a map image, so the
mapping is kept as one configurable 2x3 matrix rather than being baked into the renderer: calibrating a map is
then a config change, not a code change.
"""

from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

Row = tuple[float, float, float]
IDENTITY: tuple[Row, Row] = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))


class MapTransform(BaseModel):
    """
    Affine map from world metres to plot coordinates: ``[u, v] = M @ [x_m, y_m, 1]``.

    The default is the identity, so plots are in metres with the game's own axes. ``background_image_path`` is
    reserved for the calibrated map image the matrix will one day point into; it is accepted and ignored for now.
    When it is used, note that image rows grow downwards, so a calibrated matrix will usually flip the y axis.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    matrix: tuple[Row, Row] = IDENTITY
    unit: str = Field(default="m", max_length=16)  # axis label unit for the transformed coordinates
    background_image_path: Path | None = None  # not used yet, see the class docstring

    @property
    def scale(self) -> float:
        """
        How much the transform stretches lengths, on average (square root of the linear part's determinant).

        :return: plot units per metre; 1.0 for a degenerate matrix
        """
        det = abs(self.matrix[0][0] * self.matrix[1][1] - self.matrix[0][1] * self.matrix[1][0])
        return float(np.sqrt(det)) if det > 0 else 1.0

    def apply(self, x_m: np.ndarray, y_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Transform world coordinates in metres. NaN stays NaN.

        :param x_m: x coordinates in metres, any shape
        :param y_m: y coordinates in metres, same shape as ``x_m``
        :return: (u, v) plot coordinates, same shape as the inputs
        """
        (a, b, c), (d, e, f) = self.matrix
        x = np.asarray(x_m, dtype=float)
        y = np.asarray(y_m, dtype=float)
        return a * x + b * y + c, d * x + e * y + f


def game_to_plot(
    x: np.ndarray, y: np.ndarray, transform: MapTransform, units_per_metre: float = 100.0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert game coordinates (Unreal units) to plot coordinates.

    :param x: x coordinates in game units
    :param y: y coordinates in game units
    :param transform: world metres -> plot coordinates
    :param units_per_metre: game units per metre (Unreal: 100)
    :return: (u, v) plot coordinates
    :raises ValueError: if ``units_per_metre`` is not positive
    """
    if units_per_metre <= 0:
        raise ValueError("units_per_metre must be positive")
    return transform.apply(np.asarray(x, dtype=float) / units_per_metre, np.asarray(y, dtype=float) / units_per_metre)
