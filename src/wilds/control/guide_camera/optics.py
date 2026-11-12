"""The simulated guide camera's *true* optics: what the calibration has to find.

The simulator used to take its geometry from the stored calibration, so a calibration could only
ever read its own answer back. These are the simulator's own, hidden values: the field it draws
and the slit shadow it casts follow them whatever the stored calibration says. The stored
calibration starts out as a rough guess (``guide_camera._seed_simulated_calibration``), and the
calibration procedure (plate solve and slit shadow) has to recover the numbers below.

Angles are counter-clockwise on the screen from the top of the image; pixels are the unbinned
sensor's, x to the right and y downward. The camera and the slit are on the instrument, so both
turn with the rotator: the slit stays where it is in the image and the sky turns past it.
"""

import math
from typing import Literal

from pydantic import BaseModel, Field


class SimulatedOptics(BaseModel):
    plate_scale_arcsec_px: float = Field(0.272, gt=0)
    parity: Literal[1, -1] = 1
    orientation_deg: float = 31.0  # angle of sky north at slit PA 0
    slit_x_px: float = 668.0  # the slit's centre; the target sits on it
    slit_y_px: float = 431.0
    # The slit's long axis in the image. A little off the orientation: the rotator's zero is not
    # quite where the slit points north (a few tenths of a degree, as in a real instrument).
    slit_angle_deg: float = 31.4
    slit_length_arcsec: float = 30.0
    leak: float = Field(0.03, ge=0, le=1)  # the share of light seen through the gap (it is dark)

    def slit_axes(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Unit vectors (x right, y down) along the slit and across it."""
        a = math.radians(self.slit_angle_deg)
        along = (-math.sin(a), -math.cos(a))
        across = (math.cos(a), -math.sin(a))
        return along, across

    def shadow(self, width_arcsec: float | None, shape: tuple[int, int], x0: int = 0, y0: int = 0):
        """The slit's transmission over a ``shape`` = (h, w) patch of the unbinned sensor whose
        corner is (x0, y0): 1 on the jaws, ``leak`` in the gap, anti-aliased at its edges. Without
        a width (the stage is moving or unknown) there is no gap."""
        import numpy as np

        h, w = shape
        if not width_arcsec or width_arcsec <= 0:
            return np.ones((h, w), np.float32)
        half_len = self.slit_length_arcsec / self.plate_scale_arcsec_px / 2
        half_wid = width_arcsec / self.plate_scale_arcsec_px / 2
        (ux, uy), (vx, vy) = self.slit_axes()
        xs = np.arange(w, dtype=np.float32)[None, :] + x0 - self.slit_x_px
        ys = np.arange(h, dtype=np.float32)[:, None] + y0 - self.slit_y_px
        along = xs * ux + ys * uy
        across = xs * vx + ys * vy

        # what a one-pixel-wide pixel at ``d`` from the centre has of a strip ``half`` each side
        def cover(d, half):
            return np.clip(np.minimum(d + 0.5, half) - np.maximum(d - 0.5, -half), 0.0, 1.0)

        cov_l = cover(along, half_len)
        cov_w = cover(across, half_wid)
        gap = cov_l * cov_w
        return (1.0 - gap * (1.0 - self.leak)).astype(np.float32)
