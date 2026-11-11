"""An image and thermal simulator of the guide camera (plan §2: every device has one).

**Sky.** A fixed catalogue of stars, drawn afresh for every frame as Gaussian stamps, so the
field can move: a slow ``drift`` (pixels per second), a pointing ``offset`` set by
:meth:`nudge` (a telescope offset), and per-frame image motion ``jitter_px`` (seeing and
tracking), mostly common to all stars. The star size is ``seeing_fwhm_px`` varying a little
from frame to frame, broadened by ``defocus_px`` (the knob a focus stage will drive). Lengths
are in unbinned sensor pixels; the plate scale is not known yet.

**Pacing.** A frame takes its exposure time, or the time to draw it if that is longer, as a
real camera's video mode does: about 30 fps at the shortest exposures.

**Noise and heat.** Shot noise and read noise are drawn once per (binned) pixel. Dark current
doubles every 6 °C, so cooling the sensor visibly cleans the image. The sensor follows the
cooler setpoint with a first-order lag (``tau_s``), limited to ``MAX_DELTA_C`` below ambient;
with the cooler off it drifts back to ambient. Time comes from the injectable ``clock``.
"""

import math
import threading
import time
from collections.abc import Callable
from typing import ClassVar

import numpy as np
from pydantic import Field, PrivateAttr

from .base import CaptureSettings, ExposureAborted, GuideCameraBase

SENSOR = (1296, 882)  # a fifth of the ASI294MM's 4144 x 2822: cheap to draw and to encode
MAX_DELTA_C = 35.0  # the ASI294MM Pro cools about this far below ambient
STARS = 150
SKY_E_PER_S = 2.0
IMAGE_MOTION_SHARE = 0.05  # per-star scatter around the common image motion, in jitter_px
DARK_E_PER_S_AT_20C = 0.4
READ_NOISE = 3.0


class SimulatedASICamera(GuideCameraBase):
    kind: ClassVar[str] = "simulated"

    ambient_c: float = 20.0
    tau_s: float = 45.0  # thermal time constant

    seeing_fwhm_px: float = 5.0
    defocus_px: float = 0.0  # added in quadrature to the star width
    jitter_px: float = 0.3  # rms image motion per frame, common to every star
    drift_x_px_s: float = 0.02
    drift_y_px_s: float = -0.01
    offset_x_px: float = 0.0  # where the telescope has been told to point, see nudge()
    offset_y_px: float = 0.0
    # Is the light path blocked (the shutter closed)? Then only the dark current is recorded.
    light_blocked: Callable[[], bool] = Field(default=lambda: False, exclude=True)

    _stars: tuple[np.ndarray, np.ndarray, np.ndarray] = PrivateAttr()  # x, y, flux (e/s)
    _drift: list[float] = PrivateAttr(default_factory=lambda: [0.0, 0.0])
    _drift_t: float = PrivateAttr(default=0.0)
    _rng: np.random.Generator = PrivateAttr()
    _abort: threading.Event = PrivateAttr(default_factory=threading.Event)
    _temp: float = PrivateAttr(default=0.0)
    _t: float = PrivateAttr(default=0.0)
    _enabled: bool = PrivateAttr(default=False)
    _setpoint: float = PrivateAttr(default=0.0)
    _lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        self._rng = np.random.default_rng()
        self._stars = _catalogue(SENSOR, STARS)
        self._temp, self._t = self.ambient_c, self.clock()
        self._drift_t = self._t

    @property
    def sensor_size(self) -> tuple[int, int]:
        return SENSOR

    # -- thermal model ------------------------------------------------------------------

    def _advance(self) -> None:
        now = self.clock()
        dt, self._t = now - self._t, now
        if dt <= 0:
            return
        target = self.ambient_c
        if self._enabled:
            target = max(self._setpoint, self.ambient_c - MAX_DELTA_C)
        self._temp += (target - self._temp) * (1 - math.exp(-dt / self.tau_s))

    def _read_temperature(self) -> float:
        with self._lock:
            self._advance()
            return round(self._temp, 1)

    def _read_cooler_power(self) -> float:
        with self._lock:
            self._advance()
            if not self._enabled:
                return 0.0
            holding = max(0.0, self.ambient_c - self._temp) / MAX_DELTA_C * 100
            pulling = 10.0 * max(0.0, self._temp - self._setpoint)
            return min(100.0, holding + pulling)

    def _write_cooler(self, enabled: bool, setpoint_c: float) -> None:
        with self._lock:
            self._advance()  # up to now under the old settings
            self._enabled, self._setpoint = enabled, setpoint_c

    # -- exposures ----------------------------------------------------------------------

    def _abort_exposure(self) -> None:
        self._abort.set()

    def nudge(self, dx_px: float, dy_px: float) -> None:
        """Move the field as a telescope offset would (the stars move by this much)."""
        self.offset_x_px += dx_px
        self.offset_y_px += dy_px

    def _electrons(self, s: CaptureSettings) -> np.ndarray:
        """The noise-free image in electrons, after ROI and binning, stars where they are now."""
        now = self.clock()
        dt, self._drift_t = now - self._drift_t, now
        self._drift[0] += self.drift_x_px_s * dt
        self._drift[1] += self.drift_y_px_s * dt
        rng = self._rng
        shift_x = self.offset_x_px + self._drift[0] + self.jitter_px * rng.standard_normal()
        shift_y = self.offset_y_px + self._drift[1] + self.jitter_px * rng.standard_normal()
        scatter = IMAGE_MOTION_SHARE * self.jitter_px

        full_w, full_h = SENSOR
        x0, y0, w, h = (
            (s.roi.x, s.roi.y, s.roi.width, s.roi.height) if s.roi else (0, 0, full_w, full_h)
        )
        dark = DARK_E_PER_S_AT_20C * 2 ** ((self._read_temperature() - 20.0) / 6.0)
        if self.light_blocked():
            canvas = np.full((h, w), dark * s.exposure_s, np.float32)
            return _bin(canvas, s.binning)
        canvas = np.full((h, w), (SKY_E_PER_S + dark) * s.exposure_s, np.float32)

        seeing = self.seeing_fwhm_px / 2.355 * (1 + 0.1 * rng.standard_normal())  # this frame's
        sigma = max(0.5, math.hypot(seeing, self.defocus_px))
        half = math.ceil(4 * sigma)
        for x, y, flux in zip(*self._stars, strict=True):
            cx = x + shift_x + scatter * rng.standard_normal() - x0
            cy = y + shift_y + scatter * rng.standard_normal() - y0
            ix, iy = round(cx), round(cy)
            if ix + half < 0 or ix - half >= w or iy + half < 0 or iy - half >= h:
                continue
            xs = np.arange(max(ix - half, 0), min(ix + half + 1, w))
            ys = np.arange(max(iy - half, 0), min(iy + half + 1, h))
            gx = np.exp(-0.5 * ((xs - cx) / sigma) ** 2)
            gy = np.exp(-0.5 * ((ys - cy) / sigma) ** 2)
            canvas[ys[0] : ys[-1] + 1, xs[0] : xs[-1] + 1] += (
                flux * s.exposure_s / (2 * math.pi * sigma**2) * np.outer(gy, gx)
            )
        return _bin(canvas, s.binning)

    def _expose(self, settings: CaptureSettings) -> np.ndarray:
        started = time.monotonic()
        try:
            electrons = self._electrons(settings)
            b = settings.binning
            noise = self._rng.standard_normal(electrons.shape, dtype=np.float32)
            sigma = np.sqrt(electrons + READ_NOISE**2 * b * b)
            gain = 10 ** (settings.gain / 200)  # the ASI gain unit is 0.1 dB
            img = (electrons + sigma * noise) * gain + settings.offset * 16
            # The frame is ready when the exposure is over, or when it has been drawn.
            if self._abort.wait(max(0.0, settings.exposure_s - (time.monotonic() - started))):
                raise ExposureAborted
        finally:
            self._abort.clear()
        return np.clip(img, 0, 65535).astype(np.uint16)


def _bin(canvas: np.ndarray, b: int) -> np.ndarray:
    """Sum b x b pixels into one."""
    if b > 1:
        h2, w2 = (canvas.shape[0] // b) * b, (canvas.shape[1] // b) * b
        canvas = canvas[:h2, :w2].reshape(h2 // b, b, w2 // b, b).sum(axis=(1, 3))
    return canvas


def _catalogue(size: tuple[int, int], count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fixed star positions (pixels) and fluxes (electrons per second)."""
    rng = np.random.default_rng(294)
    w, h = size
    margin = 40  # room for the field to move before stars leave it
    x = rng.uniform(margin, w - margin, count)
    y = rng.uniform(margin, h - margin, count)
    flux = rng.lognormal(8.0, 1.0, count)
    return x, y, flux
