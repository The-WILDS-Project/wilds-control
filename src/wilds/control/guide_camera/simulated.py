"""An image and thermal simulator of the guide camera (plan §2: every device has one).

The sky is a fixed random star field, so frames are repeatable up to noise. Dark current
doubles every 6 °C, so cooling the sensor visibly cleans the image. The sensor follows the
cooler setpoint with a first-order lag (``tau_s``), limited to ``MAX_DELTA_C`` below ambient;
with the cooler off it drifts back to ambient. Time comes from the injectable ``clock``.
"""

import math
import threading
from typing import ClassVar

import numpy as np
from pydantic import PrivateAttr

from .base import ExposureAborted, GuideCameraBase, Settings

SENSOR = (1296, 882)  # a fifth of the ASI294MM's 4144 x 2822: cheap to draw and to encode
MAX_DELTA_C = 35.0  # the ASI294MM Pro cools about this far below ambient
STARS = 60
SKY_E_PER_S = 2.0
DARK_E_PER_S_AT_20C = 0.4
READ_NOISE = 3.0


class SimulatedASICamera(GuideCameraBase):
    kind: ClassVar[str] = "simulated"

    ambient_c: float = 20.0
    tau_s: float = 45.0  # thermal time constant

    _field: np.ndarray = PrivateAttr()  # electrons per second per pixel
    _rng: np.random.Generator = PrivateAttr()
    _abort: threading.Event = PrivateAttr(default_factory=threading.Event)
    _temp: float = PrivateAttr(default=0.0)
    _t: float = PrivateAttr(default=0.0)
    _enabled: bool = PrivateAttr(default=False)
    _setpoint: float = PrivateAttr(default=0.0)
    _lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)

    def model_post_init(self, __context) -> None:
        self._rng = np.random.default_rng()
        self._field = _star_field(SENSOR, STARS)
        self._temp, self._t = self.ambient_c, self.clock()

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

    def _expose(self, s: Settings) -> np.ndarray:
        try:
            if self._abort.wait(s.exposure_s):
                raise ExposureAborted
        finally:
            self._abort.clear()
        temp = self._read_temperature()
        dark = DARK_E_PER_S_AT_20C * 2 ** ((temp - 20.0) / 6.0)
        gain = 10 ** (s.gain / 200)  # the ASI gain unit is 0.1 dB
        electrons = self._field + SKY_E_PER_S + dark
        img = self._rng.poisson(electrons * s.exposure_s).astype(np.float32) * gain
        img += self._rng.normal(0.0, READ_NOISE, img.shape) * gain + s.offset * 16
        if s.roi is not None:
            r = s.roi
            img = img[r.y : r.y + r.height, r.x : r.x + r.width]
        if s.binning > 1:
            b = s.binning
            h, w = (img.shape[0] // b) * b, (img.shape[1] // b) * b
            img = img[:h, :w].reshape(h // b, b, w // b, b).sum(axis=(1, 3))
        return np.clip(img, 0, 65535).astype(np.uint16)


def _star_field(size: tuple[int, int], count: int) -> np.ndarray:
    """A fixed field of Gaussian stars, in electrons per second."""
    rng = np.random.default_rng(294)
    w, h = size
    field = np.zeros((h, w), np.float32)
    half = 12
    yy, xx = np.mgrid[-half : half + 1, -half : half + 1]
    for _ in range(count):
        x, y = int(rng.integers(half, w - half)), int(rng.integers(half, h - half))
        flux = float(rng.lognormal(8.0, 1.0))
        sigma = float(rng.uniform(1.8, 2.8))
        stamp = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        field[y - half : y + half + 1, x - half : x + half + 1] += flux * stamp / stamp.sum()
    return field
