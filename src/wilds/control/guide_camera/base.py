"""The guide camera's abstract base (plan §3.8, conventions memory).

The base holds everything shared: settings and their validation, the cooler and its status,
the latest frame and its fan-out to viewers, frame numbering and JPEG encoding, streaming,
and the activities. A subclass supplies only the hardware hooks (``_expose``, the temperature
and cooler calls), so the simulator exercises all the shared code too.

Hooks are blocking and are called from a worker thread. Nothing here imports FastAPI; the
router turns :class:`SettingError` and :class:`CameraBusy` into the standard error body.
"""

import abc
import asyncio
import io
import logging
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, ValidationError

from ..activities import Activities, registry

log = logging.getLogger(__name__)

COMPONENT = "guide-camera"

EXPOSURE_RANGE_S = (0.001, 60.0)
GAIN_RANGE = (0, 500)
OFFSET_RANGE = (0, 100)
MIN_ROI_PIXELS = 16

COOLER_RANGE_C = (-40.0, 30.0)  # the setpoints the ASI SDK accepts
DEFAULT_SETPOINT_C = -10.0
AT_SETPOINT_TOL_C = 1.0  # "stable" when the sensor is this close to the setpoint
COOLER_CACHE_S = 1.0  # status is polled by every viewer; the hardware is read at most this often

KEPT_FRAMES = 8  # recent frames still downloadable as FITS


class GuideCameraActivity(Enum):
    Exposing = 1
    Streaming = 2
    Cancelling = 3  # an exposure is being aborted


class SettingError(ValueError):
    """A setting or cooler request that is not acceptable (HTTP 422, ``invalid``)."""


class CameraBusy(RuntimeError):
    """The camera is doing something that excludes this (HTTP 409, ``busy``)."""


class ExposureAborted(Exception):
    """The exposure was cut short by :meth:`GuideCameraBase.abort`."""


# -- models -----------------------------------------------------------------------------


class Roi(BaseModel):
    """A window in unbinned sensor pixels."""

    x: int = Field(ge=0)
    y: int = Field(ge=0)
    width: int = Field(ge=MIN_ROI_PIXELS)
    height: int = Field(ge=MIN_ROI_PIXELS)


class Settings(BaseModel):
    exposure_s: float = Field(default=0.5, ge=EXPOSURE_RANGE_S[0], le=EXPOSURE_RANGE_S[1])
    gain: int = Field(default=100, ge=GAIN_RANGE[0], le=GAIN_RANGE[1])
    offset: int = Field(default=10, ge=OFFSET_RANGE[0], le=OFFSET_RANGE[1])
    binning: Literal[1, 2] = 1
    roi: Roi | None = None  # None: the whole sensor


class SettingsUpdate(BaseModel):
    """A partial change: only the fields sent are changed (``roi: null`` clears the window)."""

    model_config = ConfigDict(extra="forbid")

    exposure_s: float | None = None
    gain: int | None = None
    offset: int | None = None
    binning: Literal[1, 2] | None = None
    roi: Roi | None = None


CoolerState = Literal["unsupported", "off", "cooling", "stable"]


class CoolerRequest(BaseModel):
    """Switch the cooler and/or change its setpoint; whatever is left out stays as it is."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool | None = None
    setpoint_c: float | None = None


class Cooler(BaseModel):
    supported: bool
    enabled: bool = False
    setpoint_c: float | None = None  # the target, once one has been set
    temperature_c: float | None = None  # the sensor, measured
    power_pct: float | None = None  # cooler drive, 0-100
    state: CoolerState = "unsupported"


class FrameMeta(BaseModel):
    id: int
    taken_ms: int  # end of the exposure, ms since the Unix epoch, UTC
    exposure_s: float
    gain: int
    offset: int
    binning: int
    roi: Roi | None
    width: int  # pixels of this frame, after ROI and binning
    height: int
    temperature_c: float | None


class GuideCameraStatus(BaseModel):
    """What ``/status`` carries for the camera."""

    kind: str
    streaming: bool = False
    exposing: bool = False
    settings: Settings
    cooler: Cooler
    last_frame_id: int | None = None


@dataclass
class Frame:
    meta: FrameMeta
    jpeg: bytes
    pixels: np.ndarray  # uint16, kept so the raw frame can be fetched as FITS

    def fits(self) -> bytes:
        from astropy.io import fits  # imported on use: only needed for downloads

        hdu = fits.PrimaryHDU(self.pixels)
        h = hdu.header
        h["EXPTIME"] = (self.meta.exposure_s, "[s] exposure time")
        h["GAIN"] = (self.meta.gain, "camera gain")
        h["OFFSET"] = (self.meta.offset, "camera offset")
        h["XBINNING"] = h["YBINNING"] = self.meta.binning
        if self.meta.temperature_c is not None:
            h["CCD-TEMP"] = (self.meta.temperature_c, "[C] sensor temperature")
        h["FRAMEID"] = self.meta.id
        out = io.BytesIO()
        hdu.writeto(out)
        return out.getvalue()


# -- the base ---------------------------------------------------------------------------


class GuideCameraBase(BaseModel, abc.ABC):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    kind: ClassVar[str]
    cooler_supported: ClassVar[bool] = True

    settings: Settings = Field(default_factory=Settings)
    clock: Callable[[], float] = Field(default=time.monotonic, exclude=True)

    _seq: int = PrivateAttr(default=0)
    _latest: Frame | None = PrivateAttr(default=None)
    _frames: "OrderedDict[int, Frame]" = PrivateAttr(default_factory=OrderedDict)
    _subscribers: "set[asyncio.Queue[Frame]]" = PrivateAttr(default_factory=set)
    _exposing: bool = PrivateAttr(default=False)
    _streaming: bool = PrivateAttr(default=False)
    _stream_task: "asyncio.Task[None] | None" = PrivateAttr(default=None)
    _cooler_enabled: bool = PrivateAttr(default=False)
    _setpoint_c: float | None = PrivateAttr(default=None)
    _cooler_cache: "tuple[float, Cooler] | None" = PrivateAttr(default=None)

    # -- hardware hooks -----------------------------------------------------------------

    @property
    @abc.abstractmethod
    def sensor_size(self) -> tuple[int, int]:
        """(width, height) of the whole sensor in pixels, unbinned."""

    @abc.abstractmethod
    def _expose(self, settings: Settings) -> np.ndarray:
        """Take one exposure (blocking); a uint16 image after ROI and binning. Raise
        :class:`ExposureAborted` if :meth:`_abort_exposure` cut it short."""

    @abc.abstractmethod
    def _abort_exposure(self) -> None:
        """Make a running ``_expose`` return early. Called from another thread."""

    @abc.abstractmethod
    def _read_temperature(self) -> float:
        """The sensor temperature in °C."""

    @abc.abstractmethod
    def _read_cooler_power(self) -> float:
        """Cooler drive, 0-100 %."""

    @abc.abstractmethod
    def _write_cooler(self, enabled: bool, setpoint_c: float) -> None:
        """Switch the cooler and set its target."""

    def _close(self) -> None:  # noqa: B027 - optional hook
        """Release the hardware."""

    # -- activities ---------------------------------------------------------------------

    @property
    def activities(self) -> Activities:
        # Looked up each time, so a registry cleared by a test is simply registered again.
        try:
            return registry.get(COMPONENT)
        except KeyError:
            return registry.register(COMPONENT, GuideCameraActivity)

    # -- settings -----------------------------------------------------------------------

    def update_settings(self, update: SettingsUpdate) -> Settings:
        """Apply a partial change, validated as a whole; nothing changes if it is refused."""
        merged = {**self.settings.model_dump(), **update.model_dump(exclude_unset=True)}
        try:
            new = Settings.model_validate(merged)
        except ValidationError as e:
            first = e.errors()[0]
            where = ".".join(str(p) for p in first["loc"])
            raise SettingError(f"{where}: {first['msg']}") from e
        self._check_roi(new)
        self.settings = new
        return new

    def _check_roi(self, s: Settings) -> None:
        if s.roi is None:
            return
        width, height = self.sensor_size
        if s.roi.x + s.roi.width > width or s.roi.y + s.roi.height > height:
            raise SettingError(f"roi: outside the {width}x{height} sensor")
        if s.roi.width % s.binning or s.roi.height % s.binning:
            raise SettingError(f"roi: width and height must be multiples of binning {s.binning}")

    # -- cooler -------------------------------------------------------------------------

    def cooler(self) -> Cooler:
        """The cooler and the sensor temperature. Read from the hardware at most once a
        second, however many viewers poll."""
        now = self.clock()
        if self._cooler_cache is None or now - self._cooler_cache[0] >= COOLER_CACHE_S:
            self._cooler_cache = (now, self._read_cooler())
        return self._cooler_cache[1]

    def _read_cooler(self) -> Cooler:
        temp = self._read_temperature()
        if not self.cooler_supported:
            return Cooler(supported=False, temperature_c=temp, state="unsupported")
        enabled, setpoint = self._cooler_enabled, self._setpoint_c
        if not enabled:
            state: CoolerState = "off"
        elif setpoint is not None and abs(temp - setpoint) <= AT_SETPOINT_TOL_C:
            state = "stable"
        else:
            state = "cooling"
        return Cooler(
            supported=True,
            enabled=enabled,
            setpoint_c=setpoint,
            temperature_c=temp,
            power_pct=round(self._read_cooler_power(), 1) if enabled else 0.0,
            state=state,
        )

    def set_cooler(self, request: CoolerRequest) -> Cooler:
        """Switch the cooler on or off and/or change the target. Turning it on without a
        setpoint uses the last one, or :data:`DEFAULT_SETPOINT_C`."""
        if not self.cooler_supported:
            raise SettingError("this camera has no cooler")
        lo, hi = COOLER_RANGE_C
        if request.setpoint_c is not None and not lo <= request.setpoint_c <= hi:
            raise SettingError(f"setpoint_c: must be between {lo:g} and {hi:g} °C")
        enabled = self._cooler_enabled if request.enabled is None else request.enabled
        setpoint = request.setpoint_c
        if setpoint is None:
            setpoint = self._setpoint_c if self._setpoint_c is not None else DEFAULT_SETPOINT_C
        self._write_cooler(enabled, setpoint)
        self._cooler_enabled, self._setpoint_c = enabled, setpoint
        self._cooler_cache = None
        log.info("cooler %s, setpoint %.1f °C", "on" if enabled else "off", setpoint)
        return self.cooler()

    # -- frames -------------------------------------------------------------------------

    @property
    def video_frame(self) -> Frame | None:
        """The latest frame, or ``None`` before the first. Never blocks."""
        return self._latest

    def frame(self, frame_id: int) -> Frame | None:
        return self._frames.get(frame_id)

    @property
    def exposing(self) -> bool:
        return self._exposing

    @property
    def streaming(self) -> bool:
        return self._streaming

    def subscribe(self) -> "asyncio.Queue[Frame]":
        """A queue that always holds the newest frame only: a slow viewer skips frames, it
        never builds a backlog and never slows the camera."""
        q: asyncio.Queue[Frame] = asyncio.Queue(maxsize=1)
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: "asyncio.Queue[Frame]") -> None:
        self._subscribers.discard(q)

    def _publish(self, frame: Frame) -> None:
        self._latest = frame
        self._frames[frame.meta.id] = frame
        while len(self._frames) > KEPT_FRAMES:
            self._frames.popitem(last=False)
        for q in self._subscribers:
            if q.full():
                q.get_nowait()  # drop the stale one
            q.put_nowait(frame)

    async def single_frame(self) -> Frame:
        """A fresh exposure with the current settings. Raises :class:`CameraBusy` while
        streaming, :class:`ExposureAborted` if :meth:`abort` cut it short."""
        if self._streaming:
            raise CameraBusy("the guide camera is streaming")
        return await self._take()

    async def _take(self) -> Frame:
        if self._exposing:
            raise CameraBusy("the guide camera is already exposing")
        s = self.settings.model_copy(deep=True)
        self._exposing = True
        self.activities.start(GuideCameraActivity.Exposing, details=[f"{s.exposure_s:g} s"])
        try:
            pixels = await asyncio.to_thread(self._expose, s)
        finally:
            self._exposing = False
            self.activities.end(GuideCameraActivity.Exposing)
            self.activities.end(GuideCameraActivity.Cancelling)
        frame = await asyncio.to_thread(self._make_frame, pixels, s)
        self._publish(frame)
        return frame

    def abort(self) -> None:
        """Cut a running exposure short. A no-op when nothing is exposing."""
        if self._exposing:
            if not self.activities.is_active(GuideCameraActivity.Cancelling):
                self.activities.start(GuideCameraActivity.Cancelling)
            self._abort_exposure()

    def _make_frame(self, pixels: np.ndarray, s: Settings) -> Frame:
        self._seq += 1
        h, w = pixels.shape
        meta = FrameMeta(
            id=self._seq,
            taken_ms=time.time_ns() // 1_000_000,
            exposure_s=s.exposure_s,
            gain=s.gain,
            offset=s.offset,
            binning=s.binning,
            roi=s.roi,
            width=w,
            height=h,
            temperature_c=self.cooler().temperature_c,
        )
        return Frame(meta=meta, jpeg=_jpeg(pixels), pixels=pixels)

    # -- streaming ----------------------------------------------------------------------

    async def start(self) -> None:
        """Take frames continuously until :meth:`stop`. A no-op if already streaming."""
        if self._streaming:
            return
        if self._exposing:
            raise CameraBusy("the guide camera is exposing")
        self._streaming = True
        self.activities.start(GuideCameraActivity.Streaming)
        self._stream_task = asyncio.create_task(self._stream())

    async def _stream(self) -> None:
        try:
            while self._streaming:
                await self._take()
        except ExposureAborted:
            pass
        except Exception:
            log.exception("guide camera stream stopped on an error")
        finally:
            self._streaming = False
            self.activities.end(GuideCameraActivity.Streaming)

    async def stop(self) -> None:
        self._streaming = False
        self.abort()
        task, self._stream_task = self._stream_task, None
        if task is not None:
            await task

    async def close(self) -> None:
        await self.stop()
        self._close()

    # -- status -------------------------------------------------------------------------

    def status(self) -> GuideCameraStatus:
        return GuideCameraStatus(
            kind=self.kind,
            streaming=self._streaming,
            exposing=self._exposing,
            settings=self.settings,
            cooler=self.cooler(),
            last_frame_id=self._latest.meta.id if self._latest else None,
        )


def _jpeg(pixels: np.ndarray) -> bytes:
    """A display stretch from the background (median - 2 sigma) to median + 25 sigma, encoded
    once for every viewer. Starting at the background keeps noise a quiet grey, which also
    compresses far better. (Viewer-chosen stretches are the GUI's business, plan §3.8.)"""
    from PIL import Image

    sample = pixels[::4, ::4].astype(np.float32)  # a sample is enough for the statistics
    median = float(np.median(sample))
    sigma = max(1.4826 * float(np.median(np.abs(sample - median))), 1.0)
    lo, hi = median - 2 * sigma, median + 25 * sigma
    scaled = np.clip((pixels.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)
    out = io.BytesIO()
    Image.fromarray((scaled * 255).astype(np.uint8), "L").save(out, "JPEG", quality=80)
    return out.getvalue()
