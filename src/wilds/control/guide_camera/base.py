"""The guide camera's abstract base (plan §3.8, conventions memory).

The base holds everything shared: its two settings slices (``conf``, operational, and ``cal``,
calibration, plan §3.10) and the cooler and its status,
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
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from ..activities import Activities, registry
from ..conf import ChangeBusy, ConfHandle, ConfModel, InvalidChange, by_opmode, setting
from ..lifecycle import Lifecycle, LifecycleStatus, OpState

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

CaptureMode = Literal["single_frame", "video"]  # the two settings groups
FrameMode = Literal["single_frame", "video", "dark", "bias"]  # what a frame was taken as
CalibrationKind = Literal["dark", "bias"]
IMAGETYP = {
    "single_frame": "Light Frame",
    "video": "Light Frame",
    "dark": "Dark Frame",
    "bias": "Bias Frame",
}

KEPT_FRAMES = 8  # recent frames still downloadable as FITS


class GuideCameraActivity(Enum):
    Exposing = 1
    Streaming = 2
    Cancelling = 3  # an exposure is being aborted
    StartingUp = 4
    ShuttingDown = 5


class SettingError(InvalidChange):
    """A setting or cooler request that is not acceptable (HTTP 422, ``invalid``)."""


class CameraBusy(ChangeBusy):
    """The camera is doing something that excludes this (HTTP 409, ``busy``)."""


class ExposureAborted(Exception):
    """The exposure was cut short by :meth:`GuideCameraBase.abort`."""


# -- models -----------------------------------------------------------------------------


class Roi(ConfModel):
    """A region of interest in unbinned sensor pixels."""

    x: int = Field(ge=0)
    y: int = Field(ge=0)
    width: int = Field(ge=MIN_ROI_PIXELS)
    height: int = Field(ge=MIN_ROI_PIXELS)


class CaptureSettings(ConfModel):
    """How one kind of capture is taken. The camera keeps one group for single frames and one for
    the video stream (``camera.conf.current.single_frame`` / ``.video``), so a live view can run
    short and binned while a science-grade frame is long and full-resolution."""

    exposure_s: float = setting(
        0.5, unit="s", label="Exposure", ge=EXPOSURE_RANGE_S[0], le=EXPOSURE_RANGE_S[1]
    )
    gain: int = setting(
        100, label="Gain", help="In 0.1 dB steps", ge=GAIN_RANGE[0], le=GAIN_RANGE[1]
    )
    offset: int = setting(
        10,
        label="Offset",
        help="Black level added to every pixel, so noise near zero is not clipped",
        ge=OFFSET_RANGE[0],
        le=OFFSET_RANGE[1],
    )
    binning: Literal[1, 2] = setting(
        1,
        label="Binning",
        help="Pixels combined into one: 1 is none, 2 is 2×2 (more sensitive, coarser)",
    )
    roi: Roi | None = setting(
        None,
        label="ROI",
        help="Region of interest: read out only this part of the sensor (smaller, faster frames); "
        "x, y, width and height are in full-resolution pixels, whatever the binning. "
        "Not set is the full sensor",
    )


class GuideCameraConf(ConfModel):
    """The operational settings (``camera.conf``): the control holder may change them. They
    are kept in the settings store; these are the defaults of a camera nobody has set up."""

    single_frame: CaptureSettings = Field(
        default_factory=CaptureSettings,
        title="Single frame",
        description="Settings for one frame taken on request",
    )
    video: CaptureSettings = Field(
        default_factory=CaptureSettings,
        title="Video",
        description="Settings for the continuous video stream",
    )
    # Start cooling at startup? Yes in development; production is to be decided (off until then).
    cooler_enabled: bool = by_opmode(
        development=True, production=False, label="Cooler", help="Cool the sensor to the setpoint"
    )
    cooler_setpoint_c: float = setting(
        DEFAULT_SETPOINT_C,
        unit="°C",
        label="Cooler setpoint",
        ge=COOLER_RANGE_C[0],
        le=COOLER_RANGE_C[1],
    )


class GuideCameraCal(ConfModel):
    """The calibration values (``camera.cal``): the holder in engineering mode may change them.
    None until measured; a procedure that needs one calls ``camera.cal.require(...)``."""

    plate_scale_arcsec_px: float | None = setting(None, unit="arcsec/px", label="Plate scale", gt=0)
    slit_pixel_x: float | None = setting(None, unit="px", label="Slit pixel x")
    slit_pixel_y: float | None = setting(None, unit="px", label="Slit pixel y")
    orientation_deg: float | None = setting(
        None, unit="°", label="Orientation", help="Position angle of sky north in the image"
    )
    parity: Literal[1, -1] | None = setting(
        None, label="Parity", help="-1 if the image is mirrored"
    )


class CaptureUpdate(BaseModel):
    """A partial change of one group: only the fields sent are changed (``roi: null`` clears it)."""

    model_config = ConfigDict(extra="forbid")

    exposure_s: float | None = None
    gain: int | None = None
    offset: int | None = None
    binning: Literal[1, 2] | None = None
    roi: Roi | None = None


class SettingsUpdate(BaseModel):
    """A partial change: only the groups and fields sent are changed."""

    model_config = ConfigDict(extra="forbid")

    single_frame: CaptureUpdate | None = None
    video: CaptureUpdate | None = None


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
    mode: FrameMode = "single_frame"  # which settings group took it, or dark / bias
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
    settings: GuideCameraConf
    conf_version: int = 0  # the stored version of the settings (0: nothing saved yet)
    cal_version: int = 0
    cooler: Cooler
    last_frame_id: int | None = None
    # the newest dark or bias frame (kept among the recent frames, downloadable as FITS), so a
    # viewer can show a calibration run's frames as they are taken
    last_calibration_frame_id: int | None = None
    lifecycle: LifecycleStatus


@dataclass
class Frame:
    meta: FrameMeta
    jpeg: bytes
    pixels: np.ndarray  # uint16, kept so the raw frame can be fetched as FITS

    def fits(self, extra: dict | None = None) -> bytes:
        """The frame as a FITS file; ``extra`` adds header cards (a master adds NCOMBINE)."""
        from astropy.io import fits  # imported on use: only needed for downloads

        hdu = fits.PrimaryHDU(self.pixels)
        h = hdu.header
        h["IMAGETYP"] = (IMAGETYP[self.meta.mode], "kind of frame")
        h["EXPTIME"] = (self.meta.exposure_s, "[s] exposure time")
        h["GAIN"] = (self.meta.gain, "camera gain")
        h["OFFSET"] = (self.meta.offset, "camera offset")
        h["XBINNING"] = h["YBINNING"] = self.meta.binning
        if self.meta.temperature_c is not None:
            h["CCD-TEMP"] = (self.meta.temperature_c, "[C] sensor temperature")
        if self.meta.roi is not None:
            h["XORGSUBF"] = (self.meta.roi.x, "[px] sub-frame origin, unbinned")
            h["YORGSUBF"] = (self.meta.roi.y, "[px] sub-frame origin, unbinned")
        h["FRAMEID"] = self.meta.id
        for key, value in (extra or {}).items():
            h[key] = value
        out = io.BytesIO()
        hdu.writeto(out)
        return out.getvalue()


# -- the base ---------------------------------------------------------------------------


class GuideCameraBase(BaseModel, abc.ABC):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    kind: ClassVar[str]
    cooler_supported: ClassVar[bool] = True

    clock: Callable[[], float] = Field(default=time.monotonic, exclude=True)

    _seq: int = PrivateAttr(default=0)
    _latest: Frame | None = PrivateAttr(default=None)
    _frames: OrderedDict[int, Frame] = PrivateAttr(default_factory=OrderedDict)
    _subscribers: set[asyncio.Queue[Frame]] = PrivateAttr(default_factory=set)
    _exposing: bool = PrivateAttr(default=False)
    _streaming: bool = PrivateAttr(default=False)
    _stream_task: asyncio.Task[None] | None = PrivateAttr(default=None)
    _conf: ConfHandle[GuideCameraConf] = PrivateAttr()
    _cal: ConfHandle[GuideCameraCal] = PrivateAttr()
    _cooler_cache: tuple[float, Cooler] | None = PrivateAttr(default=None)
    _life: Lifecycle = PrivateAttr()
    _last_calibration_id: int | None = PrivateAttr(default=None)

    # -- hardware hooks -----------------------------------------------------------------

    @property
    @abc.abstractmethod
    def sensor_size(self) -> tuple[int, int]:
        """(width, height) of the full sensor in pixels, unbinned."""

    @abc.abstractmethod
    def _expose(self, settings: CaptureSettings) -> np.ndarray:
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

    def model_post_init(self, __context) -> None:
        self._conf = ConfHandle(COMPONENT, "conf", GuideCameraConf)
        self._cal = ConfHandle(COMPONENT, "cal", GuideCameraCal)
        self._conf.before_change(self._accept)
        self._life = Lifecycle(
            COMPONENT,
            lambda: self.activities,
            startup=self.apply_conf,
            shutdown=self._deactivate,
            halt=self.stop,
        )

    # -- lifecycle ----------------------------------------------------------------------

    @property
    def life(self) -> Lifecycle:
        """Startup, shutdown, ``opstate``, ``operational`` and ``why_not_operational``."""
        return self._life

    async def startup(self) -> LifecycleStatus:
        """Put the stored settings on the hardware (the cooler starts) and become operational."""
        return await self._life.startup()

    async def shutdown(self, *, force: bool = False) -> LifecycleStatus:
        """Stop, and switch the cooler off. The saved settings are not touched."""
        return await self._life.shutdown(force=force)

    def _deactivate(self) -> None:
        if self.cooler_supported:
            self._write_cooler(False, self._conf.current.cooler_setpoint_c)
        self._cooler_cache = None

    @property
    def conf(self) -> ConfHandle[GuideCameraConf]:
        """The operational settings: ``camera.conf.current``, ``.snapshot()``, ``.patch(...)``."""
        return self._conf

    @property
    def cal(self) -> ConfHandle[GuideCameraCal]:
        """The calibration values; ``camera.cal.require("plate_scale_arcsec_px")``."""
        return self._cal

    @property
    def settings(self) -> GuideCameraConf:
        """The settings as they are now (a frozen snapshot)."""
        return self._conf.current

    def apply_conf(self) -> None:
        """Put the stored (or default) settings on the hardware: once at startup."""
        c = self._conf.current
        self._check_roi(c)
        if self.cooler_supported:
            self._write_cooler(c.cooler_enabled, c.cooler_setpoint_c)
        self._cooler_cache = None

    def _accept(self, old: GuideCameraConf, new: GuideCameraConf, changed: list[str]) -> None:
        """Before a change is recorded: refuse what this camera cannot take, and put a cooler
        change on the hardware, so a failed write is never recorded as the setting."""
        self._check_roi(new)
        if "cooler_enabled" in changed or "cooler_setpoint_c" in changed:
            if not self.cooler_supported:
                raise SettingError("this camera has no cooler")
            if self._life.opstate is OpState.startedup:  # an inactive camera is not touched;
                self._write_cooler(new.cooler_enabled, new.cooler_setpoint_c)  # startup applies it
            self._cooler_cache = None
            log.info(
                "cooler %s, setpoint %.1f °C",
                "on" if new.cooler_enabled else "off",
                new.cooler_setpoint_c,
            )

    def _check_roi(self, conf: GuideCameraConf) -> None:
        for name in ("single_frame", "video"):
            self._check_roi_of(getattr(conf, name), name)

    def _check_roi_of(self, s: CaptureSettings, name: str) -> None:
        if s.roi is None:
            return
        width, height = self.sensor_size
        if s.roi.x + s.roi.width > width or s.roi.y + s.roi.height > height:
            raise SettingError(f"{name} roi: outside the {width}x{height} sensor")
        if s.roi.width % s.binning or s.roi.height % s.binning:
            raise SettingError(
                f"{name} roi: width and height must be multiples of binning {s.binning}"
            )

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
        c = self._conf.current
        enabled, setpoint = c.cooler_enabled, c.cooler_setpoint_c
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

    def subscribe(self) -> asyncio.Queue[Frame]:
        """A queue that always holds the newest frame only: a slow viewer skips frames, it
        never builds a backlog and never slows the camera."""
        q: asyncio.Queue[Frame] = asyncio.Queue(maxsize=1)
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[Frame]) -> None:
        self._subscribers.discard(q)

    def _remember(self, frame: Frame) -> None:
        self._frames[frame.meta.id] = frame
        while len(self._frames) > KEPT_FRAMES:
            self._frames.popitem(last=False)

    def _publish(self, frame: Frame) -> None:
        self._latest = frame
        self._remember(frame)
        for q in self._subscribers:
            if q.full():
                q.get_nowait()  # drop the stale one
            q.put_nowait(frame)

    async def single_frame(self) -> Frame:
        """A fresh exposure with the ``single_frame`` settings. Raises :class:`CameraBusy` while
        streaming, :class:`ExposureAborted` if :meth:`abort` cut it short, and ``409
        not_operational`` until the camera is started up."""
        self._life.require_operational()
        if self._streaming:
            raise CameraBusy("the guide camera is streaming")
        return await self._take("single_frame")

    async def _take(
        self,
        mode: FrameMode,
        settings: CaptureSettings | None = None,
        *,
        publish: bool = True,
    ) -> Frame:
        """One exposure. By default with the ``mode`` settings group, shown to every viewer; a
        calibration frame brings its own settings and is only returned (and filed by the caller)."""
        if self._exposing:
            raise CameraBusy("the guide camera is already exposing")
        s: CaptureSettings = settings or getattr(self._conf.snapshot(), mode)
        self._exposing = True
        self.activities.start(GuideCameraActivity.Exposing, details=[f"{s.exposure_s:g} s"])
        try:
            pixels = await asyncio.to_thread(self._expose, s)
        finally:
            self._exposing = False
            self.activities.end(GuideCameraActivity.Exposing)
            self.activities.end(GuideCameraActivity.Cancelling)
        frame = await asyncio.to_thread(self._make_frame, pixels, s, mode)
        if publish:
            self._publish(frame)
        return frame

    async def calibration_frame(self, kind: CalibrationKind, settings: CaptureSettings) -> Frame:
        """A dark or bias frame with exactly ``settings``. The caller has blocked the light path
        (a dark) and files the frame; it is not shown to the live viewers and not kept among the recent ones.
        Raises :class:`CameraBusy` while streaming, :class:`SettingError` for a window outside the
        sensor, :class:`ExposureAborted` if :meth:`abort` cut it short, ``409 not_operational``
        until the camera is started up."""
        self._life.require_operational()
        if self._streaming:
            raise CameraBusy("the guide camera is streaming")
        self._check_roi_of(settings, kind)
        frame = await self._take(kind, settings, publish=False)
        self._remember(frame)  # downloadable like any recent frame, until the run files it
        self._last_calibration_id = frame.meta.id
        return frame

    def abort(self) -> None:
        """Cut a running exposure short. A no-op when nothing is exposing."""
        if self._exposing:
            if not self.activities.is_active(GuideCameraActivity.Cancelling):
                self.activities.start(GuideCameraActivity.Cancelling)
            self._abort_exposure()

    def _make_frame(self, pixels: np.ndarray, s: CaptureSettings, mode: FrameMode) -> Frame:
        self._seq += 1
        h, w = pixels.shape
        meta = FrameMeta(
            id=self._seq,
            mode=mode,
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
        self._life.require_operational()
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
                await self._take("video")
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
            settings=self._conf.current,
            conf_version=self._conf.version,
            cal_version=self._cal.version,
            cooler=self.cooler(),
            last_frame_id=self._latest.meta.id if self._latest else None,
            last_calibration_frame_id=self._last_calibration_id,
            lifecycle=self._life.status(),
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
