"""The ADC's abstract base (WILDS optics report §4.1, mechanical report §4.3).

The atmospheric dispersion corrector is two pairs of Amici prisms, each turned by its own
endless rotary stage (a SmarAct SR-12012 on the MCS2). Together they make a dispersion of any
size up to ``dmax`` in any direction (``model.py``); the ADC is set so that it cancels the
atmosphere's: its size follows the zenith distance, its direction the parallactic angle seen
from the instrument (the parallactic angle minus the slit PA). The positions are of the two
pairs, in degrees.

The base holds everything shared: the settings (``adc.conf``) and calibration values
(``adc.cal``), the lifecycle, the activities, the moves, and **tracking**, a task that follows
the telescope: every ``update_s`` it works out where the pairs should be from the telescope's
zenith distance and parallactic angle, its rotator angle and the weather, and moves them when
they are off by more than the dead-band. Tracking is off at start-up (the first nights are
manual) and a hand move is refused while it is on. A subclass supplies only the hardware hooks,
so the simulator exercises all of it. Hooks are blocking and run in a worker thread. Nothing
here imports FastAPI; the router turns :class:`AdcBusy` and :class:`AdcRange` into the standard
errors.

The stages' zero points (where a stage reads when its pair is at the reference) and the offset
of the direction are calibration values: until the instrument is aligned they are 0.

The ADC is optional: it is a removable module and WILDS works without it (optics report §4.1). The
setting ``installed`` (default yes) says whether it is there. Not installed, it is not part of the
instrument: the lifecycle treats it as absent (never operational, not started up or shut down, and
ignored by the system's own status), every command is refused with ``409 not_installed``, tracking
ends, and the status says so. Anything that relies on the ADC must ask ``adc.installed`` first.
"""

import abc
import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, PrivateAttr

from ..activities import Activities, registry
from ..conf import ConfHandle, ConfModel, setting
from ..errors import ApiError
from ..lifecycle import Lifecycle, LifecycleStatus
from ..weather import Weather, get_weather
from . import model

log = logging.getLogger(__name__)

COMPONENT = "adc"

TrackingState = Literal["off", "waiting", "holding", "moving", "error"]


class AdcActivity(Enum):
    Moving = 1
    StartingUp = 2
    ShuttingDown = 3


class AdcBusy(RuntimeError):
    """The pairs are moving, tracking is on, or a refused request for them (HTTP 409, ``busy``)."""


class AdcRange(ValueError):
    """More dispersion than the ADC can make, or a position that is not an angle (HTTP 422)."""


class AdcConf(ConfModel):
    """The operational settings (``adc.conf``)."""

    installed: bool = setting(
        True,
        label="Installed",
        help="Whether the ADC is in the instrument. It is a removable module and WILDS works "
        "without it: switched off, its controls are greyed out, tracking ends, and WILDS "
        "starts up and counts as operational without it",
    )

    dmax_arcsec: float = setting(
        6.0,
        unit="arcsec",
        label="Largest dispersion",
        help="The most dispersion the two pairs can make together between the band edges "
        "(optics report Appendix A.5: 6.0 arcsec, which matches the sea-level atmosphere at "
        "zenith distance 60 degrees)",
        gt=0,
        le=20,
    )
    deadband_deg: float = setting(
        0.25,
        unit="°",
        label="Dead-band",
        help="Tracking moves the pairs only when either is further than this from where it "
        "should be (a 1 degree error costs 0.1 arcsec of residual at the zenith, none at 60)",
        ge=0,
        le=5,
    )
    update_s: float = setting(
        5.0,
        unit="s",
        label="Tracking interval",
        help="How often tracking looks at the telescope",
        ge=0.5,
        le=300,
    )
    short_nm: float = setting(
        305.0,
        unit="nm",
        label="Shortest wavelength",
        help="The blue edge to correct",
        ge=300,
        le=500,
    )
    long_nm: float = setting(
        915.0,
        unit="nm",
        label="Longest wavelength",
        help="The red edge to correct",
        ge=500,
        le=1000,
    )
    settle_s: float = setting(
        0.1,
        unit="s",
        label="Settle time",
        help="Wait after the stages stop before the positions count as reached",
        ge=0,
        le=5,
    )


class AdcCal(ConfModel):
    """The calibration values (``adc.cal``): whoever holds control may change them."""

    zero1_deg: float = setting(
        0.0,
        unit="°",
        label="Pair 1 zero",
        help="What stage 1 reads when pair 1 is at its reference position",
        ge=-360,
        le=360,
    )
    zero2_deg: float = setting(
        0.0,
        unit="°",
        label="Pair 2 zero",
        help="What stage 2 reads when pair 2 is at its reference position",
        ge=-360,
        le=360,
    )
    direction_offset_deg: float = setting(
        0.0,
        unit="°",
        label="Direction offset",
        help="Added to (parallactic angle - slit PA) to give the direction the pairs "
        "must make their dispersion in: how the prisms sit in the instrument",
        ge=-360,
        le=360,
    )


@dataclass(frozen=True)
class SkyState:
    """What the telescope says about its target, as far as the ADC needs."""

    zenith_distance_deg: float
    parallactic_angle_deg: float
    rot_pa_deg: float  # the slit's sky PA now


class AdcTarget(BaseModel):
    """What tracking worked out at its last look."""

    zenith_distance_deg: float
    parallactic_angle_deg: float
    rot_pa_deg: float
    atmosphere_arcsec: float  # the atmosphere's dispersion between the band edges
    dispersion_arcsec: float  # what the pairs are to make (at most dmax)
    direction_deg: float
    pair1_deg: float
    pair2_deg: float
    saturated: bool  # the atmosphere disperses more than the ADC can cancel
    updated_ms: int


class AdcStatus(BaseModel):
    kind: str
    installed: bool  # False: not in the instrument; everything else is as it was, and unused
    # Where the pairs are, live: during a move too (the hardware's own reading as it turns). None
    # only when not known (not started up, or the hardware cannot say).
    pair1_deg: float | None
    pair2_deg: float | None
    goal1_deg: float | None  # where a move in progress is going (the pairs' angles); None at rest
    goal2_deg: float | None
    stage1_deg: (
        float | None
    )  # what the stages themselves read (the pair's angle plus its zero point)
    stage2_deg: float | None
    zero1_deg: float  # the calibrated zero points
    zero2_deg: float
    direction_offset_deg: float
    dispersion_arcsec: float | None  # what the pairs make now
    direction_deg: float | None
    moving: bool
    referenced: bool
    tracking: bool
    tracking_state: TrackingState
    tracking_note: str | None  # why it waits, or what failed
    target: AdcTarget | None
    weather: Weather | None  # as tracking last used it
    dmax_arcsec: float
    deadband_deg: float
    update_s: float
    lifecycle: LifecycleStatus


def _off_by(a: float, b: float) -> float:
    """How far apart two angles are, 0 to 180."""
    return abs((a - b + 180.0) % 360.0 - 180.0)


def _now_ms() -> int:
    return time.time_ns() // 1_000_000


class AdcBase(BaseModel, abc.ABC):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    kind: ClassVar[str]

    # Where tracking looks: the app wires these to the telescope and the weather; tests inject.
    sky: Callable[[], SkyState | None] = lambda: None
    weather: Callable[[], Weather] = get_weather

    _conf: ConfHandle[AdcConf] = PrivateAttr()
    _cal: ConfHandle[AdcCal] = PrivateAttr()
    _life: Lifecycle = PrivateAttr()
    _raw: tuple[float, float] | None = PrivateAttr(default=None)  # as the stages read, at rest
    # Where a move in progress is going, as the stages read (None at rest)
    _goal: tuple[float, float] | None = PrivateAttr(default=None)
    _referenced: bool = PrivateAttr(default=False)
    _moving: bool = PrivateAttr(default=False)
    _task: asyncio.Task | None = PrivateAttr(default=None)
    _state: TrackingState = PrivateAttr(default="off")
    _note: str | None = PrivateAttr(default=None)
    _target: AdcTarget | None = PrivateAttr(default=None)
    _weather: Weather | None = PrivateAttr(default=None)

    # -- hardware hooks (raw stage angles, degrees) ---------------------------------------

    @abc.abstractmethod
    def _reference(self) -> None:
        """Find the stages' reference (blocking). How is still to be agreed with SmarAct."""

    @abc.abstractmethod
    def _read_angles(self) -> tuple[float, float] | None:
        """Where the two stages are now, from the hardware. Also asked for while they move (the
        status shows the pairs turning), from another thread than the move's: it must be quick
        and must not wait for the move."""

    @abc.abstractmethod
    def _move(self, raw1: float, raw2: float) -> None:
        """Turn the stages to these readings (blocking); return when they have stopped."""

    def _close(self) -> None:  # noqa: B027 - optional hook
        """Release the hardware."""

    # -- activities and lifecycle -------------------------------------------------------

    @property
    def activities(self) -> Activities:
        try:
            return registry.get(COMPONENT)
        except KeyError:
            return registry.register(COMPONENT, AdcActivity)

    def model_post_init(self, __context) -> None:
        self._conf = ConfHandle(COMPONENT, "conf", AdcConf)
        self._cal = ConfHandle(COMPONENT, "cal", AdcCal)
        self._life = Lifecycle(
            COMPONENT,
            lambda: self.activities,
            startup=self._find_angles,
            ashutdown=self.stop_tracking,
            why_not=self._why_not,
            absent=lambda: not self._conf.current.installed,
        )

    @property
    def life(self) -> Lifecycle:
        return self._life

    @property
    def conf(self) -> ConfHandle[AdcConf]:
        return self._conf

    @property
    def cal(self) -> ConfHandle[AdcCal]:
        return self._cal

    async def startup(self) -> LifecycleStatus:
        """Reference the stages and read where they are. Tracking stays off."""
        return await self._life.startup()

    async def shutdown(self, *, force: bool = False) -> LifecycleStatus:
        """Stop tracking and leave the pairs where they are."""
        return await self._life.shutdown(force=force)

    def _find_angles(self) -> None:
        self._reference()
        self._referenced = True
        self._raw = self._read_angles()

    def _why_not(self) -> list[str]:
        out = []
        if not self._referenced:
            out.append("ADC not referenced")
        elif self._raw is None and not self._moving:
            out.append("ADC positions unknown")
        return out

    # -- raw stage readings and the angles the user sees -----------------------------------

    def _to_user(self, raw: tuple[float, float] | None) -> tuple[float, float] | None:
        if raw is None:
            return None
        c = self._cal.current
        return ((raw[0] - c.zero1_deg) % 360.0, (raw[1] - c.zero2_deg) % 360.0)

    def _to_raw(self, a1: float, a2: float) -> tuple[float, float]:
        c = self._cal.current
        return ((a1 + c.zero1_deg) % 360.0, (a2 + c.zero2_deg) % 360.0)

    # -- state --------------------------------------------------------------------------

    @property
    def installed(self) -> bool:
        return self._conf.current.installed

    def _require_installed(self) -> None:
        """``409 not_installed`` when the ADC is not in the instrument."""
        if not self.installed:
            raise ApiError(
                409, "not_installed", "the ADC is not installed", {"component": COMPONENT}
            )

    @property
    def angles(self) -> tuple[float, float] | None:
        return None if self._moving else self._to_user(self._raw)

    @property
    def moving(self) -> bool:
        return self._moving

    @property
    def tracking(self) -> bool:
        return self._task is not None and self.installed

    # -- moving by hand -----------------------------------------------------------------

    async def move_to(self, pair1_deg: float, pair2_deg: float) -> tuple[float, float]:
        """Turn the pairs to these positions (degrees, any value: it is taken round the
        circle). Refused while tracking is on or the pairs are moving."""
        self._require_installed()
        self._life.require_operational()
        if self.tracking:
            raise AdcBusy("tracking is on: switch it off to move the pairs by hand")
        return await self._go(pair1_deg % 360.0, pair2_deg % 360.0, why="by hand")

    async def move_dispersion(
        self, dispersion_arcsec: float, direction_deg: float
    ) -> tuple[float, float]:
        """Make a dispersion of this size (arcsec) in this direction (degrees). ``AdcRange`` if it
        is more than the ADC can make."""
        self._require_installed()
        self._life.require_operational()
        dmax = self._conf.current.dmax_arcsec
        if dispersion_arcsec < 0 or dispersion_arcsec > dmax:
            raise AdcRange(
                f"the ADC makes 0 to {dmax:g} arcsec of dispersion, not {dispersion_arcsec:g}"
            )
        if self.tracking:
            raise AdcBusy("tracking is on: switch it off to move the pairs by hand")
        p = model.pairs_for(dispersion_arcsec, direction_deg, dmax)
        return await self._go(p.pair1_deg, p.pair2_deg, why="by hand")

    async def jog(
        self, stage: int, *, to_deg: float | None = None, by_deg: float | None = None
    ) -> tuple[float, float]:
        """Turn one stage, in the stage's own coordinates (what it reads, not the pair's angle, so
        the zero points play no part), to a reading or by an amount; the other stays where it is.
        This is how a pair is put at its reference mark. Returns what the two stages read. Refused
        while tracking is on or the stages move; ``AdcRange`` for a bad stage or amount."""
        if stage not in (1, 2):
            raise AdcRange(f"the ADC has stages 1 and 2, not {stage}")
        if (to_deg is None) == (by_deg is None):
            raise AdcRange("give either to_deg or by_deg, not both and not neither")
        if by_deg is not None and abs(by_deg) > 360.0:
            raise AdcRange(f"a jog is at most 360 degrees, not {by_deg:g}")
        self._require_installed()
        self._life.require_operational()
        if self.tracking:
            raise AdcBusy("tracking is on: switch it off to jog a stage")
        if self._moving:
            raise AdcBusy("the ADC is moving")
        raw = self._raw
        if raw is None:
            raise AdcBusy("the stages' readings are not known")
        goal = list(raw)
        goal[stage - 1] = (to_deg if to_deg is not None else raw[stage - 1] + by_deg) % 360.0  # type: ignore[operator]
        await self._go_raw(goal[0], goal[1], why=f"jog stage {stage}")
        return self._raw  # type: ignore[return-value]

    async def _go(self, a1: float, a2: float, *, why: str) -> tuple[float, float]:
        await self._go_raw(*self._to_raw(a1, a2), why=why)
        return self._to_user(self._raw)  # type: ignore[return-value]

    async def _go_raw(self, r1: float, r2: float, *, why: str) -> None:
        if self._moving:
            raise AdcBusy("the ADC is moving")
        self._moving = True
        self._goal = (r1, r2)
        self.activities.start(AdcActivity.Moving, details=[f"{r1:.1f}°, {r2:.1f}° ({why})"])
        try:
            await asyncio.to_thread(self._turn, r1, r2)
        finally:
            self._moving = False
            self._goal = None
            self.activities.end(AdcActivity.Moving)

    def _turn(self, r1: float, r2: float) -> None:
        """Turn the stages to these readings (blocking) and check they got there."""
        self._raw = None  # until the move has finished
        self._move(r1, r2)
        self._raw = self._read_angles()
        if self._raw is None or max(_off_by(self._raw[0], r1), _off_by(self._raw[1], r2)) > 0.05:
            raise AdcBusy(f"the ADC did not reach {r1:.1f}°, {r2:.1f}°")
        log.info("adc stages at %.2f°, %.2f°", *self._raw)

    # -- tracking -----------------------------------------------------------------------

    async def set_tracking(self, enabled: bool) -> None:
        if enabled:
            await self.start_tracking()
        else:
            await self.stop_tracking()

    async def start_tracking(self) -> None:
        self._require_installed()
        self._life.require_operational()
        if self._task is None:
            self._state, self._note = "waiting", None
            self._task = asyncio.create_task(self._track(), name="adc-tracking")
            log.info("adc tracking on")

    async def stop_tracking(self) -> None:
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            log.info("adc tracking off")
        self._state, self._note = "off", None

    async def _track(self) -> None:
        while True:
            if not self.installed:  # taken out of the instrument while tracking: stop following
                self._task, self._state, self._note = None, "off", "the ADC is not installed"
                return
            try:
                await self.track_once()
            except asyncio.CancelledError:
                raise
            except Exception as e:  # keep following: the next look may be fine
                log.exception("adc tracking")
                self._state, self._note = "error", str(e)
            await asyncio.sleep(self._conf.current.update_s)

    async def track_once(self) -> None:
        """One look at the telescope, and a move if the pairs are off by more than the dead-band."""
        if not self.installed:
            return
        sky = self.sky()
        if sky is None:
            self._state, self._note = "waiting", "the telescope has no target"
            return
        c = self._conf.current
        weather = self.weather()
        atmosphere = model.atmospheric_dispersion_arcsec(
            sky.zenith_distance_deg, weather, c.short_nm, c.long_nm
        )
        direction = (
            sky.parallactic_angle_deg - sky.rot_pa_deg + self._cal.current.direction_offset_deg
        )
        p = model.pairs_for(atmosphere, direction, c.dmax_arcsec)
        self._weather = weather
        self._target = AdcTarget(
            zenith_distance_deg=sky.zenith_distance_deg,
            parallactic_angle_deg=sky.parallactic_angle_deg,
            rot_pa_deg=sky.rot_pa_deg,
            atmosphere_arcsec=atmosphere,
            dispersion_arcsec=p.dispersion_arcsec,
            direction_deg=p.direction_deg,
            pair1_deg=p.pair1_deg,
            pair2_deg=p.pair2_deg,
            saturated=p.saturated,
            updated_ms=_now_ms(),
        )
        self._note = (
            f"the atmosphere disperses {atmosphere:.1f} arcsec, more than the {c.dmax_arcsec:g} the ADC makes"
            if p.saturated
            else None
        )
        have = self.angles
        if self._moving:
            self._state = "moving"
            return
        if (
            have is not None
            and max(_off_by(have[0], p.pair1_deg), _off_by(have[1], p.pair2_deg)) <= c.deadband_deg
        ):
            self._state = "holding"
            return
        self._state = "moving"
        try:
            await self._go(p.pair1_deg, p.pair2_deg, why="tracking")
        finally:
            self._state = "holding" if self._task is not None else "off"

    # -- status -------------------------------------------------------------------------

    def status(self) -> AdcStatus:
        c = self._conf.current
        # live: while the stages turn, what the hardware reads as it turns
        raw = self._read_angles() if self._moving else self._raw
        a = self._to_user(raw)
        made = None if a is None else model.delivered(*a, c.dmax_arcsec)
        goal = None if self._goal is None else self._to_user(self._goal)
        cal = self._cal.current
        return AdcStatus(
            kind=self.kind,
            installed=self.installed,
            pair1_deg=None if a is None else a[0],
            pair2_deg=None if a is None else a[1],
            goal1_deg=None if goal is None else goal[0],
            goal2_deg=None if goal is None else goal[1],
            stage1_deg=None if raw is None else raw[0],
            stage2_deg=None if raw is None else raw[1],
            zero1_deg=cal.zero1_deg,
            zero2_deg=cal.zero2_deg,
            direction_offset_deg=cal.direction_offset_deg,
            dispersion_arcsec=None if made is None else made[0],
            direction_deg=None if made is None else made[1],
            moving=self._moving,
            referenced=self._referenced,
            tracking=self.tracking,
            tracking_state=self._state if self.installed else "off",
            tracking_note=self._note if self.installed else "the ADC is not installed",
            target=self._target if self.tracking else None,
            weather=self._weather if self.tracking else None,
            dmax_arcsec=c.dmax_arcsec,
            deadband_deg=c.deadband_deg,
            update_s=c.update_s,
            lifecycle=self._life.status(),
        )

    async def close_hardware(self) -> None:
        await self.stop_tracking()
        self._close()
