"""The instrument cover's abstract base (plan §3.17).

The LDT instrument cube has a rolling opaque cloth cover at its entrance aperture. Closed, it
blocks the light to WILDS, which is how the guide camera gets its dark and bias frames (it has
no shutter of its own: the shutter is for the science cameras). The Telescope Operator
normally works it; LDT agreed that the instrument operator may, so long as the TO is told first
(the GUI reminds them).

The state is ``open``, ``closed``, ``opening``, ``closing`` or ``unknown`` (nobody has moved it
since wilds-control started and nothing has reported it). The instrument cube reports its stages
on ``instrumentCube.loisTelemetry`` (about once a second: the cover as OPEN or CLOSED, and the
fold mirrors as HOME or EXTENDED, with positions in mm). A subclass passes each report to
:meth:`feed`; while reports keep coming they are the truth: they settle a move as soon as the
cover arrives, show a cover moved by somebody else, and catch a move that did not happen. Without
them (a report not seen for :data:`STALE_S`) the state follows the commands, after the assumed
travel time. A subclass supplies how a command is sent and how it listens. The base owns the
state, the lifecycle and the activity. Nothing here imports FastAPI; the router turns
:class:`CoverBusy` into the standard error.

The cover is never moved by a startup or a shutdown: a cover closed on a daytime instrument is
not ours to open.
"""

import abc
import asyncio
import contextlib
import logging
import time
from enum import Enum
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, PrivateAttr

from ..activities import Activities, registry
from ..lifecycle import Lifecycle, LifecycleStatus

log = logging.getLogger(__name__)

COMPONENT = "cover"

CoverState = Literal["open", "closed", "opening", "closing", "unknown"]

# A report older than this is no longer the truth (the cube sends about one a second).
STALE_S = 10.0
# While reports are coming the cover has this long, beyond the assumed travel time, to be seen
# where it was sent before the move is taken to have failed.
SENSE_GRACE_S = 2.0

# WILDS is planned for port C of the instrument cube: fold mirror C feeds it, EXTENDED into the
# beam. (An assumption, to be confirmed with LDT; ``to_wilds`` follows it.)
WILDS_MIRROR = "C"
WILDS_FEEDING = "EXTENDED"


class CoverActivity(Enum):
    Opening = 1
    Closing = 2
    StartingUp = 3
    ShuttingDown = 4


class CoverBusy(RuntimeError):
    """The cover is moving (HTTP 409, ``busy``)."""


class CoverError(RuntimeError):
    """The command could not be sent (HTTP 502)."""


class CubeReport(BaseModel):
    """One telemetry packet from the instrument cube, as far as WILDS reads it. States are the
    wire's upper-case strings; positions are mm."""

    cover_state: str | None = None  # OPEN or CLOSED
    cover_position_mm: float | None = None
    fm_c_state: str | None = None  # HOME or EXTENDED
    fm_c_position_mm: float | None = None


class CubeStatus(BaseModel):
    """What the cube last said, for the operator."""

    age_s: float  # how long ago
    cover_position_mm: float | None
    fold_mirror: str  # the one that feeds WILDS ("C")
    fold_mirror_state: str | None  # HOME or EXTENDED
    fold_mirror_position_mm: float | None
    to_wilds: bool | None  # that mirror is where it sends the beam to WILDS (None: not known)


class CoverStatus(BaseModel):
    kind: str
    state: CoverState
    since_ms: int  # when the state last changed, ms since the Unix epoch, UTC
    moving: bool
    travel_s: float  # how long a move is assumed to take
    sensed: bool  # reports from the cube are coming: the state is what it says, not only what was commanded
    cube: CubeStatus | None  # the last report (None: none seen yet)
    lifecycle: LifecycleStatus


def _now_ms() -> int:
    return time.time_ns() // 1_000_000


class CoverBase(BaseModel, abc.ABC):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    kind: ClassVar[str]
    start_state: ClassVar[CoverState] = "unknown"  # what the cover is taken to be at startup
    travel_s: float = 5.0

    _life: Lifecycle = PrivateAttr()
    _state: CoverState = PrivateAttr(default="unknown")
    _since: int = PrivateAttr(default_factory=_now_ms)
    _task: asyncio.Task | None = PrivateAttr(default=None)
    _goal: CoverState = PrivateAttr(default="unknown")  # where a move in progress is going
    _arrived: asyncio.Event | None = PrivateAttr(default=None)  # the cube says it is there
    _report: CubeReport | None = PrivateAttr(default=None)
    _report_t: float | None = PrivateAttr(default=None)  # monotonic time of the last report

    # -- hooks ----------------------------------------------------------------------------

    @abc.abstractmethod
    async def _send(self, close: bool) -> None:
        """Send the command to move the cover (close or open). Return once it is sent, not once
        the cover has arrived. Raise to say it could not be sent."""

    async def _start_feed(self) -> None:
        """Begin listening for the cube's reports and pass each to :meth:`feed`. Best effort:
        never raises (the cover works, on its commands, without them)."""

    async def _stop_feed(self) -> None:
        """Stop listening."""

    async def _close_hardware(self) -> None:
        """Release the connection."""

    # -- the cube's reports -----------------------------------------------------------------

    @property
    def sensed(self) -> bool:
        """Reports are coming (the last is under :data:`STALE_S` old)."""
        return self._report_t is not None and time.monotonic() - self._report_t <= STALE_S

    def _sense(self) -> CoverState | None:
        """The state the cube says, or ``None`` when it is not reporting (or says something else
        than OPEN or CLOSED)."""
        if not self.sensed or self._report is None:
            return None
        return {"OPEN": "open", "CLOSED": "closed"}.get((self._report.cover_state or "").upper())

    def feed(self, report: CubeReport) -> None:
        """A report from the cube. Not moving: the cover is where it says (somebody else may have
        moved it). Moving: it has arrived when it says it is where it was sent; otherwise it is
        still on its way, and the timer judges."""
        self._report, self._report_t = report, time.monotonic()
        seen = self._sense()
        if seen is None:
            return
        if self.moving:
            if seen == self._goal and self._arrived is not None:
                self._go(seen)
                self._arrived.set()  # arrived: the timer's wait is over
        elif self._task is None:
            self._go(seen)

    # -- lifecycle ------------------------------------------------------------------------

    @property
    def activities(self) -> Activities:
        try:
            return registry.get(COMPONENT)
        except KeyError:
            return registry.register(COMPONENT, CoverActivity)

    def model_post_init(self, __context) -> None:
        self._life = Lifecycle(
            COMPONENT,
            lambda: self.activities,
            startup=self._find_state,
            astartup=self._start_feed,
            ashutdown=self._stop_feed,
        )
        self._state = self.start_state

    @property
    def life(self) -> Lifecycle:
        return self._life

    async def startup(self) -> LifecycleStatus:
        """Find out where the cover is, if anything can tell. It is not moved."""
        return await self._life.startup()

    async def shutdown(self, *, force: bool = False) -> LifecycleStatus:
        """Leave the cover as it is."""
        return await self._life.shutdown(force=force)

    def _find_state(self) -> None:
        if self._task is None and (sensed := self._sense()) is not None:
            self._go(sensed)

    # -- state ----------------------------------------------------------------------------

    @property
    def state(self) -> CoverState:
        return self._state

    @property
    def moving(self) -> bool:
        return self._state in ("opening", "closing")

    @property
    def closed(self) -> bool:
        """Light is blocked: the cover is closed, and has stopped moving."""
        return self._state == "closed"

    def _go(self, state: CoverState) -> None:
        if state != self._state:
            log.info("cover: %s -> %s", self._state, state)
            self._state = state
            self._since = _now_ms()

    # -- moving ---------------------------------------------------------------------------

    async def close(self) -> CoverStatus:
        return await self._move(close=True)

    async def open(self) -> CoverStatus:
        return await self._move(close=False)

    async def _move(self, *, close: bool) -> CoverStatus:
        """Start the move and return at once: the state is ``closing``/``opening`` until the
        cover has had time to arrive. Asking for where it already is does nothing. Raises
        :class:`CoverBusy` while it moves and :class:`CoverError` if the command is not sent."""
        self._life.require_operational()
        goal: CoverState = "closed" if close else "open"
        if self.moving:
            raise CoverBusy(f"the cover is {self._state}")
        if self._state == goal:
            return self.status()
        moving: CoverState = "closing" if close else "opening"
        self._goal = goal
        self._arrived = asyncio.Event()
        self._go(moving)  # before the send: the command may take effect even if the reply is lost
        activity = CoverActivity.Closing if close else CoverActivity.Opening
        try:
            await self._send(close)
        except Exception as e:
            self._go("unknown")  # a command that may or may not have gone out: nobody knows
            log.exception("cover: could not send %s", moving)
            raise CoverError(f"could not send the command to the cover: {e}") from e
        self.activities.start(activity)
        self._task = asyncio.create_task(
            self._arrive(goal, activity, self._arrived), name=f"cover-{goal}"
        )
        return self.status()

    async def _arrive(
        self, goal: CoverState, activity: CoverActivity, arrived: asyncio.Event
    ) -> None:
        try:
            wait = self.travel_s + (SENSE_GRACE_S if self.sensed else 0.0)
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(arrived.wait(), wait)
            if (seen := self._sense()) is not None and seen != goal:
                log.warning("cover: sent to %s but the cube still says %s", goal, seen)
            self._go(self._sense() or goal)  # the cube's word if it has one, else the command's
        finally:
            if self._task is asyncio.current_task():  # not a newer move's
                self._task = None
            self.activities.end(activity)

    # -- status ---------------------------------------------------------------------------

    def _cube(self) -> CubeStatus | None:
        r, t = self._report, self._report_t
        if r is None or t is None:
            return None
        fm = (r.fm_c_state or "").upper() or None
        return CubeStatus(
            age_s=round(time.monotonic() - t, 1),
            cover_position_mm=r.cover_position_mm,
            fold_mirror=WILDS_MIRROR,
            fold_mirror_state=fm,
            fold_mirror_position_mm=r.fm_c_position_mm,
            to_wilds=None if fm is None else fm == WILDS_FEEDING,
        )

    def status(self) -> CoverStatus:
        return CoverStatus(
            kind=self.kind,
            state=self._state,
            since_ms=self._since,
            moving=self.moving,
            travel_s=self.travel_s,
            sensed=self.sensed,
            cube=self._cube(),
            lifecycle=self._life.status(),
        )

    async def close_hardware(self) -> None:
        if (task := self._task) is not None:
            task.cancel()
        await self._stop_feed()
        await self._close_hardware()
