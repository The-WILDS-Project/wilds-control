"""The instrument cover's abstract base (plan §3.17).

The LDT instrument cube has a rolling opaque cloth cover at its entrance aperture. Closed, it
blocks the light to WILDS, which is how the guide camera gets its dark and bias frames (it has
no shutter of its own: the shutter is for the science cameras). The Telescope Operator
normally works it; LDT agreed that the instrument operator may, so long as the TO is told first
(the GUI reminds them).

The state is ``open``, ``closed``, ``opening``, ``closing`` or ``unknown`` (nobody has moved it
since wilds-control started and nothing senses it). A subclass supplies how a command is sent
and how long the cover takes; if the hardware ever reports its state, ``_sense`` is the hook for
it. The base owns the state, the lifecycle and the activity. Nothing here imports FastAPI; the
router turns :class:`CoverBusy` into the standard error.

The cover is never moved by a startup or a shutdown: a cover closed on a daytime instrument is
not ours to open.
"""

import abc
import asyncio
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


class CoverActivity(Enum):
    Opening = 1
    Closing = 2
    StartingUp = 3
    ShuttingDown = 4


class CoverBusy(RuntimeError):
    """The cover is moving (HTTP 409, ``busy``)."""


class CoverError(RuntimeError):
    """The command could not be sent (HTTP 502)."""


class CoverStatus(BaseModel):
    kind: str
    state: CoverState
    since_ms: int  # when the state last changed, ms since the Unix epoch, UTC
    moving: bool
    travel_s: float  # how long a move is assumed to take, when nothing senses it
    sensed: bool  # the state comes from a sensor, not only from the commands sent
    lifecycle: LifecycleStatus


def _now_ms() -> int:
    return time.time_ns() // 1_000_000


class CoverBase(BaseModel, abc.ABC):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    kind: ClassVar[str]
    sensed: ClassVar[bool] = False
    start_state: ClassVar[CoverState] = "unknown"  # what the cover is taken to be at startup
    travel_s: float = 5.0

    _life: Lifecycle = PrivateAttr()
    _state: CoverState = PrivateAttr(default="unknown")
    _since: int = PrivateAttr(default_factory=_now_ms)
    _task: asyncio.Task | None = PrivateAttr(default=None)

    # -- hooks ----------------------------------------------------------------------------

    @abc.abstractmethod
    async def _send(self, close: bool) -> None:
        """Send the command to move the cover (close or open). Return once it is sent, not once
        the cover has arrived. Raise to say it could not be sent."""

    def _sense(self) -> CoverState | None:
        """The state from a sensor, or ``None`` when nothing senses it (the default)."""
        return None

    async def _close_hardware(self) -> None:
        """Release the connection."""

    # -- lifecycle ------------------------------------------------------------------------

    @property
    def activities(self) -> Activities:
        try:
            return registry.get(COMPONENT)
        except KeyError:
            return registry.register(COMPONENT, CoverActivity)

    def model_post_init(self, __context) -> None:
        self._life = Lifecycle(COMPONENT, lambda: self.activities, startup=self._find_state)
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
        self._go(moving)  # before the send: the command may take effect even if the reply is lost
        activity = CoverActivity.Closing if close else CoverActivity.Opening
        try:
            await self._send(close)
        except Exception as e:
            self._go("unknown")  # a command that may or may not have gone out: nobody knows
            log.exception("cover: could not send %s", moving)
            raise CoverError(f"could not send the command to the cover: {e}") from e
        self.activities.start(activity)
        self._task = asyncio.create_task(self._arrive(goal, activity), name=f"cover-{goal}")
        return self.status()

    async def _arrive(self, goal: CoverState, activity: CoverActivity) -> None:
        try:
            await asyncio.sleep(self.travel_s)
            self._go(self._sense() or goal)
        finally:
            self._task = None
            self.activities.end(activity)

    # -- status ---------------------------------------------------------------------------

    def status(self) -> CoverStatus:
        return CoverStatus(
            kind=self.kind,
            state=self._state,
            since_ms=self._since,
            moving=self.moving,
            travel_s=self.travel_s,
            sensed=self.sensed,
            lifecycle=self._life.status(),
        )

    async def close_hardware(self) -> None:
        if (task := self._task) is not None:
            task.cancel()
        await self._close_hardware()
