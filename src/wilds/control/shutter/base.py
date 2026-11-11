"""The shutter's abstract base (plan §3.17).

WILDS has one software-controlled shutter in the guide camera's light path (nothing the camera
triggers): it is closed to take darks and bias frames, and otherwise open. The base holds
everything shared: the ``conf`` settings slice, the lifecycle, the activities and the state; a
subclass supplies only the hardware hooks, so the simulator exercises all of it.

Closed is the safe state, so a shutdown leaves the shutter closed. A procedure that closes it
puts it back as it found it (``async with shutter.closed(): ...``), whatever happens inside.

Hooks are blocking and are called from a worker thread. Nothing here imports FastAPI; the router
turns :class:`ShutterBusy` into the standard error body.
"""

import abc
import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator
from enum import Enum
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, PrivateAttr

from ..activities import Activities, registry
from ..conf import ConfHandle, ConfModel, setting
from ..lifecycle import Lifecycle, LifecycleStatus

log = logging.getLogger(__name__)

COMPONENT = "shutter"

ShutterState = Literal["open", "closed", "moving", "unknown"]


class ShutterActivity(Enum):
    Opening = 1
    Closing = 2
    StartingUp = 3
    ShuttingDown = 4


class ShutterBusy(RuntimeError):
    """The shutter is moving, or a refused request for it (HTTP 409, ``busy``)."""


class ShutterConf(ConfModel):
    """The operational settings (``shutter.conf``)."""

    settle_s: float = setting(
        0.1,
        unit="s",
        label="Settle time",
        help="Wait after the blade stops before the shutter counts as open or closed "
        "(vibration, or the last of the light)",
        ge=0,
        le=5,
    )


class ShutterStatus(BaseModel):
    kind: str
    state: ShutterState
    settle_s: float
    lifecycle: LifecycleStatus


class ShutterBase(BaseModel, abc.ABC):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    kind: ClassVar[str]

    _conf: ConfHandle[ShutterConf] = PrivateAttr()
    _life: Lifecycle = PrivateAttr()
    _state: ShutterState = PrivateAttr(default="unknown")
    _moving: bool = PrivateAttr(default=False)

    # -- hardware hooks -----------------------------------------------------------------

    @abc.abstractmethod
    def _read_state(self) -> Literal["open", "closed", "unknown"]:
        """Where the blade is now, from the hardware."""

    @abc.abstractmethod
    def _move(self, open: bool) -> None:
        """Open or close the shutter (blocking); return when the blade has stopped."""

    def _close(self) -> None:  # noqa: B027 - optional hook
        """Release the hardware."""

    # -- activities and lifecycle -------------------------------------------------------

    @property
    def activities(self) -> Activities:
        try:
            return registry.get(COMPONENT)
        except KeyError:
            return registry.register(COMPONENT, ShutterActivity)

    def model_post_init(self, __context) -> None:
        self._conf = ConfHandle(COMPONENT, "conf", ShutterConf)
        self._life = Lifecycle(
            COMPONENT,
            lambda: self.activities,
            startup=self._find_state,
            shutdown=self._shut,
            why_not=self._why_not,
        )

    @property
    def life(self) -> Lifecycle:
        return self._life

    async def startup(self) -> LifecycleStatus:
        """Find out where the blade is. It is not moved: a control restart must not click it."""
        return await self._life.startup()

    async def shutdown(self, *, force: bool = False) -> LifecycleStatus:
        """Close the shutter (the safe state)."""
        return await self._life.shutdown(force=force)

    def _find_state(self) -> None:
        self._state = self._read_state()

    def _shut(self) -> None:
        self._move_blade(open=False)

    def _why_not(self) -> list[str]:
        return ["shutter state unknown"] if self.state == "unknown" else []

    # -- settings and state -------------------------------------------------------------

    @property
    def conf(self) -> ConfHandle[ShutterConf]:
        return self._conf

    @property
    def state(self) -> ShutterState:
        return "moving" if self._moving else self._state

    # -- moving -------------------------------------------------------------------------

    async def open(self) -> ShutterState:
        return await self._go(open=True)

    async def close(self) -> ShutterState:
        return await self._go(open=False)

    @contextlib.asynccontextmanager
    async def closed(self) -> AsyncIterator[None]:
        """Hold the shutter closed for a block and put it back as it was found, however the
        block ends. Raises :class:`ShutterBusy` if it is moving."""
        before = self.state
        await self.close()
        try:
            yield
        finally:
            if before == "open":
                await self.open()

    async def _go(self, *, open: bool) -> ShutterState:
        self._life.require_operational()
        if self._moving:
            raise ShutterBusy("the shutter is moving")
        activity = ShutterActivity.Opening if open else ShutterActivity.Closing
        self._moving = True
        self.activities.start(activity)
        try:
            await asyncio.to_thread(self._move_blade, open)
        finally:
            self._moving = False
            self.activities.end(activity)
        return self._state

    def _move_blade(self, open: bool) -> None:
        want = "open" if open else "closed"
        if self._state == want:
            return
        self._state = "unknown"  # until the move has finished
        self._move(open)
        self._state = self._read_state()
        if self._state != want:
            raise ShutterBusy(f"the shutter did not {'open' if open else 'close'}")
        log.info("shutter %s", want)

    # -- status -------------------------------------------------------------------------

    def status(self) -> ShutterStatus:
        return ShutterStatus(
            kind=self.kind,
            state=self.state,
            settle_s=self._conf.current.settle_s,
            lifecycle=self._life.status(),
        )

    async def close_hardware(self) -> None:
        self._close()
