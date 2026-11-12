"""The slit stage's abstract base (plan §3.3, §3.17).

The slit is a gap between two jaws that a stage opens and closes, symmetrically about a fixed
centre: the stage position is the gap's width in microns at the telescope's focal plane. The guide
camera looks at the sky reflected from the jaws, so the slit shows as a dark strip (see the
calibration). Two named positions are kept in ``conf``: ``closed`` (near the stage's lower
limit, no light reaches the spectrograph) and ``open`` (its full travel); any position in between
is allowed (the working slit is about 1 arcsec, ~125 µm).

The base holds everything shared (settings, lifecycle, activities, the position and what it is in
arcsec); a subclass supplies only the hardware hooks. Hooks are blocking, called from a worker
thread. Nothing here imports FastAPI; the router turns :class:`SlitBusy` into the standard error.

Closed is the safe state, so a shutdown closes the slit; start-up finds where it is and does not
move it.
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

COMPONENT = "slit"

Preset = Literal["open", "closed"]
POSITION_TOLERANCE_UM = 0.5  # closer than this to a preset counts as being at it


class SlitActivity(Enum):
    Moving = 1
    StartingUp = 2
    ShuttingDown = 3


class SlitBusy(RuntimeError):
    """The stage is moving, or a refused request for it (HTTP 409, ``busy``)."""


class SlitRange(RuntimeError):
    """A position outside the stage's travel (HTTP 422, ``invalid``)."""


class SlitConf(ConfModel):
    """The operational settings (``slit.conf``)."""

    closed_um: float = setting(
        2.0,
        unit="µm",
        label="Closed position",
        help="The 'closed' preset: near the stage's lower limit, so no light gets through",
        ge=0,
        le=100000,
    )
    open_um: float = setting(
        21000.0,
        unit="µm",
        label="Open position",
        help="The 'open' preset: the stage's full travel (21 mm, 168 arcsec: far wider than "
        "the slit is long)",
        gt=0,
        le=100000,
    )
    um_per_arcsec: float = setting(
        124.69,
        unit="µm/arcsec",
        label="Microns per arcsec",
        help="The telescope's scale at the slit: a gap this many microns wide is 1 arcsec "
        "on the sky (LDT focal length 25715.4 mm: 8.02 arcsec/mm, so 124.69)",
        gt=0,
        le=1000,
    )
    settle_s: float = setting(
        0.1,
        unit="s",
        label="Settle time",
        help="Wait after the stage stops before the position counts as reached",
        ge=0,
        le=5,
    )


class SlitStatus(BaseModel):
    kind: str
    position_um: float | None  # None: not known (not started up, or lost)
    width_arcsec: float | None  # the gap on the sky, from the position
    preset: Preset | None  # the preset it is at, if it is at one
    moving: bool
    min_um: float  # the stage's travel
    max_um: float
    closed_um: float
    open_um: float
    um_per_arcsec: float
    lifecycle: LifecycleStatus


class SlitBase(BaseModel, abc.ABC):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    kind: ClassVar[str]

    _conf: ConfHandle[SlitConf] = PrivateAttr()
    _life: Lifecycle = PrivateAttr()
    _position: float | None = PrivateAttr(default=None)
    _moving: bool = PrivateAttr(default=False)

    # -- hardware hooks -----------------------------------------------------------------

    @abc.abstractmethod
    def _travel(self) -> tuple[float, float]:
        """The stage's lower and upper limit, in microns."""

    @abc.abstractmethod
    def _read_position(self) -> float | None:
        """Where the stage is now, in microns, from the hardware."""

    @abc.abstractmethod
    def _move(self, um: float) -> None:
        """Move to a position (blocking); return when the stage has stopped."""

    def _close(self) -> None:  # noqa: B027 - optional hook
        """Release the hardware."""

    # -- activities and lifecycle -------------------------------------------------------

    @property
    def activities(self) -> Activities:
        try:
            return registry.get(COMPONENT)
        except KeyError:
            return registry.register(COMPONENT, SlitActivity)

    def model_post_init(self, __context) -> None:
        self._conf = ConfHandle(COMPONENT, "conf", SlitConf)
        self._life = Lifecycle(
            COMPONENT,
            lambda: self.activities,
            startup=self._find_position,
            shutdown=self._shut,
            why_not=self._why_not,
        )

    @property
    def life(self) -> Lifecycle:
        return self._life

    @property
    def conf(self) -> ConfHandle[SlitConf]:
        return self._conf

    async def startup(self) -> LifecycleStatus:
        """Find out where the stage is. It is not moved."""
        return await self._life.startup()

    async def shutdown(self, *, force: bool = False) -> LifecycleStatus:
        """Close the slit (the safe state)."""
        return await self._life.shutdown(force=force)

    def _find_position(self) -> None:
        self._position = self._read_position()

    def _shut(self) -> None:
        self._go_to(self._conf.current.closed_um)

    def _why_not(self) -> list[str]:
        # while it moves the position is not known, and that is not a fault
        return ["slit position unknown"] if self._position is None and not self._moving else []

    # -- position -----------------------------------------------------------------------

    @property
    def position_um(self) -> float | None:
        return None if self._moving else self._position

    @property
    def width_arcsec(self) -> float | None:
        pos = self.position_um
        return None if pos is None else pos / self._conf.current.um_per_arcsec

    def um_for_arcsec(self, arcsec: float) -> float:
        return arcsec * self._conf.current.um_per_arcsec

    def preset_um(self, preset: Preset) -> float:
        c = self._conf.current
        return c.open_um if preset == "open" else c.closed_um

    def at_preset(self) -> Preset | None:
        pos = self.position_um
        if pos is None:
            return None
        for name in ("closed", "open"):
            if abs(pos - self.preset_um(name)) <= POSITION_TOLERANCE_UM:
                return name
        return None

    # -- moving -------------------------------------------------------------------------

    async def move_to(self, um: float) -> float:
        """Go to a position in microns within the travel; returns where it stopped. Raises
        :class:`SlitRange` outside the travel, :class:`SlitBusy` while it is moving."""
        self._life.require_operational()
        lo, hi = self._travel()
        if not lo <= um <= hi:
            raise SlitRange(f"{um:g} µm is outside the slit's travel, {lo:g} to {hi:g} µm")
        if self._moving:
            raise SlitBusy("the slit is moving")
        self._moving = True
        self.activities.start(SlitActivity.Moving, details=[f"{um:g} µm"])
        try:
            await asyncio.to_thread(self._go_to, um)
        finally:
            self._moving = False
            self.activities.end(SlitActivity.Moving)
        return self._position  # type: ignore[return-value]

    async def move_to_preset(self, preset: Preset) -> float:
        return await self.move_to(self.preset_um(preset))

    @contextlib.asynccontextmanager
    async def holding(self) -> AsyncIterator[None]:
        """Put the stage back where it was after a block that moves it, however the block
        ends."""
        before = self.position_um
        try:
            yield
        finally:
            if before is not None and self._position != before:
                await self.move_to(before)

    def _go_to(self, um: float) -> None:
        if self._position is not None and abs(self._position - um) < 1e-9:
            return
        self._position = None  # until the move has finished
        self._move(um)
        self._position = self._read_position()
        if self._position is None or abs(self._position - um) > POSITION_TOLERANCE_UM:
            raise SlitBusy(f"the slit did not reach {um:g} µm")
        log.info("slit at %.1f µm", self._position)

    # -- status -------------------------------------------------------------------------

    def status(self) -> SlitStatus:
        lo, hi = self._travel()
        c = self._conf.current
        return SlitStatus(
            kind=self.kind,
            position_um=self.position_um,
            width_arcsec=self.width_arcsec,
            preset=self.at_preset(),
            moving=self._moving,
            min_um=lo,
            max_um=hi,
            closed_um=c.closed_um,
            open_um=c.open_um,
            um_per_arcsec=c.um_per_arcsec,
            lifecycle=self._life.status(),
        )

    async def close_hardware(self) -> None:
        self._close()
