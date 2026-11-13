"""The telescope as WILDS sees it (plan §3.3, §3.12): an abstract base and what is shared.

WILDS does not drive the LDT's mount. It sends the TCS targets, offsets and a rotator angle
through the broker (the ``wilds-bridge`` package) and learns what happened from the TCS's own
telemetry. This module is the shape of that: a few hooks that *send* a command, one that reads
the telemetry, and everything else (lifecycle, activities, the status, who may command and
why not) here. :class:`~wilds.control.telescope.simulated.SimulatedTelescope` stands in until
the bridge is wired in.

Nothing here imports FastAPI. A command is *sent*; there is no negotiation with the Telescope
Operator. The TCS acts on any valid command while it accepts external commands (the Telescope
Operator switches that on once the system is up; the telemetry says which it is), and whether a
command was carried out is only ever read from the telemetry (``operations.py``), never assumed
from having sent it.
"""

import abc
import asyncio
import logging
from enum import Enum
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from ..activities import Activities, registry
from ..conf import ConfHandle, ConfModel, setting
from ..lifecycle import Lifecycle, LifecycleStatus

log = logging.getLogger(__name__)

COMPONENT = "telescope"

RotatorFrame = Literal["Target", "Fixed"]
OffsetNum = Literal["User", "Handset"]
OffsetType = Literal["TPLANE", "SIMPLE"]


class TelescopeActivity(Enum):
    Slewing = 1
    Rotating = 2
    Offsetting = 3
    StartingUp = 4
    ShuttingDown = 5


class TelescopeError(RuntimeError):
    """A command could not be sent (HTTP 409, ``not_operational`` or ``refused``)."""


class TelescopeConf(ConfModel):
    """The operational settings (``telescope.conf``)."""

    motion_timeout_s: float = setting(
        300.0,
        unit="s",
        label="Wait for the motion",
        help="How long a command may take before the telescope is reported as not having arrived",
        ge=5,
        le=1800,
    )
    max_offset_arcsec: float = setting(
        120.0,
        unit="arcsec",
        label="Largest offset",
        help="An offset beyond this is refused: acquisition offsets are small, and a larger "
        "one is probably a mistake",
        gt=0,
        le=3600,
    )


class Target(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    ra_hours: float = Field(ge=0, lt=24)
    dec_deg: float = Field(ge=-90, le=90)


class Offset(BaseModel):
    d1_arcsec: float = 0.0  # RA (or Az)
    d2_arcsec: float = 0.0  # Dec (or El)


class Telemetry(BaseModel):
    """What the TCS reports, as one snapshot (the hook :meth:`TelescopeBase._read` returns it)."""

    target: Target | None = None
    rot_pa: float | None = None  # degrees E of N, the slit's sky PA now
    rot_goal: float | None = None  # where the rotator is going
    rot_frame: RotatorFrame = "Target"
    offsets: dict[OffsetNum, Offset] = Field(default_factory=dict)
    in_position: bool | None = None
    # Does the TCS act on external commands? (None: unknown.) The TCS reports its External
    # Command Preview (externalTargetCfgCmdPreviewIsTrue): with that on it holds an external
    # target instead of acting on it, so "accepts" is its opposite.
    accepts_commands: bool | None = None
    state: str | None = None  # TCS state
    health: str | None = None
    access_mode: str | None = None
    stale: bool = False  # the last report is old


class TelescopeStatus(BaseModel):
    kind: str
    lifecycle: LifecycleStatus
    telemetry: Telemetry
    max_offset_arcsec: float


class TelescopeBase(BaseModel, abc.ABC):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    kind: ClassVar[str]

    _conf: ConfHandle[TelescopeConf] = PrivateAttr()
    _life: Lifecycle = PrivateAttr()

    # -- the hooks: send a command, read the telemetry (blocking, in a worker thread) ----

    @abc.abstractmethod
    def _read(self) -> Telemetry: ...

    @abc.abstractmethod
    def _send_target(self, target: Target, rot_pa: float, rot_frame: RotatorFrame) -> None: ...

    @abc.abstractmethod
    def _send_rotator(self, rot_pa: float, rot_frame: RotatorFrame | None) -> None: ...

    @abc.abstractmethod
    def _send_offset(self, num: OffsetNum, offset: Offset, type: OffsetType) -> None: ...

    @abc.abstractmethod
    def _send_clear(self, num: OffsetNum) -> None: ...

    @abc.abstractmethod
    def _send_absorb(self, num: OffsetNum) -> None: ...

    def _connect(self) -> None:  # noqa: B027 - optional hook
        """Reach the broker."""

    def _disconnect(self) -> None:  # noqa: B027 - optional hook
        """Let go of the broker."""

    # -- activities and lifecycle ----------------------------------------------------------

    @property
    def activities(self) -> Activities:
        try:
            return registry.get(COMPONENT)
        except KeyError:
            return registry.register(COMPONENT, TelescopeActivity)

    def model_post_init(self, __context) -> None:
        self._conf = ConfHandle(COMPONENT, "conf", TelescopeConf)
        self._life = Lifecycle(
            COMPONENT,
            lambda: self.activities,
            startup=self._connect,
            shutdown=self._disconnect,
            why_not=self._why_not,
        )

    @property
    def life(self) -> Lifecycle:
        return self._life

    @property
    def conf(self) -> ConfHandle[TelescopeConf]:
        return self._conf

    async def startup(self) -> LifecycleStatus:
        return await self._life.startup()

    async def shutdown(self, *, force: bool = False) -> LifecycleStatus:
        return await self._life.shutdown(force=force)

    def _why_not(self) -> list[str]:
        t = self.telemetry()
        if t.stale:
            return ["TCS telemetry is stale"]
        return []

    # -- telemetry -------------------------------------------------------------------------

    def telemetry(self) -> Telemetry:
        return self._read()

    def why_not_available(self, *, engineering: bool = False) -> str:
        """Why a command to the telescope cannot be sent now, or ``""`` (plan §3.12). The
        lease itself (a TO-granted one) is checked by the caller; engineering mode does not
        command the telescope."""
        if engineering:
            return "Engineering mode does not command the telescope"
        if not self._life.operational:
            return "The telescope is not operational: " + "; ".join(
                self._life.why_not_operational() or ["not started up"]
            )
        t = self.telemetry()
        if t.accepts_commands is False:
            return (
                "The TCS is not accepting external commands: the Telescope Operator switches "
                "that on in the TCS"
            )
        if t.state not in (None, "ENABLED"):
            return f"The TCS is {t.state.lower()}, not enabled"
        return ""

    def why_not_on_target(self) -> str:
        """Why an adjustment (offsets, rotation) cannot be made now, or ``""``: it needs a target
        the telescope has arrived at."""
        t = self.telemetry()
        if t.target is None:
            return "No target yet: slew to one first"
        if t.in_position is not True:
            return "The telescope is not on target yet"
        return ""

    # -- commands: each one sends and returns; ``operations`` watches the outcome -----------

    async def send_target(self, target: Target, rot_pa: float, rot_frame: RotatorFrame) -> None:
        self._check()
        await asyncio.to_thread(self._send_target, target, rot_pa % 360.0, rot_frame)
        log.info("target %s sent", target.name)

    async def send_rotator(self, rot_pa: float, rot_frame: RotatorFrame | None = None) -> None:
        self._check()
        if self.telemetry().target is None:
            raise TelescopeError("no target yet: slew to one first")
        await asyncio.to_thread(self._send_rotator, rot_pa % 360.0, rot_frame)
        log.info("rotator %.2f sent", rot_pa % 360.0)

    async def send_offset(self, offset: Offset, num: OffsetNum, type: OffsetType) -> None:
        self._check()
        limit = self._conf.current.max_offset_arcsec
        if max(abs(offset.d1_arcsec), abs(offset.d2_arcsec)) > limit:
            raise TelescopeError(f"an offset beyond {limit:g} arcsec is refused")
        await asyncio.to_thread(self._send_offset, num, offset, type)

    async def send_clear(self, num: OffsetNum) -> None:
        self._check()
        await asyncio.to_thread(self._send_clear, num)

    async def send_absorb(self, num: OffsetNum) -> None:
        self._check()
        await asyncio.to_thread(self._send_absorb, num)

    def _check(self) -> None:
        self._life.require_operational()
        if why := self.why_not_available():
            raise TelescopeError(why)

    # -- status ----------------------------------------------------------------------------

    def status(self) -> TelescopeStatus:
        return TelescopeStatus(
            kind=self.kind,
            lifecycle=self._life.status(),
            telemetry=self.telemetry(),
            max_offset_arcsec=self._conf.current.max_offset_arcsec,
        )

    async def close_hardware(self) -> None:
        self._disconnect()
