"""GET /status and GET /status/stream (plan §2).

Skeleton: the status model has the shape wilds-gui's Alpine store expects (``control``,
``tcs``, ``actions``), filled with "nothing known yet" values until the broker, devices and
control lease exist.
"""

import asyncio
import time
from collections.abc import AsyncIterator

from fastapi import APIRouter
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from .. import config
from ..activities import registry
from ..conf import confs
from ..filing import FilingStatus, get_filing
from ..guide_camera import GuideCameraStatus, get_guide_camera
from ..jobs import Job, manager
from ..lease import state as lease
from ..lifecycle import SystemStatus, system
from ..light_path import LightPathStatus, get_light_path
from ..shutter import ShutterStatus, get_shutter
from ..slit import SlitStatus, get_slit
from ..telescope import TelescopeStatus, get_telescope

router = APIRouter(prefix="/status", tags=["status"])

STATUS_PERIOD_S = 1.0


class ControlLease(BaseModel):
    holder: str | None = None
    holder_code: str | None = None
    holder_info: str | None = None  # where the holder comes from, plain text
    engineering: bool = False


class TcsStatus(BaseModel):
    state: str | None = None
    health: str | None = None
    accessMode: str | None = None
    preview: bool | None = None
    lockedOut: bool | None = None
    stale: bool = True


class ToStatus(BaseModel):
    online: bool = False


class ServerTime(BaseModel):
    """The server's clock, so every page shows the same time whatever its own clock says."""

    utc_ms: int  # milliseconds since the Unix epoch, UTC
    zone: str  # IANA zone for LDT local time (config.toml [site] timezone)


class ActivityStatus(BaseModel):
    name: str
    since_ms: int  # when it started, ms since the Unix epoch, UTC
    details: list[str] = Field(default_factory=list)


class Action(BaseModel):
    ok: bool
    why: str | None = None


class Status(BaseModel):
    opmode: config.OpMode = "development"
    control: ControlLease = Field(default_factory=ControlLease)
    tcs: TcsStatus = Field(default_factory=TcsStatus)
    time: ServerTime
    to: ToStatus = Field(default_factory=ToStatus)
    activities: dict[str, list[ActivityStatus]] = Field(default_factory=dict)  # running only
    system: SystemStatus | None = None  # every component's opstate, operational and reasons
    guide_camera: GuideCameraStatus | None = None  # settings, streaming and the cooler
    shutter: ShutterStatus | None = None  # blade state and lifecycle
    slit: SlitStatus | None = None  # stage position in µm, the gap in arcsec, presets
    telescope: TelescopeStatus | None = (
        None  # target, offsets, rotator, preview, what waits for the TO
    )
    light_path: LightPathStatus | None = None  # lit, or blocked for darks (asked of the TO)
    conf_versions: dict[str, int] = Field(default_factory=dict)  # 'component/slice' -> version
    filing: FilingStatus | None = None  # staging, the data share, what waits to be delivered
    jobs: list[Job] = Field(default_factory=list)  # active, then the last few finished
    warnings: list[str] = Field(default_factory=list)
    actions: dict[str, Action] = Field(default_factory=dict)


def _tcs(telescope) -> TcsStatus:
    """The header's TCS pill, from the telescope (the simulator until the bridge is wired in)."""
    t = telescope.telemetry()
    stale = t.stale or not telescope.life.operational
    return TcsStatus(
        state=t.state,
        health=t.health,
        accessMode=t.access_mode,
        preview=t.preview,
        lockedOut=t.locked_out,
        stale=stale,
    )


def _telescope_actions(telescope) -> dict[str, Action]:
    """What may be sent to the telescope now, for the GUIs to grey out with the reason (plan
    §3.12). A slew needs only that the telescope takes commands; the adjustments (offset,
    rotator) also need a target it is on. Whether this browser's session holds control is the
    GUI's own (local) check."""
    why = telescope.why_not_available(engineering=lease.engineering)
    adjust = why or telescope.why_not_on_target()
    return {
        "slew": Action(ok=not why, why=why or None),
        "offset": Action(ok=not adjust, why=adjust or None),
        "rotator": Action(ok=not adjust, why=adjust or None),
    }


def current_status() -> Status:
    # TODO: add the broker and devices
    lease.expire()
    telescope = get_telescope()
    holder = lease.holder
    filing = get_filing().status()
    return Status(
        opmode=config.get().opmode,
        control=ControlLease(
            holder=holder.name if holder else None,
            holder_code=holder.code if holder else None,
            holder_info=holder.prov.summary() if holder and holder.prov else None,
            engineering=lease.engineering,
        ),
        time=ServerTime(utc_ms=time.time_ns() // 1_000_000, zone=config.get().site.timezone),
        to=ToStatus(online=lease.to_online),
        activities={
            component: [
                ActivityStatus(name=i.name, since_ms=i.since_ms, details=list(i.details))
                for i in infos
            ]
            for component, infos in registry.snapshot().items()
        },
        system=system.status(),
        guide_camera=get_guide_camera().status(),
        shutter=get_shutter().status(),
        slit=get_slit().status(),
        light_path=get_light_path().status(),
        telescope=telescope.status(),
        tcs=_tcs(telescope),
        actions=_telescope_actions(telescope),
        conf_versions=confs.versions(),
        filing=filing,
        jobs=manager.snapshot(),
        warnings=config.get().warnings() + filing.warnings(),
    )


@router.get("", response_model=Status)
async def get_status() -> Status:
    return current_status()


@router.get("/stream")
async def stream_status() -> EventSourceResponse:
    async def events() -> AsyncIterator[dict[str, str]]:
        jobs = manager.subscribe()
        loop = asyncio.get_running_loop()
        try:
            while True:
                yield {"event": "status", "data": current_status().model_dump_json()}
                deadline = loop.time() + STATUS_PERIOD_S
                while (left := deadline - loop.time()) > 0:
                    try:
                        job = await asyncio.wait_for(jobs.get(), left)
                    except TimeoutError:
                        break
                    yield {"event": "job", "data": job.model_dump_json()}
        finally:
            manager.unsubscribe(jobs)

    return EventSourceResponse(events())
