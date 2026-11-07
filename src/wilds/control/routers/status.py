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
from ..jobs import Job, manager
from ..lease import state as lease

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
    jobs: list[Job] = Field(default_factory=list)  # active, then the last few finished
    warnings: list[str] = Field(default_factory=list)
    actions: dict[str, Action] = Field(default_factory=dict)


def current_status() -> Status:
    # TODO: add the broker, devices and jobs
    lease.expire()
    holder = lease.holder
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
        jobs=manager.snapshot(),
        warnings=config.get().warnings(),
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
