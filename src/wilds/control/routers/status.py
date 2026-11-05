"""GET /status and GET /status/stream (plan §2).

Skeleton: the status model has the shape wilds-gui's Alpine store expects (``control``,
``tcs``, ``actions``), filled with "nothing known yet" values until the broker, devices and
control lease exist.
"""

import asyncio
from collections.abc import AsyncIterator

from fastapi import APIRouter
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from ..lease import state as lease

router = APIRouter(prefix="/status", tags=["status"])

STATUS_PERIOD_S = 1.0


class ControlLease(BaseModel):
    holder: str | None = None
    engineering: bool = False


class TcsStatus(BaseModel):
    state: str | None = None
    health: str | None = None
    accessMode: str | None = None
    preview: bool | None = None
    lockedOut: bool | None = None
    stale: bool = True


class Action(BaseModel):
    ok: bool
    why: str | None = None


class Status(BaseModel):
    control: ControlLease = Field(default_factory=ControlLease)
    tcs: TcsStatus = Field(default_factory=TcsStatus)
    actions: dict[str, Action] = Field(default_factory=dict)


def current_status() -> Status:
    # TODO: add the broker, devices and jobs
    lease.expire()
    holder = lease.holder
    return Status(
        control=ControlLease(holder=holder.name if holder else None, engineering=lease.engineering)
    )


@router.get("", response_model=Status)
async def get_status() -> Status:
    return current_status()


@router.get("/stream")
async def stream_status() -> EventSourceResponse:
    async def events() -> AsyncIterator[dict[str, str]]:
        while True:
            yield {"event": "status", "data": current_status().model_dump_json()}
            await asyncio.sleep(STATUS_PERIOD_S)

    return EventSourceResponse(events())
