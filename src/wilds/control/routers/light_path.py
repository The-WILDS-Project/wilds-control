"""The light path to WILDS: ask the TO to block or restore it (plan §3.17).

Anyone may read. The lease holder asks and may cancel; the TO (passphrase) acknowledges,
refuses with a reason, or states how things are.
"""

from typing import Annotated, Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from ..light_path import LightPath, LightPathStatus, get_light_path
from .control import ToOnly, holder_who, require_lease

router = APIRouter(prefix="/light-path", tags=["light-path"])

Light = Annotated[LightPath, Depends(get_light_path)]
NeedsLease = Depends(require_lease)


class Note(BaseModel):
    note: str | None = Field(None, max_length=200)


class Reason(BaseModel):
    reason: str = Field(min_length=1, max_length=200)


class Stated(Note):
    state: Literal["lit", "blocked"]


@router.get("", response_model=LightPathStatus)
async def get_status(light: Light) -> LightPathStatus:
    return light.status()


@router.post("/block-request", response_model=LightPathStatus, dependencies=[NeedsLease])
async def request_block(light: Light) -> LightPathStatus:
    """Ask the TO to block the light to WILDS. ``409 conflict`` if a request waits or
    it is blocked already."""
    light.request_block(holder_who().name)
    return light.status()


@router.post("/restore-request", response_model=LightPathStatus, dependencies=[NeedsLease])
async def request_restore(light: Light) -> LightPathStatus:
    """Ask the TO to let the light in again."""
    light.request_restore(holder_who().name)
    return light.status()


@router.post("/cancel", response_model=LightPathStatus, dependencies=[NeedsLease])
async def cancel(light: Light) -> LightPathStatus:
    light.cancel()
    return light.status()


@router.post("/acknowledge", response_model=LightPathStatus, dependencies=[ToOnly])
async def acknowledge(light: Light, body: Note | None = None) -> LightPathStatus:
    """The TO has done it: a block request becomes ``blocked``, a restore request ``lit``."""
    light.acknowledge((body or Note()).note)
    return light.status()


@router.post("/refuse", response_model=LightPathStatus, dependencies=[ToOnly])
async def refuse(body: Reason, light: Light) -> LightPathStatus:
    """The TO cannot now; the state goes back to what it was and the reason is shown."""
    light.refuse(body.reason)
    return light.status()


@router.put("/state", response_model=LightPathStatus, dependencies=[ToOnly])
async def set_state(body: Stated, light: Light) -> LightPathStatus:
    """The TO states how things are (also how they restore the light unasked)."""
    light.set(body.state, body.note)
    return light.status()
