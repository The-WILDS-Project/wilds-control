"""The shutter: its state, and opening and closing it by hand (plan §3.17).

Anyone may read; moving it needs the control lease. A job that reserved the shutter (a dark or
bias sequence) has it to itself: a hand request meanwhile is refused with ``409 busy``.
"""

from typing import Annotated

from fastapi import APIRouter, Depends

from ..errors import ApiError
from ..jobs import manager
from ..shutter import COMPONENT, ShutterBase, ShutterBusy, ShutterStatus, get_shutter
from .control import require_lease

router = APIRouter(prefix="/shutter", tags=["shutter"])

Shutter = Annotated[ShutterBase, Depends(get_shutter)]
NeedsLease = Depends(require_lease)


def _check_free() -> None:
    if job := manager.busy_job(COMPONENT):
        raise ApiError(409, "busy", f"the shutter is in use by job {job}", {"job_id": job})


def _busy(e: ShutterBusy) -> ApiError:
    return ApiError(409, "busy", str(e), {"component": COMPONENT})


@router.get("", response_model=ShutterStatus)
async def get_status(shutter: Shutter) -> ShutterStatus:
    return shutter.status()


@router.post("/open", response_model=ShutterStatus, dependencies=[NeedsLease])
async def open_shutter(shutter: Shutter) -> ShutterStatus:
    """``409 not_operational`` until the shutter is started up, ``409 busy`` while it moves or a
    job has it."""
    _check_free()
    try:
        await shutter.open()
    except ShutterBusy as e:
        raise _busy(e) from e
    return shutter.status()


@router.post("/close", response_model=ShutterStatus, dependencies=[NeedsLease])
async def close_shutter(shutter: Shutter) -> ShutterStatus:
    _check_free()
    try:
        await shutter.close()
    except ShutterBusy as e:
        raise _busy(e) from e
    return shutter.status()
