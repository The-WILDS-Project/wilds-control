"""The instrument cover: its state, and opening and closing it (plan §3.17).

Anyone may read; moving it needs the control lease (the instrument operator's).
The GUI reminds whoever moves it to tell the Telescope Operator first: LDT asks that the TO
knows, and no software step enforces it. A move returns at once with ``opening`` or
``closing``; the status says when it has arrived.
"""

from typing import Annotated

from fastapi import APIRouter, Depends

from ..cover import COMPONENT, CoverBase, CoverBusy, CoverError, CoverStatus, get_cover
from ..errors import ApiError
from ..jobs import manager
from .control import require_lease

router = APIRouter(prefix="/cover", tags=["cover"])

Cover = Annotated[CoverBase, Depends(get_cover)]
NeedsLease = Depends(require_lease)


def _free() -> None:
    if job := manager.busy_job(COMPONENT):
        raise ApiError(409, "busy", f"the cover is in use by job {job}", {"job_id": job})


async def _do(cover: CoverBase, close: bool) -> CoverStatus:
    _free()
    try:
        return await (cover.close() if close else cover.open())
    except CoverBusy as e:
        raise ApiError(409, "busy", str(e), {"component": COMPONENT}) from e
    except CoverError as e:
        raise ApiError(502, "unavailable", str(e)) from e


@router.get("", response_model=CoverStatus)
async def get_status(cover: Cover) -> CoverStatus:
    return cover.status()


@router.post("/close", response_model=CoverStatus, dependencies=[NeedsLease])
async def close(cover: Cover) -> CoverStatus:
    """Close the cover (blocks the light). ``409 not_operational`` until started up, ``409 busy``
    while it moves; already closed is not an error."""
    return await _do(cover, True)


@router.post("/open", response_model=CoverStatus, dependencies=[NeedsLease])
async def open_(cover: Cover) -> CoverStatus:
    """Open the cover (lets the light in)."""
    return await _do(cover, False)
