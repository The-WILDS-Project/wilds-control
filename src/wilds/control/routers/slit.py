"""The slit stage: its position and moving it by hand (plan §3.3).

Anyone may read; moving it needs the control lease. A job that reserved the slit (the guide
camera and slit calibration) has it to itself: a hand request meanwhile is refused with
``409 busy``. A position is in microns at the focal plane, or a preset (``open``, ``closed``);
the status also gives it as the gap's width in arcseconds.
"""

from typing import Annotated, Self

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field, model_validator

from ..errors import ApiError
from ..jobs import manager
from ..slit import COMPONENT, Preset, SlitBase, SlitBusy, SlitRange, SlitStatus, get_slit
from .control import require_lease

router = APIRouter(prefix="/slit", tags=["slit"])

Slit = Annotated[SlitBase, Depends(get_slit)]
NeedsLease = Depends(require_lease)


class SlitMove(BaseModel):
    """Where to go: exactly one of a position or a preset."""

    position_um: float | None = Field(None, ge=0, description="microns at the focal plane")
    preset: Preset | None = None

    @model_validator(mode="after")
    def _one(self) -> Self:
        if (self.position_um is None) == (self.preset is None):
            raise ValueError("give either position_um or preset, not both and not neither")
        return self


def _check_free() -> None:
    if job := manager.busy_job(COMPONENT):
        raise ApiError(409, "busy", f"the slit is in use by job {job}", {"job_id": job})


@router.get("", response_model=SlitStatus)
async def get_status(slit: Slit) -> SlitStatus:
    return slit.status()


@router.post("/move", response_model=SlitStatus, dependencies=[NeedsLease])
async def move(body: SlitMove, slit: Slit) -> SlitStatus:
    """``409 not_operational`` until the slit is started up, ``409 busy`` while it moves or a
    job has it, ``422 invalid`` outside its travel."""
    _check_free()
    try:
        if body.preset is not None:
            await slit.move_to_preset(body.preset)
        else:
            await slit.move_to(body.position_um)  # type: ignore[arg-type]
    except SlitRange as e:
        raise ApiError(422, "invalid", str(e)) from e
    except SlitBusy as e:
        raise ApiError(409, "busy", str(e), {"component": COMPONENT}) from e
    return slit.status()
