"""The ADC: where its two prism pairs are, moving them by hand, and tracking the telescope.

Anyone may read; moving the pairs or switching tracking needs the control lease. A hand move is
refused while tracking is on (``409 busy``, as it is while the pairs move), and a dispersion
the ADC cannot make is ``422 invalid``. The ADC is optional (``adc.conf`` ``installed``): not
installed, every command is ``409 not_installed`` and the status says so. The status also says what tracking last worked out (the
atmosphere's dispersion at the target's zenith distance, and where the pairs are to be), for the
monitoring page.
"""

from typing import Annotated, Self

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field, model_validator

from ..adc import COMPONENT, AdcBase, AdcBusy, AdcRange, AdcStatus, get_adc
from ..errors import ApiError
from .control import require_lease

router = APIRouter(prefix="/adc", tags=["adc"])

Adc = Annotated[AdcBase, Depends(get_adc)]
NeedsLease = Depends(require_lease)


class AdcMove(BaseModel):
    """Where to put the pairs: both positions, or the dispersion to make and its direction."""

    pair1_deg: float | None = Field(None, description="position of pair 1, degrees")
    pair2_deg: float | None = Field(None, description="position of pair 2, degrees")
    dispersion_arcsec: float | None = Field(None, ge=0, description="dispersion to make, arcsec")
    direction_deg: float | None = Field(None, description="its direction, degrees")

    @model_validator(mode="after")
    def _one_way(self) -> Self:
        angles = self.pair1_deg is not None and self.pair2_deg is not None
        dispersion = self.dispersion_arcsec is not None and self.direction_deg is not None
        any_angle = self.pair1_deg is not None or self.pair2_deg is not None
        any_disp = self.dispersion_arcsec is not None or self.direction_deg is not None
        if angles == dispersion or (angles and any_disp) or (dispersion and any_angle):
            raise ValueError(
                "give both pair1_deg and pair2_deg, or both dispersion_arcsec and direction_deg"
            )
        return self


class Tracking(BaseModel):
    enabled: bool


@router.get("", response_model=AdcStatus)
async def get_status(adc: Adc) -> AdcStatus:
    return adc.status()


@router.post("/move", response_model=AdcStatus, dependencies=[NeedsLease])
async def move(body: AdcMove, adc: Adc) -> AdcStatus:
    """``409 not_installed``, ``409 not_operational`` until the ADC is started up, ``409 busy`` while it
    moves or tracking is on, ``422 invalid`` for more dispersion than it can make."""
    try:
        if body.pair1_deg is not None:
            await adc.move_to(body.pair1_deg, body.pair2_deg)  # type: ignore[arg-type]
        else:
            await adc.move_dispersion(body.dispersion_arcsec, body.direction_deg)  # type: ignore[arg-type]
    except AdcRange as e:
        raise ApiError(422, "invalid", str(e)) from e
    except AdcBusy as e:
        raise ApiError(409, "busy", str(e), {"component": COMPONENT}) from e
    return adc.status()


@router.post("/tracking", response_model=AdcStatus, dependencies=[NeedsLease])
async def tracking(body: Tracking, adc: Adc) -> AdcStatus:
    """Switch tracking of the telescope on or off. ``409 not_installed``; ``409 not_operational`` until started up."""
    await adc.set_tracking(body.enabled)
    return adc.status()
