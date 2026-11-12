"""Calibrating the guide camera and the slit from the sky (plan §3.3, ``optics_calibration``).

Starting it needs the control lease and is a job that reserves the camera, the slit stage and the
telescope (nothing may move the field meanwhile). The telescope must be on a star field and the
light path not blocked. It closes the slit, takes a frame and plate solves it, opens the slit to a
measuring width, takes another, and finds the gap in the difference. The result is a measurement
and nothing more: ``GET /calibration/optics/last`` shows it with its warnings, and
``POST /calibration/optics/apply`` (naming the measurement) writes it into the camera's
calibration, with who and when.
"""

import asyncio
from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from .. import optics_calibration as oc
from ..errors import ApiError
from ..field_solver import FieldSolver, SolveFailed, get_field_solver
from ..guide_camera import (
    CameraBusy,
    ExposureAborted,
    GuideCameraBase,
    SettingError,
    get_guide_camera,
)
from ..guide_camera.base import COMPONENT as CAMERA
from ..guide_camera.base import EXPOSURE_RANGE_S, GAIN_RANGE, CaptureSettings
from ..jobs import Job, JobCancelled, JobContext, StartedBy, manager
from ..light_path import LightPath, get_light_path
from ..slit import COMPONENT as SLIT
from ..slit import SlitBase, SlitBusy, SlitRange, get_slit
from ..slit_detect import SlitNotFound
from ..telescope import COMPONENT as TELESCOPE
from ..telescope import TelescopeBase, get_telescope
from .control import Me, require_lease
from .guide_calibration import until_done

router = APIRouter(prefix="/calibration/optics", tags=["calibration"])

Camera = Annotated[GuideCameraBase, Depends(get_guide_camera)]
Slit = Annotated[SlitBase, Depends(get_slit)]
Telescope = Annotated[TelescopeBase, Depends(get_telescope)]
Solver = Annotated[FieldSolver, Depends(get_field_solver)]
Light = Annotated[LightPath, Depends(get_light_path)]
NeedsLease = Depends(require_lease)

FALLBACK_SCALE = 0.037  # arcsec per unbinned pixel: the guide camera at 1:1 (4.63 um, 8.02"/mm)


class OpticsRequest(BaseModel):
    exposure_s: float = Field(
        4.0,
        ge=EXPOSURE_RANGE_S[0],
        le=EXPOSURE_RANGE_S[1],
        description="Each of the two frames; the gap is faint, so longer is better",
    )
    gain: int | None = Field(
        None,
        ge=GAIN_RANGE[0],
        le=GAIN_RANGE[1],
        description="The single-frame group's if not given",
    )
    binning: int | None = Field(None, description="1 or 2; the single-frame group's if not given")
    slit_width_arcsec: float = Field(
        oc.DEFAULT_WIDTH_ARCSEC,
        gt=0,
        description="The measuring gap: a strip, narrower than the slit is long",
    )
    nominal_scale_arcsec_px: float | None = Field(
        None,
        gt=0,
        description="Roughly the plate scale per unbinned pixel, to help the solver (within 30%); "
        "the calibration's, else the guide camera's design value, if not given",
    )


def _conflict(message: str, **details) -> ApiError:
    return ApiError(409, "conflict", message, details or None)


def settings_for(request: OpticsRequest, camera: GuideCameraBase) -> CaptureSettings:
    base: CaptureSettings = camera.conf.snapshot().single_frame
    update: dict = {"exposure_s": request.exposure_s, "roi": None}
    if request.gain is not None:
        update["gain"] = request.gain
    if request.binning is not None:
        if request.binning not in (1, 2):
            raise ApiError(422, "invalid", "binning is 1 or 2")
        update["binning"] = request.binning
    return base.model_copy(update=update)


def check_ready(
    camera: GuideCameraBase, slit: SlitBase, telescope: TelescopeBase, light: LightPath
):
    camera.life.require_operational()
    slit.life.require_operational()
    if camera.streaming:
        raise ApiError(409, "busy", "the guide camera is streaming", {"component": CAMERA})
    if light.state in ("blocked", "block_requested"):
        raise _conflict(
            "the light path is blocked (or being blocked): there is no sky to calibrate on",
            state=light.state,
        )
    if why := telescope.why_not_on_target():
        raise _conflict(f"{why}: the telescope must be on a star field")


@router.post("", status_code=202, response_model=Job, dependencies=[NeedsLease])
async def run_calibration(
    request: OpticsRequest,
    camera: Camera,
    slit: Slit,
    telescope: Telescope,
    solver: Solver,
    light: Light,
    me: Me,
) -> Job:
    """The plate scale, the field's rotation and the slit's place, direction and size, as a job
    (see the module docstring). ``409 conflict`` unless the telescope is on a field and the
    light path is open, ``409 busy`` while another job has the camera, the slit or the
    telescope; the job fails with ``solve_failed`` or ``slit_not_found``."""
    check_ready(camera, slit, telescope, light)
    settings = settings_for(request, camera)
    um = slit.um_for_arcsec(request.slit_width_arcsec)
    status = slit.status()
    lo, hi = status.min_um, status.max_um
    if not lo <= um <= hi:
        raise ApiError(
            422,
            "invalid",
            f"a {request.slit_width_arcsec:g} arcsec gap is outside the slit's travel",
        )
    nominal = (
        request.nominal_scale_arcsec_px
        or camera.cal.current.plate_scale_arcsec_px
        or FALLBACK_SCALE
    )
    who = StartedBy(name=me.name, code=me.code)

    async def run(ctx: JobContext) -> dict:
        async def expose(s: CaptureSettings):
            return await until_done(ctx, camera, asyncio.create_task(camera.exposure(s)))

        try:
            m = await oc.measure(
                expose=expose,
                slit=slit,
                telescope=telescope,
                solver=solver,
                settings=settings,
                width_arcsec=request.slit_width_arcsec,
                nominal_scale_arcsec_px=nominal,
                who=me.name,
                progress=ctx.progress,
            )
        except SolveFailed as e:
            raise ApiError(409, "solve_failed", str(e)) from e
        except SlitNotFound as e:
            raise ApiError(409, "slit_not_found", str(e)) from e
        except CameraBusy as e:
            raise ApiError(409, "busy", str(e), {"component": CAMERA}) from e
        except SlitBusy as e:
            raise ApiError(409, "busy", str(e), {"component": SLIT}) from e
        except SlitRange as e:
            raise ApiError(422, "invalid", str(e)) from e
        except SettingError as e:
            raise ApiError(422, "invalid", str(e)) from e
        except ExposureAborted as e:
            raise JobCancelled from e
        oc.remember(m)
        return m.model_dump()

    return manager.submit("optics-calibration", [CAMERA, SLIT, TELESCOPE], run, started_by=who)


@router.get("/last", response_model=oc.Measurement)
async def get_last() -> oc.Measurement:
    """The most recent measurement, applied or not (``404 not_found`` before the first)."""
    m = oc.last()
    if m is None:
        raise ApiError(404, "not_found", "no optics calibration has been run yet")
    return m


class ApplyRequest(BaseModel):
    taken_ms: int = Field(
        description="The measurement to apply (its taken_ms), so a newer run "
        "is not applied by mistake"
    )


@router.post("/apply", dependencies=[NeedsLease])
async def apply(body: ApplyRequest, camera: Camera, me: Me) -> dict:
    """Write the last measurement into the camera's calibration (plate scale, orientation, parity,
    slit pixel, slit angle, slit length). ``409 conflict`` if it is not the one named."""
    m = oc.last()
    if m is None or m.taken_ms != body.taken_ms:
        raise _conflict("that measurement is not the latest: run it again or fetch the latest")
    values = oc.cal_values(m)
    camera.cal.patch(
        values,
        who=StartedBy(name=me.name, code=me.code),
        reason=f"optics calibration of {m.taken_ms}",
    )
    return {"applied": values, "cal_version": camera.cal.version}
