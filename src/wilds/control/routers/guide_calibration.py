"""The guide camera's dark and bias frames (plan §3.17).

Taking them needs the control lease and is a job that reserves the camera and the shutter: it
closes the shutter, takes ``count`` frames with the camera's own settings (those of the
``single_frame`` or ``video`` group, a dark with its own exposure if asked, a bias at the shortest
exposure), puts the shutter back as it found it, and files the frames and their median in the
active calibration session. Sessions are time-stamped folders on the local disk; any of them
can be made the active one again.
"""

import asyncio
import contextlib
from typing import Annotated, Literal

from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, model_validator

from ..errors import ApiError
from ..guide_camera import (
    CameraBusy,
    ExposureAborted,
    GuideCameraBase,
    SettingError,
    get_guide_camera,
)
from ..guide_camera.base import COMPONENT as CAMERA
from ..guide_camera.base import EXPOSURE_RANGE_S, CalibrationKind, CaptureSettings
from ..guide_camera.calibration_frames import SessionInfo, store
from ..jobs import Job, JobCancelled, JobContext, StartedBy, manager
from ..shutter import COMPONENT as SHUTTER
from ..shutter import ShutterBase, ShutterBusy, get_shutter
from .control import Me, require_lease

router = APIRouter(prefix="/guide-camera/calibration", tags=["guide-camera"])

Camera = Annotated[GuideCameraBase, Depends(get_guide_camera)]
Shutter = Annotated[ShutterBase, Depends(get_shutter)]
NeedsLease = Depends(require_lease)


class CalibrationRequest(BaseModel):
    kind: CalibrationKind
    count: int = Field(10, ge=1, le=200, description="Frames to take; the master is their median")
    match: Literal["single_frame", "video"] = Field(
        "single_frame", description="The settings group whose gain, offset, binning and ROI to use"
    )
    exposure_s: float | None = Field(
        None,
        ge=EXPOSURE_RANGE_S[0],
        le=EXPOSURE_RANGE_S[1],
        description="A dark's exposure; the group's own if not given. A bias takes none",
    )
    require_stable_cooler: bool = Field(
        True, description="Refuse while the sensor is still cooling to its setpoint"
    )

    @model_validator(mode="after")
    def _bias_has_no_exposure(self) -> CalibrationRequest:
        if self.kind == "bias" and self.exposure_s is not None:
            raise ValueError("a bias frame is the shortest exposure; it takes no exposure_s")
        return self


def settings_for(request: CalibrationRequest, camera: GuideCameraBase) -> CaptureSettings:
    """What the frames are taken with: the matched group, with the dark's or bias's exposure."""
    group: CaptureSettings = getattr(camera.conf.snapshot(), request.match)
    if request.kind == "bias":
        return group.model_copy(update={"exposure_s": EXPOSURE_RANGE_S[0]})
    if request.exposure_s is not None:
        return group.model_copy(update={"exposure_s": request.exposure_s})
    return group


async def until_done(ctx: JobContext, camera: GuideCameraBase, task: asyncio.Task):
    """Wait for ``task`` (an exposure); on cancel, cut the exposure short and re-raise."""
    try:
        while not (await asyncio.wait({task}, timeout=0.1))[0]:
            ctx.check()
    except JobCancelled:
        camera.abort()
        with contextlib.suppress(ExposureAborted, CameraBusy):
            await task
        raise
    return task.result()


async def _run(
    ctx: JobContext,
    camera: GuideCameraBase,
    shutter: ShutterBase,
    request: CalibrationRequest,
    settings: CaptureSettings,
) -> dict:
    cal = store()
    session = cal.ensure_session()
    frames = []
    try:
        async with shutter.closed():
            for i in range(request.count):
                ctx.check()
                ctx.progress(i / request.count, f"{request.kind} {i + 1} of {request.count}")
                task = asyncio.create_task(camera.calibration_frame(request.kind, settings))
                frames.append(await until_done(ctx, camera, task))
    except ShutterBusy as e:
        raise ApiError(409, "busy", str(e), {"component": SHUTTER}) from e
    except CameraBusy as e:
        raise ApiError(409, "busy", str(e), {"component": CAMERA}) from e
    except SettingError as e:
        raise ApiError(422, "invalid", str(e)) from e
    except ExposureAborted as e:
        raise JobCancelled from e
    ctx.progress(1.0, "filing the frames and their median")
    entry = await asyncio.to_thread(cal.add, session, request.kind, frames, match=request.match)
    return {"session": session, "entry": entry.id, "master": entry.master, "count": entry.count}


@router.post("/frames", status_code=202, response_model=Job, dependencies=[NeedsLease])
async def take_calibration_frames(
    request: CalibrationRequest, camera: Camera, shutter: Shutter, me: Me
) -> Job:
    """Dark or bias frames, as a job. ``409 busy`` while streaming or while another job has the
    camera or the shutter, ``409 not_operational`` until both are started up, and
    ``409 cooler_not_stable`` for a dark while the sensor is still cooling."""
    camera.life.require_operational()
    shutter.life.require_operational()
    if camera.streaming:
        raise ApiError(409, "busy", "the guide camera is streaming", {"component": CAMERA})
    if request.require_stable_cooler and camera.cooler().state == "cooling":
        raise ApiError(
            409,
            "cooler_not_stable",
            "the sensor is still cooling to its setpoint; darks taken now would not match later "
            "ones (ask for require_stable_cooler false to take them anyway)",
        )
    settings = settings_for(request, camera)
    return manager.submit(
        f"guide-camera-{request.kind}",
        [CAMERA, SHUTTER],
        lambda ctx: _run(ctx, camera, shutter, request, settings),
        started_by=StartedBy(name=me.name, code=me.code),
    )


@router.get("/sessions", response_model=list[SessionInfo])
async def list_sessions() -> list[SessionInfo]:
    """Every calibration session, newest first; the active one is marked."""
    return await asyncio.to_thread(lambda: store().sessions())


class NewSession(BaseModel):
    label: str | None = Field(None, max_length=80)


@router.post("/sessions", response_model=SessionInfo, dependencies=[NeedsLease])
async def start_session(body: NewSession | None = None) -> SessionInfo:
    """Start a new calibration session and make it the active one."""
    return await asyncio.to_thread(lambda: store().new_session((body or NewSession()).label))


@router.post(
    "/sessions/{session_id}/activate", response_model=SessionInfo, dependencies=[NeedsLease]
)
async def activate_session(session_id: str) -> SessionInfo:
    return await asyncio.to_thread(lambda: store().activate(session_id))


@router.delete(
    "/sessions/{session_id}/entries/{entry_id}",
    response_model=SessionInfo,
    dependencies=[NeedsLease],
)
async def discard_entry(session_id: str, entry_id: str) -> SessionInfo:
    """Delete an entry's frames and master from the disk; answers with the updated session."""
    return await asyncio.to_thread(lambda: store().delete_entry(session_id, entry_id))


@router.get("/sessions/{session_id}/entries/{entry_id}/{name}")
async def get_file(session_id: str, entry_id: str, name: str) -> FileResponse:
    """One filed FITS frame, or the entry's ``master.fits``."""
    path = store().path(session_id, entry_id, name)
    return FileResponse(
        path,
        media_type="application/fits",
        headers={"Content-Disposition": f'attachment; filename="{entry_id}-{name}"'},
    )
