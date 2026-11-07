"""The guide camera: settings, cooler, streaming, single frames, the video stream (plan §3.8).

Anyone may read; changing anything needs the control lease. A single frame is a job that
reserves the ``guide-camera`` component. Streaming is switched on and off, and its frames go
to every ``WS /guide-camera/stream`` viewer as a JSON metadata message followed by the JPEG
as a binary message; each viewer gets only the newest frame.
"""

import asyncio
import contextlib
from typing import Annotated

from fastapi import APIRouter, Depends, WebSocket
from fastapi.responses import Response

from ..errors import ApiError
from ..guide_camera import (
    CameraBusy,
    Cooler,
    CoolerRequest,
    ExposureAborted,
    GuideCameraBase,
    GuideCameraStatus,
    SettingError,
    Settings,
    SettingsUpdate,
    get_guide_camera,
)
from ..guide_camera.base import COMPONENT
from ..jobs import Job, JobCancelled, JobContext, StartedBy, manager
from .control import Me, require_lease

router = APIRouter(prefix="/guide-camera", tags=["guide-camera"])

Camera = Annotated[GuideCameraBase, Depends(get_guide_camera)]
NeedsLease = Depends(require_lease)


def _invalid(e: SettingError) -> ApiError:
    return ApiError(422, "invalid", str(e))


def _busy(e: CameraBusy) -> ApiError:
    return ApiError(409, "busy", str(e), {"component": COMPONENT})


@router.get("", response_model=GuideCameraStatus)
async def get_camera(camera: Camera) -> GuideCameraStatus:
    return camera.status()


@router.get("/settings", response_model=Settings)
async def get_settings(camera: Camera) -> Settings:
    return camera.settings


@router.put("/settings", response_model=Settings, dependencies=[NeedsLease])
async def put_settings(update: SettingsUpdate, camera: Camera) -> Settings:
    try:
        return camera.update_settings(update)
    except SettingError as e:
        raise _invalid(e) from e


@router.get("/cooler", response_model=Cooler)
async def get_cooler(camera: Camera) -> Cooler:
    return camera.cooler()


@router.put("/cooler", response_model=Cooler, dependencies=[NeedsLease])
async def put_cooler(request: CoolerRequest, camera: Camera) -> Cooler:
    try:
        return await asyncio.to_thread(camera.set_cooler, request)
    except SettingError as e:
        raise _invalid(e) from e


@router.post("/start", response_model=GuideCameraStatus, dependencies=[NeedsLease])
async def start(camera: Camera) -> GuideCameraStatus:
    try:
        await camera.start()
    except CameraBusy as e:
        raise _busy(e) from e
    return camera.status()


@router.post("/stop", response_model=GuideCameraStatus, dependencies=[NeedsLease])
async def stop(camera: Camera) -> GuideCameraStatus:
    await camera.stop()
    return camera.status()


async def _snap(ctx: JobContext, camera: GuideCameraBase) -> dict[str, int]:
    ctx.progress(message=f"exposing {camera.settings.exposure_s:g} s")
    task = asyncio.create_task(camera.single_frame())
    try:
        while not (await asyncio.wait({task}, timeout=0.1))[0]:
            ctx.check()
    except JobCancelled:
        camera.abort()
        with contextlib.suppress(ExposureAborted, CameraBusy):
            await task
        raise
    try:
        return {"frame_id": task.result().meta.id}
    except ExposureAborted as e:
        raise JobCancelled from e
    except CameraBusy as e:
        raise _busy(e) from e


@router.post("/frames", status_code=202, response_model=Job, dependencies=[NeedsLease])
async def take_frame(camera: Camera, me: Me) -> Job:
    """One exposure with the current settings, as a job. ``409 busy`` while streaming."""
    if camera.streaming:
        raise _busy(CameraBusy("the guide camera is streaming"))
    return manager.submit(
        "guide-camera-frame",
        [COMPONENT],
        lambda ctx: _snap(ctx, camera),
        started_by=StartedBy(name=me.name, code=me.code),
    )


@router.get("/frames/{frame_id}")
async def get_frame(frame_id: int, camera: Camera) -> Response:
    """A recent frame as FITS (the last few are kept)."""
    frame = camera.frame(frame_id)
    if frame is None:
        raise ApiError(404, "not_found", f"frame {frame_id} is not among the recent frames")
    return Response(
        await asyncio.to_thread(frame.fits),
        media_type="application/fits",
        headers={"Content-Disposition": f'attachment; filename="guide-{frame_id}.fits"'},
    )


@router.websocket("/stream")
async def stream(ws: WebSocket, camera: Camera) -> None:
    await ws.accept()
    queue = camera.subscribe()
    listen = asyncio.create_task(ws.receive())  # finishes when the viewer goes away
    try:
        if (latest := camera.video_frame) is not None:
            await ws.send_json(latest.meta.model_dump())
            await ws.send_bytes(latest.jpeg)
        while True:
            get = asyncio.create_task(queue.get())
            done, _ = await asyncio.wait({get, listen}, return_when=asyncio.FIRST_COMPLETED)
            if get not in done:
                get.cancel()
                break
            frame = get.result()
            await ws.send_json(frame.meta.model_dump())
            await ws.send_bytes(frame.jpeg)
    except Exception:  # the socket closed under us
        pass
    finally:
        listen.cancel()
        camera.unsubscribe(queue)
        with contextlib.suppress(Exception):
            await ws.close()
