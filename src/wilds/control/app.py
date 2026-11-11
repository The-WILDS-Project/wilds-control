from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import config, errors, paths
from .filing import close_filing, get_filing
from .guide_camera import close_guide_camera, get_guide_camera
from .jobs import manager as job_manager
from .lease import state as lease
from .lifecycle import system
from .routers import (
    acquire_science_target,
    adc,
    camera,
    control,
    dev,
    focus,
    guide_calibration,
    guide_camera,
    jobs,
    lamps,
    shutter,
    slit,
    status,
)
from .routers import (
    config as config_router,
)
from .routers import (
    system as system_router,
)
from .shutter import close_shutter, get_shutter


@asynccontextmanager
async def lifespan(app: FastAPI):
    # TODO: initialise broker, solver
    paths.require_root()
    get_guide_camera()
    get_shutter()
    get_filing().start()
    if config.get().autostart:
        await system.startup()
    yield
    close_filing()
    await close_guide_camera()
    await close_shutter()


def create_app() -> FastAPI:
    app = FastAPI(title="WILDS Control", version="0.1.0", lifespan=lifespan)

    errors.install(app)
    lease.on_holder_change(job_manager.handover)  # queued jobs are dropped at handover

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(camera.router)
    app.include_router(guide_camera.router)
    app.include_router(guide_calibration.router)
    app.include_router(shutter.router)
    app.include_router(config_router.router)
    app.include_router(system_router.router)
    app.include_router(status.router)
    app.include_router(control.router)
    app.include_router(jobs.router)
    app.include_router(acquire_science_target.router)
    app.include_router(slit.router)
    app.include_router(lamps.router)
    app.include_router(adc.router)
    app.include_router(focus.router)
    app.include_router(dev.router)  # refuses itself unless opmode is development

    return app
