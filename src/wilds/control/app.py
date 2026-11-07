from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import errors
from .jobs import manager as job_manager
from .lease import state as lease
from .routers import (
    acquire_science_target,
    adc,
    camera,
    control,
    dev,
    focus,
    jobs,
    lamps,
    slit,
    status,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # TODO: initialise broker, camera, solver
    yield
    # TODO: teardown


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
