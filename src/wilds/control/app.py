from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import acquire, adc, camera, focus, slit, telemetry


@asynccontextmanager
async def lifespan(app: FastAPI):
    # TODO: initialise broker, camera, solver
    yield
    # TODO: teardown


def create_app() -> FastAPI:
    app = FastAPI(title="WILDS Control", version="0.1.0", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(camera.router)
    app.include_router(telemetry.router)
    app.include_router(acquire.router)
    app.include_router(slit.router)
    app.include_router(adc.router)
    app.include_router(focus.router)

    return app
