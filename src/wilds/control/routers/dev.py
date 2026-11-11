"""Development helpers, refused unless ``opmode`` in config.toml is ``development``.

``POST /dev/jobs/demo`` starts a made-up job ("testing 1 2 3") that raises activities on a few
components and reports progress, so the GUI's pills, umbrella bars and Cancel can be watched
without hardware. It stands in for a real procedure and goes when the first one exists.

The check is made per request, and config.toml is re-read when it changes, so switching the
mode needs no restart.
"""

import asyncio
from enum import Enum
from typing import Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from .. import config
from ..activities import registry
from ..errors import ApiError
from ..jobs import Job, JobCancelled, JobContext, StartedBy, manager
from .control import Me, require_lease


def require_development() -> None:
    if not config.get().is_development:
        raise ApiError(404, "not_found", "The /dev helpers exist only in opmode development")


router = APIRouter(prefix="/dev", tags=["dev"], dependencies=[Depends(require_development)])

COMPONENTS = ("slit", "adc", "lamps")  # not the guide camera: that has real activities


class DemoActivity(Enum):
    Moving = 1
    Exposing = 2
    Cancelling = 3


def register_components() -> None:
    for name in COMPONENTS:
        try:
            registry.register(name, DemoActivity)
        except ValueError:
            pass  # registered by an earlier request


class DemoRequest(BaseModel):
    outcome: Literal["done", "fail"] = "done"
    seconds: float = 12.0
    queue: bool = False  # wait for the components instead of answering 409 busy


WIND_DOWN_S = 1.5  # a real job takes a while to stop; long enough to see Cancelling


async def _demo(ctx: JobContext, outcome: str, seconds: float) -> dict[str, str]:
    try:
        return await _steps(ctx, outcome, seconds)
    except JobCancelled:
        await asyncio.sleep(WIND_DOWN_S)  # Cancelling stays raised until the job returns
        raise


async def _steps(ctx: JobContext, outcome: str, seconds: float) -> dict[str, str]:
    """Three phases: slit and ADC move, then the lamps flash, then a last bit."""
    third = seconds / 3
    tick = 0.25

    async def phase(n: int, message: str) -> None:
        ctx.progress(message=message)
        for i in range(max(1, int(third / tick))):
            await ctx.sleep(tick)
            ctx.progress((n + min(1.0, (i + 1) * tick / third)) / 3)

    ctx.start_activity("slit", DemoActivity.Moving, details=["testing 1"])
    ctx.start_activity("adc", DemoActivity.Moving, details=["testing 2"])
    await phase(0, "testing 1 2")
    ctx.end_activity("slit", DemoActivity.Moving)
    ctx.start_activity("lamps", DemoActivity.Exposing, details=["testing 3"])
    await phase(1, "testing 3")
    ctx.end_activity("adc", DemoActivity.Moving)
    if outcome == "fail":
        raise ApiError(409, "refused", "testing 1 2 3 went wrong", {"phase": 3})
    await phase(2, "last bit")
    return {"said": "testing 1 2 3"}


@router.post(
    "/jobs/demo", status_code=202, response_model=Job, dependencies=[Depends(require_lease)]
)
async def start_demo(me: Me, body: DemoRequest | None = None) -> Job:
    body = body or DemoRequest()
    if not 1 <= body.seconds <= 120:
        raise ApiError(422, "invalid", "seconds must be between 1 and 120")
    register_components()
    return manager.submit(
        "testing-123",
        list(COMPONENTS),
        lambda ctx: _demo(ctx, body.outcome, body.seconds),
        started_by=StartedBy(name=me.name, code=me.code),
        queue=body.queue,
    )


# -- the simulated TCS: what the TO does in the TCS GUI (plan §3.12) ------------------------------


class PreviewRequest(BaseModel):
    on: bool


def _simulated():
    from ..telescope import SimulatedTelescope, get_telescope

    telescope = get_telescope()
    if not isinstance(telescope, SimulatedTelescope):
        raise ApiError(404, "not_found", "The telescope is not simulated")
    return telescope


@router.put("/telescope/preview")
async def set_preview(body: PreviewRequest) -> dict[str, bool]:
    """Switch the simulated TCS's External Command Preview."""
    _simulated().set_preview(body.on)
    return {"preview": body.on}


@router.post("/telescope/accept")
async def accept_pending() -> dict[str, bool]:
    """The TO accepts the previewed command in the (simulated) TCS."""
    return {"accepted": _simulated().accept_pending()}


@router.post("/telescope/reject")
async def reject_pending() -> dict[str, bool]:
    """The TO rejects the previewed command in the (simulated) TCS."""
    return {"rejected": _simulated().reject_pending()}
