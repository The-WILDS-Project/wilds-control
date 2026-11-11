"""Start WILDS up and shut it down (plan §3.16), as a whole or one component at a time.

Either the **Telescope Operator** (passphrase) or whoever **holds control** may do this, so the TO
can get WILDS operational before any Instrument Operator has connected, and shut it down after.
Both are jobs: they take a while (a lamp warms up, a stage finds home), show on the activity
board as ``StartingUp`` / ``ShuttingDown``, and finish with each component's ``opstate`` and
``why_not_operational``. A startup that does not get a component operational is not a failed job:
the job is done and the reasons are in its result.
"""

from collections.abc import Awaitable, Callable
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Header, HTTPException

from .. import config
from ..jobs import Job, JobContext, StartedBy, manager
from ..lease import LeaseError
from ..lease import state as lease
from ..lifecycle import Lifecycle, SystemStatus, system
from .control import holder_who

router = APIRouter(prefix="/system", tags=["system"])

TO_WHO = StartedBy(name="Telescope Operator", code="TO")


def authority(
    x_to_passphrase: Annotated[str | None, Header()] = None,
    x_lease_token: Annotated[str | None, Header()] = None,
) -> StartedBy:
    """Who may start WILDS up or shut it down: the TO, or the holder of control."""
    if x_to_passphrase and config.check_to(x_to_passphrase):
        lease.touch_to()
        return TO_WHO
    try:
        lease.check_token(x_lease_token)
    except LeaseError as e:
        raise HTTPException(403, "Telescope Operator passphrase or control required") from e
    return holder_who()


Who = Annotated[StartedBy, Depends(authority)]


def submit(
    kind: str, components: list[str], run: Callable[[JobContext], Awaitable[Any]], who: StartedBy
) -> Job:
    return manager.submit(kind, components, run, started_by=who)


@router.get("", response_model=SystemStatus)
async def get_system() -> SystemStatus:
    return system.status()


@router.post("/startup", status_code=202, response_model=Job)
async def start_up(who: Who) -> Job:
    """Start every component up, together. Each reports its own result."""

    async def run(ctx: JobContext) -> dict[str, Any]:
        return (await system.startup(lambda f, m: ctx.progress(f, m))).model_dump()

    return submit("startup", system.components, run, who)


@router.post("/shutdown", status_code=202, response_model=Job)
async def shut_down(who: Who, force: bool = False) -> Job:
    """Shut every component down. A component that is busy refuses unless ``force``."""

    async def run(ctx: JobContext) -> dict[str, Any]:
        status = await system.shutdown(force=force, progress=lambda f, m: ctx.progress(f, m))
        return status.model_dump()

    return submit("shutdown", system.components, run, who)


def _life(component: str) -> Lifecycle:
    return system.get(component)


@router.post("/{component}/startup", status_code=202, response_model=Job)
async def start_component(component: str, who: Who) -> Job:
    life = _life(component)

    async def run(ctx: JobContext) -> dict[str, Any]:
        return (await life.startup()).model_dump()

    return submit("startup", [component], run, who)


@router.post("/{component}/shutdown", status_code=202, response_model=Job)
async def stop_component(component: str, who: Who, force: bool = False) -> Job:
    life = _life(component)

    async def run(ctx: JobContext) -> dict[str, Any]:
        return (await life.shutdown(force=force)).model_dump()

    return submit("shutdown", [component], run, who)
