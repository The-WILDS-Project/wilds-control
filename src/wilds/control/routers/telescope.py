"""The telescope: send a target, offsets and a rotator angle, and follow them (plan §3.3, §3.12).

Anyone may read. Every command needs a control lease that the TO granted (engineering mode
does not command the telescope) and is a job that reserves the ``telescope`` component, so a
second one is refused with ``409 busy`` while one runs. With the TCS's External Command Preview
on, a target or rotator command waits for the TO to accept it; the job's message says so
(``sent``, ``waiting for the TO to accept …``, ``in progress``) and it ends ``refused`` if the
TO does not accept in time. Offsets are not previewed. Progress is always read from the
telemetry, never assumed from having sent the command.
"""

import math
from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from ..errors import ApiError
from ..jobs import Job, JobContext, StartedBy, manager
from ..lease import state as lease
from ..telescope import (
    COMPONENT,
    Offset,
    OffsetNum,
    OffsetType,
    RotatorFrame,
    Target,
    TelescopeActivity,
    TelescopeBase,
    TelescopeError,
    TelescopeStatus,
    get_telescope,
)
from ..telescope.operations import confirm
from .control import Me, require_lease

router = APIRouter(prefix="/telescope", tags=["telescope"])

Telescope = Annotated[TelescopeBase, Depends(get_telescope)]
NeedsLease = Depends(require_lease)


def guard(telescope: TelescopeBase) -> None:
    """Refuse, with the reason, a command that cannot be sent now."""
    if lease.engineering:
        raise ApiError(403, "lease_required", telescope.why_not_available(engineering=True))
    telescope.life.require_operational()
    if why := telescope.why_not_available():
        raise ApiError(409, "not_operational", why)


def on_target(telescope: TelescopeBase) -> None:
    """Offsets and rotation are adjustments of a target the telescope is on: ``409 conflict``
    until it is."""
    if why := telescope.why_not_on_target():
        raise ApiError(409, "conflict", why)


def _refused(e: TelescopeError) -> ApiError:
    return ApiError(409, "refused", str(e))


def _close(a: float | None, b: float, tol: float = 0.05) -> bool:
    return a is not None and abs(a - b) <= tol


def _angle_close(a: float | None, b: float, tol: float = 0.05) -> bool:
    return a is not None and abs((a - b + 180.0) % 360.0 - 180.0) <= tol


def coordinate_name(ra_hours: float, dec_deg: float) -> str:
    """The name a target gets when none was given: its coordinates, ``J033000.0+211500``."""
    ra = round(ra_hours * 36000)  # tenths of a second of time
    dec = round(abs(dec_deg) * 3600)
    h, rest = divmod(ra % 864000, 36000)
    m, tenths = divmod(rest, 600)
    d, rest = divmod(dec, 3600)
    return f"J{h:02d}{m:02d}{tenths / 10:04.1f}{'-' if dec_deg < 0 else '+'}{d:02d}{rest // 60:02d}{rest % 60:02d}"


class SlewRequest(BaseModel):
    """Where to point. A name is optional: the coordinates are what is sent, and a target with no
    name is called by them (the TCS wants one)."""

    name: str | None = Field(None, min_length=1, max_length=80)
    ra_hours: float = Field(ge=0, lt=24, description="J2000")
    dec_deg: float = Field(ge=-90, le=90, description="J2000")
    rot_pa: float = Field(0.0, description="The slit's sky PA, degrees East of North")
    rot_frame: RotatorFrame = "Target"


class RotatorRequest(BaseModel):
    pa: float = Field(description="The slit's sky PA, degrees East of North")
    frame: RotatorFrame | None = Field(None, description="Keep the current frame if not given")


class OffsetRequest(BaseModel):
    d1_arcsec: float = Field(description="RA (or Az), arcsec")
    d2_arcsec: float = Field(description="Dec (or El), arcsec")
    num: OffsetNum = Field("Handset", description="Handset: acquisition tweaks; User: dithers")
    type: OffsetType = "TPLANE"


class OffsetWhich(BaseModel):
    num: OffsetNum = "Handset"


def _submit(kind: str, me: Session, run) -> Job:  # noqa: F821 - Session is in control.Me
    return manager.submit(
        kind,
        [COMPONENT],
        run,
        started_by=StartedBy(name=me.name, code=me.code),
    )


@router.get("", response_model=TelescopeStatus)
async def get_telescope_status(telescope: Telescope) -> TelescopeStatus:
    return telescope.status()


@router.post("/slew", status_code=202, response_model=Job, dependencies=[NeedsLease])
async def slew(body: SlewRequest, telescope: Telescope, me: Me) -> Job:
    """Send a target (and the slit's sky PA). The job ends when the telescope is in position on
    it; with Preview on it waits for the TO first."""
    guard(telescope)
    target = Target(
        name=body.name or coordinate_name(body.ra_hours, body.dec_deg),
        ra_hours=body.ra_hours,
        dec_deg=body.dec_deg,
    )

    async def run(ctx: JobContext) -> dict:
        ctx.start_activity(COMPONENT, TelescopeActivity.Slewing, details=[target.name])
        try:
            await telescope.send_target(target, body.rot_pa, body.rot_frame)
        except TelescopeError as e:
            raise _refused(e) from e
        await confirm(
            ctx,
            telescope,
            lambda t: (
                t.target is not None
                and t.target.name == target.name
                and t.in_position is True
                and _angle_close(t.rot_pa, body.rot_pa % 360.0)
            ),
            f"target {target.name}",
            took_effect=lambda t: t.target is not None and t.target.name == target.name,
        )
        return {"target": target.name}

    return _submit("telescope-slew", me, run)


@router.post("/rotator", status_code=202, response_model=Job, dependencies=[NeedsLease])
async def rotate(body: RotatorRequest, telescope: Telescope, me: Me) -> Job:
    """Rotate the slit to a sky PA, keeping the target (the TCS has no rotator-only command, so
    the current target is sent again). Needs a target the telescope is on (``409 conflict``)."""
    guard(telescope)
    on_target(telescope)
    pa = body.pa % 360.0

    async def run(ctx: JobContext) -> dict:
        ctx.start_activity(COMPONENT, TelescopeActivity.Rotating, details=[f"{pa:.2f}°"])
        try:
            await telescope.send_rotator(pa, body.frame)
        except TelescopeError as e:
            raise _refused(e) from e
        await confirm(
            ctx,
            telescope,
            lambda t: (
                _angle_close(t.rot_goal, pa)
                and _angle_close(t.rot_pa, pa)
                and t.in_position is True
            ),
            f"rotation to {pa:.2f}°",
            took_effect=lambda t: _angle_close(t.rot_goal, pa),
        )
        return {"pa": pa}

    return _submit("telescope-rotator", me, run)


def _offset_done(telescope: TelescopeBase, num: OffsetNum, want: Offset):
    def done(t) -> bool:
        have = t.offsets.get(num, Offset())
        return (
            _close(have.d1_arcsec, want.d1_arcsec, 0.005)
            and _close(have.d2_arcsec, want.d2_arcsec, 0.005)
            and t.in_position is not False
        )

    return done


@router.post("/offset", status_code=202, response_model=Job, dependencies=[NeedsLease])
async def offset(body: OffsetRequest, telescope: Telescope, me: Me) -> Job:
    """Set an offset register to these arcseconds (the register's new total, not a step). Needs
    a target the telescope is on (``409 conflict``)."""
    guard(telescope)
    on_target(telescope)
    want = Offset(d1_arcsec=body.d1_arcsec, d2_arcsec=body.d2_arcsec)
    limit = telescope.conf.current.max_offset_arcsec
    if (
        not all(math.isfinite(v) for v in (want.d1_arcsec, want.d2_arcsec))
        or max(abs(want.d1_arcsec), abs(want.d2_arcsec)) > limit
    ):
        raise ApiError(422, "invalid", f"an offset is at most {limit:g} arcsec")

    async def run(ctx: JobContext) -> dict:
        ctx.start_activity(COMPONENT, TelescopeActivity.Offsetting)
        try:
            await telescope.send_offset(want, body.num, body.type)
        except TelescopeError as e:
            raise _refused(e) from e
        await confirm(ctx, telescope, _offset_done(telescope, body.num, want), "the offset")
        return want.model_dump()

    return _submit("telescope-offset", me, run)


@router.post("/offset/clear", status_code=202, response_model=Job, dependencies=[NeedsLease])
async def clear_offset(telescope: Telescope, me: Me, body: OffsetWhich | None = None) -> Job:
    """Set an offset register back to 0, 0."""
    guard(telescope)
    on_target(telescope)
    num = (body or OffsetWhich()).num

    async def run(ctx: JobContext) -> dict:
        ctx.start_activity(COMPONENT, TelescopeActivity.Offsetting)
        try:
            await telescope.send_clear(num)
        except TelescopeError as e:
            raise _refused(e) from e
        await confirm(ctx, telescope, _offset_done(telescope, num, Offset()), "clearing the offset")
        return {"cleared": num}

    return _submit("telescope-offset-clear", me, run)


@router.post("/offset/absorb", status_code=202, response_model=Job, dependencies=[NeedsLease])
async def absorb_offset(telescope: Telescope, me: Me, body: OffsetWhich | None = None) -> Job:
    """Bake an offset register into the target's coordinates and clear it."""
    guard(telescope)
    on_target(telescope)
    num = (body or OffsetWhich()).num

    async def run(ctx: JobContext) -> dict:
        ctx.start_activity(COMPONENT, TelescopeActivity.Offsetting)
        try:
            await telescope.send_absorb(num)
        except TelescopeError as e:
            raise _refused(e) from e
        await confirm(
            ctx, telescope, _offset_done(telescope, num, Offset()), "absorbing the offset"
        )
        return {"absorbed": num}

    return _submit("telescope-offset-absorb", me, run)
