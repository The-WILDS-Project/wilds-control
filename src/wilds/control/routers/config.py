"""Settings over HTTP (plan §3.10): every component's operational (``conf``) and calibration
(``cal``) slice, read as a typed form and saved as a whole.

``GET`` gives the values, the JSON schema (labels, units, limits) and the version. ``PUT`` sends
the changed values plus the version the form was loaded at; a stale version is ``409 stale``, so a
form never silently overwrites someone else's save.
"""

import asyncio
from typing import Annotated, Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from ..conf import ChangeBusy, ConfHandle, InvalidChange, Slice, confs
from ..errors import ApiError
from ..settings_store import Entry, SettingsConflict, UnknownVersion
from .control import holder_who, require_lease

router = APIRouter(prefix="/config", tags=["config"])
NeedsLease = Depends(require_lease)


class ConfView(BaseModel):
    component: str
    slice: Slice
    version: int
    values: dict[str, Any]
    json_schema: dict[str, Any]


class ConfChange(BaseModel):
    values: dict[str, Any]  # only the fields that change
    version: int  # the version the form was loaded at
    reason: str | None = None


class RevertRequest(BaseModel):
    reason: str | None = None


class ConfIndex(BaseModel):
    components: dict[str, list[Slice]]


def _handle(component: str, slice: str) -> ConfHandle[Any]:
    if slice not in ("conf", "cal"):
        raise ApiError(404, "not_found", f"no such slice: {slice}")
    try:
        return confs.get(component, slice)  # type: ignore[arg-type]
    except KeyError:
        raise ApiError(404, "not_found", f"{component} has no {slice} settings") from None


Handle = Annotated[ConfHandle[Any], Depends(_handle)]


def _view(h: ConfHandle[Any]) -> ConfView:
    cur, version = h.current, h.version
    return ConfView(
        component=h.component,
        slice=h.slice,
        version=version,
        values=cur.model_dump(mode="json"),
        json_schema=h.schema(),
    )


async def _change(h: ConfHandle[Any], call, *args, **kwargs) -> ConfView:
    try:
        await asyncio.to_thread(call, *args, **kwargs)
    except SettingsConflict as e:
        raise ApiError(
            409,
            "stale",
            f"{h.component} was changed by someone else (now version {e.actual})",
            {"component": h.component, "version": e.actual},
        ) from e
    except UnknownVersion as e:
        raise ApiError(404, "not_found", str(e)) from e
    except ChangeBusy as e:
        raise ApiError(409, "busy", str(e), {"component": h.component}) from e
    except InvalidChange as e:
        raise ApiError(422, "invalid", str(e)) from e
    return _view(h)


@router.get("", response_model=ConfIndex)
async def index() -> ConfIndex:
    return ConfIndex(components={c: confs.slices(c) for c in confs.components()})


@router.get("/{component}/{slice}", response_model=ConfView)
async def get(h: Handle) -> ConfView:
    return _view(h)


@router.put("/{component}/{slice}", response_model=ConfView, dependencies=[NeedsLease])
async def put(change: ConfChange, h: Handle) -> ConfView:
    """Save the changed fields (the rest keep their values) if the section is still at
    ``version``; recorded under the lease holder's name."""
    return await _change(
        h,
        h.patch,
        change.values,
        who=holder_who(),
        reason=change.reason,
        expected_version=change.version,
    )


@router.get("/{component}/{slice}/history", response_model=list[Entry])
async def history(h: Handle, limit: int = 50) -> list[Entry]:
    return h.history(limit)


@router.post(
    "/{component}/{slice}/revert/{version}", response_model=ConfView, dependencies=[NeedsLease]
)
async def revert(version: int, h: Handle, request: RevertRequest | None = None) -> ConfView:
    """Restore an earlier version as a new one."""
    reason = request.reason if request else None
    return await _change(h, h.revert, version, who=holder_who(), reason=reason)
