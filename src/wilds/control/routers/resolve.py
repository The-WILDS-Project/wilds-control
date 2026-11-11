"""Turn an object name into J2000 coordinates (plan §3.3): ``GET /resolve?name=M31``.

Anyone may ask. The answer says which service gave it (``resolver``, ``database``) and the
identifier it matched (``canonical_name``): a misresolution looks like success everywhere
below this, so the operator is shown what was matched. A moving target (a planet, a comet, a
minor planet) is refused with ``422``; a name nobody knows is ``404``.
"""

import asyncio
from typing import Annotated

from fastapi import APIRouter, Query
from pydantic import BaseModel

from ..errors import ApiError
from ..object_resolver import MovingTargetError, ObjectNameError, resolve_object_name

router = APIRouter(tags=["resolve"])


class Resolved(BaseModel):
    name: str
    ra_j2000_hours: float
    dec_j2000_degs: float
    resolver: str
    canonical_name: str | None = None
    database: str | None = None


@router.get("/resolve", response_model=Resolved)
async def resolve(name: Annotated[str, Query(min_length=1, max_length=80)]) -> Resolved:
    try:
        found = await asyncio.to_thread(resolve_object_name, name)
    except MovingTargetError as e:
        raise ApiError(422, "invalid", str(e)) from e
    except ObjectNameError as e:
        raise ApiError(404, "not_found", str(e)) from e
    return Resolved(**found.as_dict())
