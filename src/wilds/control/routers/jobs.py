"""GET /jobs, GET /jobs/{id}, POST /jobs/{id}/cancel (plan §2, §3.14).

Starting a job is the business of each procedure's own router (``manager.submit``, answering
``202`` with the job); none exists yet.
"""

from fastapi import APIRouter, Depends

from ..jobs import Job, manager
from .control import require_lease

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("", response_model=list[Job])
async def list_jobs() -> list[Job]:
    return manager.list()


@router.get("/{job_id}", response_model=Job)
async def get_job(job_id: str) -> Job:
    return manager.get(job_id)


@router.post("/{job_id}/cancel", response_model=Job, dependencies=[Depends(require_lease)])
async def cancel_job(job_id: str) -> Job:
    """Whoever holds the lease may cancel, not only the session that started the job."""
    return manager.cancel(job_id)
