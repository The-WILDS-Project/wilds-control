"""Jobs (plan §3.14).

A job is one long procedure that control runs. It lives in the backend, so it survives a
closed tab. States: ``queued`` → ``running`` → ``done``, ``failed`` or ``cancelled``.

* A job declares the components it uses; two jobs never use the same component at once. The
  second is refused (``409 busy``) or, if it asked, queued until they are free.
* Whatever the job started in :class:`~wilds.control.activities.Activities` it ends again,
  whether it finishes, fails or is cancelled, so a dead job never leaves a component busy.
* Cancel is cooperative: it sets ``cancel_requested`` and raises ``Cancelling`` on the job's
  components (those that have such a member); the runner notices through its context.
* At handover a running job carries on under the new holder; a queued one is dropped.

Everything runs on the event loop. State is in memory: a restart drops it all.
"""

import asyncio
import itertools
import logging
import time
from collections import deque
from collections.abc import Awaitable, Callable
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

from .activities import registry
from .errors import ApiError, ErrorBody

audit = logging.getLogger("wilds.control.audit")
log = logging.getLogger("wilds.control.jobs")

KEEP_FINISHED = 100  # finished jobs remembered
RECENT_IN_STATUS = 5  # finished jobs shown in /status

JobState = Literal["queued", "running", "done", "failed", "cancelled"]


class StartedBy(BaseModel):
    name: str
    code: str


class Progress(BaseModel):
    fraction: float | None = None  # 0..1, None when unknown
    message: str = ""


class Job(BaseModel):
    id: str
    kind: str
    state: JobState = "queued"
    components: list[str]
    started_by: StartedBy
    created_at: int  # ms since the Unix epoch, UTC
    started_at: int | None = None
    ended_at: int | None = None
    duration_s: float | None = None  # monotonic clock
    progress: Progress = Field(default_factory=Progress)
    activities: dict[str, list[str]] = Field(default_factory=dict)  # raised and still running
    result: Any = None
    error: ErrorBody | None = None
    cancel_requested: bool = False


class JobCancelled(Exception):
    """Raised by :meth:`JobContext.check` / :meth:`JobContext.sleep` once cancel was requested."""


Runner = Callable[["JobContext"], Awaitable[Any]]


def _now_ms() -> int:
    return time.time_ns() // 1_000_000


class JobContext:
    """What a runner is given: progress, cancellation and activities."""

    def __init__(self, manager: JobManager, job: Job):
        self._manager = manager
        self.job = job
        self._cancel = asyncio.Event()
        self._started: list[tuple[str, Enum]] = []
        self._start_mono: float | None = None

    @property
    def cancel_requested(self) -> bool:
        return self._cancel.is_set()

    def check(self) -> None:
        """Raise :class:`JobCancelled` if cancel was requested. Call between steps."""
        if self._cancel.is_set():
            raise JobCancelled

    async def sleep(self, seconds: float) -> None:
        """Sleep, but wake at once (raising :class:`JobCancelled`) if cancel is requested."""
        try:
            await asyncio.wait_for(self._cancel.wait(), seconds)
        except TimeoutError:
            return
        raise JobCancelled

    def progress(self, fraction: float | None = None, message: str | None = None) -> None:
        if fraction is not None:
            self.job.progress.fraction = max(0.0, min(1.0, fraction))
        if message is not None:
            self.job.progress.message = message
        self._manager._emit(self.job)

    def start_activity(
        self, component: str, activity: Enum, *, details: list[str] | None = None
    ) -> None:
        """Raise ``activity`` on ``component``; the job ends it at the latest when it finishes."""
        try:
            registry.get(component).start(activity, details=details)
        except RuntimeError as e:
            raise ApiError(409, "busy", str(e), {"component": component}) from e
        self._started.append((component, activity))
        self.job.activities.setdefault(component, []).append(activity.name)
        self._manager._emit(self.job)

    def end_activity(self, component: str, activity: Enum) -> None:
        if (component, activity) not in self._started:
            return
        self._started.remove((component, activity))
        registry.get(component).end(activity)
        names = self.job.activities.get(component, [])
        if activity.name in names:
            names.remove(activity.name)
        if not names:
            self.job.activities.pop(component, None)
        self._manager._emit(self.job)

    def _end_all(self) -> None:
        for component, activity in reversed(self._started):
            registry.get(component).end(activity)
        self._started.clear()
        self.job.activities.clear()


class JobManager:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._seq = itertools.count(1)
        self._jobs: dict[str, Job] = {}
        self._ctx: dict[str, JobContext] = {}
        self._runners: dict[str, Runner] = {}
        self._busy: dict[str, str] = {}  # component -> id of the running job using it
        self._queue: list[str] = []  # queued ids, oldest first
        self._finished: deque[str] = deque()
        self._subscribers: set[asyncio.Queue[Job]] = set()
        self._tasks: set[asyncio.Task[None]] = set()
        self._loop: asyncio.AbstractEventLoop | None = None

    # -- starting -----------------------------------------------------------------------

    def submit(
        self,
        kind: str,
        components: list[str],
        run: Runner,
        *,
        started_by: StartedBy,
        queue: bool = False,
    ) -> Job:
        """Create a job and start it, queue it, or refuse it with ``409 busy``."""
        self._loop = asyncio.get_running_loop()
        components = list(dict.fromkeys(components))
        blocking = {self._busy[c] for c in components if c in self._busy}
        if blocking and not queue:
            first = min(blocking, key=lambda j: int(j[2:]))
            raise ApiError(
                409,
                "busy",
                f"{', '.join(c for c in components if c in self._busy)} in use by job {first}",
                {"job_id": first},
            )
        job = Job(
            id=f"j-{next(self._seq)}",
            kind=kind,
            components=components,
            started_by=started_by,
            created_at=_now_ms(),
        )
        self._jobs[job.id] = job
        self._ctx[job.id] = JobContext(self, job)
        self._runners[job.id] = run
        audit.info("job %s (%s) created by %s [%s]", job.id, kind, started_by.name, started_by.code)
        if blocking:
            self._queue.append(job.id)
            self._emit(job)
        else:
            self._start(job.id)
        return job

    def _start(self, job_id: str) -> None:
        job, ctx = self._jobs[job_id], self._ctx[job_id]
        job.state, job.started_at = "running", _now_ms()
        ctx._start_mono = time.monotonic()
        for c in job.components:
            self._busy[c] = job_id
        audit.info("job %s started", job_id)
        self._emit(job)
        task = asyncio.get_running_loop().create_task(self._run(ctx, self._runners.pop(job_id)))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _run(self, ctx: JobContext, run: Runner) -> None:
        try:
            result = await run(ctx)
        except JobCancelled:
            self._finish(ctx, "cancelled", error=ErrorBody(code="cancelled", message="cancelled"))
        except ApiError as e:
            self._finish(ctx, "failed", error=e.body)
        except asyncio.CancelledError:
            self._finish(ctx, "cancelled", error=ErrorBody(code="cancelled", message="interrupted"))
            raise
        except Exception as e:
            log.exception("job %s failed", ctx.job.id)
            msg = str(e) or type(e).__name__
            self._finish(ctx, "failed", error=ErrorBody(code="internal", message=msg))
        else:
            self._finish(ctx, "done", result=result)

    def _finish(
        self,
        ctx: JobContext,
        state: JobState,
        *,
        result: Any = None,
        error: ErrorBody | None = None,
    ) -> None:
        job = ctx.job
        ctx._end_all()
        for c in job.components:
            if self._busy.get(c) == job.id:
                del self._busy[c]
        job.state, job.result, job.error, job.ended_at = state, result, error, _now_ms()
        if ctx._start_mono is not None:
            job.duration_s = time.monotonic() - ctx._start_mono
        self._runners.pop(job.id, None)
        self._ctx.pop(job.id, None)
        self._finished.append(job.id)
        while len(self._finished) > KEEP_FINISHED:
            self._jobs.pop(self._finished.popleft(), None)
        audit.info("job %s %s%s", job.id, state, f": {error.message}" if error else "")
        self._emit(job)
        self._pump()

    def _pump(self) -> None:
        """Start queued jobs, oldest first, whose components are now free."""
        for job_id in list(self._queue):
            if all(c not in self._busy for c in self._jobs[job_id].components):
                self._queue.remove(job_id)
                self._start(job_id)

    # -- cancelling and handover --------------------------------------------------------

    def cancel(self, job_id: str) -> Job:
        """Cancel a job. Queued: at once. Running: cooperative. Finished: nothing happens."""
        job = self.get(job_id)
        ctx = self._ctx.get(job_id)
        if job.state == "queued" and ctx:
            self._queue.remove(job_id)
            audit.info("job %s cancelled while queued", job_id)
            self._finish(ctx, "cancelled", error=ErrorBody(code="cancelled", message="cancelled"))
        elif job.state == "running" and ctx and not job.cancel_requested:
            job.cancel_requested = True
            ctx._cancel.set()
            for component in job.components:
                try:
                    comp = registry.get(component)
                    cancelling = comp.members["Cancelling"]
                except KeyError:
                    continue  # not registered, or no Cancelling member
                try:
                    comp.start(cancelling)
                except RuntimeError:
                    continue  # already cancelling for someone else; theirs to end
                ctx._started.append((component, cancelling))
                job.activities.setdefault(component, []).append("Cancelling")
            audit.info("job %s: cancel requested", job_id)
            self._emit(job)
        return job

    def handover(self) -> None:
        """The lease changed hands: drop queued jobs (running ones carry on). Thread-safe."""
        loop = self._loop
        if loop is None:
            return
        try:
            same = asyncio.get_running_loop() is loop
        except RuntimeError:
            same = False
        if same:
            self._drop_queued()
        elif not loop.is_closed():
            loop.call_soon_threadsafe(self._drop_queued)

    def _drop_queued(self) -> None:
        for job_id in list(self._queue):
            ctx = self._ctx.get(job_id)
            if ctx is None:
                continue
            self._queue.remove(job_id)
            audit.info("job %s dropped: holder changed", job_id)
            self._finish(
                ctx, "cancelled", error=ErrorBody(code="cancelled", message="holder changed")
            )

    # -- reading ------------------------------------------------------------------------

    def busy_job(self, component: str) -> str | None:
        """The id of the running job that reserves ``component``, or ``None``."""
        return self._busy.get(component)

    def get(self, job_id: str) -> Job:
        try:
            return self._jobs[job_id]
        except KeyError:
            raise ApiError(404, "not_found", f"no job {job_id}") from None

    def list(self) -> list[Job]:
        """Every remembered job, newest first."""
        return sorted(self._jobs.values(), key=lambda j: int(j.id[2:]), reverse=True)

    def snapshot(self) -> list[Job]:
        """For /status: active jobs, then the last few finished ones."""
        active = [j for j in self.list() if j.state in ("queued", "running")][::-1]
        recent = [self._jobs[i] for i in list(self._finished)[-RECENT_IN_STATUS:]][::-1]
        return active + recent

    # -- events -------------------------------------------------------------------------

    def subscribe(self) -> asyncio.Queue[Job]:
        q: asyncio.Queue[Job] = asyncio.Queue(maxsize=100)
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[Job]) -> None:
        self._subscribers.discard(q)

    def _emit(self, job: Job) -> None:
        snapshot = job.model_copy(deep=True)
        for q in self._subscribers:
            if q.full():  # a slow viewer misses events; /status carries the state anyway
                continue
            q.put_nowait(snapshot)


manager = JobManager()
