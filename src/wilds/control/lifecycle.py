"""Starting components up and shutting them down (plan §3.16).

Every component is **powered but not activated** when it is built (``initialized``). A
``startup`` activates it and tries to make it operational (a stage finds home, a lamp warms
up, a detector cools); a ``shutdown`` deactivates it. Two separate facts follow:

* ``opstate``: where the lifecycle stands (``initializing`` while built, ``initialized``,
  ``startedup``, ``shutdown``). It changes when a startup or shutdown *call ends*, whether or not
  it got what it wanted.
* ``operational``: a bool worked out **live** from ``why_not_operational``, a list of reasons
  (empty means operational). It is computed, never stored: a stage can lose its reference long
  after a good startup.

While a call runs the component has the activity ``StartingUp`` or ``ShuttingDown``. The
:class:`System` has the same two as umbrellas over all its components, and a composite component
can use an :class:`Umbrella` over its parts: an umbrella runs from the moment any constituent
starts until the last one is done.

This module imports no FastAPI; a task that needs a component calls
:meth:`Lifecycle.require_operational`, which answers ``409 not_operational`` with the reasons.
"""

import asyncio
import contextlib
import logging
import threading
from collections.abc import Callable, Iterator
from enum import Enum, StrEnum

from pydantic import BaseModel

from .activities import Activities, ActivityEvent, registry
from .errors import ApiError

log = logging.getLogger("wilds.control.lifecycle")

STARTING_UP = "StartingUp"
SHUTTING_DOWN = "ShuttingDown"


class OpState(StrEnum):
    initializing = "initializing"  # while the component is being built
    initialized = "initialized"  # built, powered, not activated (the default)
    startedup = "startedup"  # a startup call has ended (it may not have got operational)
    shutdown = "shutdown"  # deactivated; start up again or power down


class SystemActivity(Enum):
    StartingUp = 1
    ShuttingDown = 2


class LifecycleStatus(BaseModel):
    opstate: OpState
    operational: bool
    why_not_operational: list[str] = []


class Lifecycle:
    """One component's lifecycle. The component supplies the hardware work as plain (blocking)
    callables; this runs them off the event loop and keeps the state, activities and reasons.

    ``startup()``: bring the component to the point of trying to be operational. May raise; the
    error is kept as a reason and ``opstate`` still becomes ``startedup``. Must be safe to run
    again (a retry) and to run when the hardware is already up (after a control restart).
    ``shutdown()``: deactivate. ``halt()``: stop whatever the component is doing (used when a
    shutdown is forced). ``why_not()``: the live reasons beyond lifecycle itself.
    """

    def __init__(
        self,
        component: str,
        activities: Callable[[], Activities],
        *,
        startup: Callable[[], None] | None = None,
        shutdown: Callable[[], None] | None = None,
        halt: Callable[[], object] | None = None,
        why_not: Callable[[], list[str]] | None = None,
    ):
        self.component = component
        self._activities = activities
        self._startup, self._shutdown, self._halt, self._why_not = startup, shutdown, halt, why_not
        self._opstate = OpState.initializing
        self._error: str | None = None  # why the last startup or shutdown did not complete
        self._opstate = OpState.initialized

    # -- reading ------------------------------------------------------------------------

    @property
    def activities(self) -> Activities:
        return self._activities()

    @property
    def opstate(self) -> OpState:
        return self._opstate

    def why_not_operational(self) -> list[str]:
        acts = self.activities
        reasons: list[str] = []
        if acts.is_active(acts.members[STARTING_UP]):
            reasons.append("starting up")
        elif acts.is_active(acts.members[SHUTTING_DOWN]):
            reasons.append("shutting down")
        elif self._opstate is not OpState.startedup:
            reasons.append("shut down" if self._opstate is OpState.shutdown else "not started up")
        if self._error:
            reasons.append(self._error)
        if self._opstate is OpState.startedup and self._why_not is not None:
            reasons += self._why_not()
        return reasons

    @property
    def operational(self) -> bool:
        return not self.why_not_operational()

    def status(self) -> LifecycleStatus:
        reasons = self.why_not_operational()
        return LifecycleStatus(
            opstate=self._opstate, operational=not reasons, why_not_operational=reasons
        )

    def require_operational(self) -> None:
        """``409 not_operational`` (with the reasons) unless this component can take tasks."""
        reasons = self.why_not_operational()
        if reasons:
            raise ApiError(
                409,
                "not_operational",
                f"{self.component} is not operational: {'; '.join(reasons)}",
                {"component": self.component, "why_not_operational": reasons},
            )

    # -- changing -----------------------------------------------------------------------

    def _busy(self, what: str) -> ApiError:
        return ApiError(409, "busy", f"{self.component}: {what}", {"component": self.component})

    async def startup(self) -> LifecycleStatus:
        """Activate the component and try to make it operational. A hardware failure does not
        raise: it ends as ``startedup`` and not operational, with the reason, so a retry is just
        another call. Raises ``409 busy`` if a startup or shutdown is already running."""
        acts = self.activities
        up, down = acts.members[STARTING_UP], acts.members[SHUTTING_DOWN]
        if acts.is_active(up) or acts.is_active(down):
            raise self._busy("a startup or shutdown is already running")
        acts.start(up)
        try:
            self._error = None
            if self._startup is not None:
                try:
                    await asyncio.to_thread(self._startup)
                except Exception as e:
                    log.exception("%s: startup failed", self.component)
                    self._error = f"startup failed: {e}" if str(e) else "startup failed"
            self._opstate = OpState.startedup
        finally:
            acts.end(up)
        log.info("%s: started up, %s", self.component, self._summary())
        return self.status()

    async def shutdown(self, *, force: bool = False) -> LifecycleStatus:
        """Deactivate the component. ``409 busy`` while it is doing something, unless ``force``,
        which halts that first."""
        acts = self.activities
        up, down = acts.members[STARTING_UP], acts.members[SHUTTING_DOWN]
        if acts.is_active(up) or acts.is_active(down):
            raise self._busy("a startup or shutdown is already running")
        doing = [n for n in acts.names if n not in (STARTING_UP, SHUTTING_DOWN)]
        if doing and not force:
            raise self._busy(f"busy ({', '.join(doing)}); stop it first or force the shutdown")
        acts.start(down)
        try:
            self._error = None
            try:
                if doing and self._halt is not None:
                    result = self._halt()
                    if asyncio.iscoroutine(result):
                        await result
                if self._shutdown is not None:
                    await asyncio.to_thread(self._shutdown)
            except Exception as e:
                log.exception("%s: shutdown failed", self.component)
                self._error = f"shutdown failed: {e}" if str(e) else "shutdown failed"
            self._opstate = OpState.shutdown
        finally:
            acts.end(down)
        log.info("%s: shut down, %s", self.component, self._summary())
        return self.status()

    def _summary(self) -> str:
        reasons = self.why_not_operational()
        return "operational" if not reasons else f"not operational ({'; '.join(reasons)})"


class Umbrella:
    """An activity of a parent that runs while the same-named activity of any child runs.

    It starts when the first child starts and ends when the last child ends, so it ends only when
    all its constituents are done. :meth:`hold` keeps it running across the gaps of a sequential
    job (between one child ending and the next starting)."""

    def __init__(self, parent: Callable[[], Activities], name: str):
        self._parent = parent
        self.name = name
        self._lock = threading.Lock()
        self._running = 0  # children currently doing it
        self._holds = 0

    def add(self, child: Activities) -> None:
        child.on_change(self._on_child)

    def _on_child(self, event: ActivityEvent) -> None:
        if event.activity != self.name:
            return
        with self._lock:
            self._running = max(0, self._running + (1 if event.kind == "start" else -1))
        self._update()

    def _update(self) -> None:
        parent = self._parent()
        member = parent.members[self.name]
        with self._lock:
            wanted = self._running > 0 or self._holds > 0
        if wanted:
            parent.start(member, existing_ok=True)
        else:
            parent.end(member)

    @contextlib.contextmanager
    def hold(self) -> Iterator[None]:
        with self._lock:
            self._holds += 1
        self._update()
        try:
            yield
        finally:
            with self._lock:
                self._holds -= 1
            self._update()


class SystemStatus(BaseModel):
    operational: bool
    why_not_operational: list[str] = []
    components: dict[str, LifecycleStatus] = {}


class System:
    """WILDS as a whole: every component's lifecycle, and the umbrella activities."""

    COMPONENT = "wilds"

    def __init__(self) -> None:
        self._lives: dict[str, Lifecycle] = {}
        self._umbrellas = {
            STARTING_UP: Umbrella(lambda: self.activities, STARTING_UP),
            SHUTTING_DOWN: Umbrella(lambda: self.activities, SHUTTING_DOWN),
        }

    @property
    def activities(self) -> Activities:
        try:
            return registry.get(self.COMPONENT)
        except KeyError:
            return registry.register(self.COMPONENT, SystemActivity)

    def register(self, life: Lifecycle) -> None:
        """Adopt a component (one that is built again takes over its name)."""
        self._lives[life.component] = life
        for umbrella in self._umbrellas.values():
            umbrella.add(life.activities)

    @property
    def components(self) -> list[str]:
        return sorted(self._lives)

    def get(self, component: str) -> Lifecycle:
        try:
            return self._lives[component]
        except KeyError:
            raise ApiError(404, "not_found", f"no such component: {component}") from None

    def status(self) -> SystemStatus:
        parts = {c: self._lives[c].status() for c in self.components}
        reasons = [f"{c}: {r}" for c, s in parts.items() for r in s.why_not_operational]
        return SystemStatus(operational=not reasons, why_not_operational=reasons, components=parts)

    async def startup(self, progress: Callable[[float, str], None] | None = None) -> SystemStatus:
        return await self._all(STARTING_UP, lambda life: life.startup(), progress)

    async def shutdown(
        self, *, force: bool = False, progress: Callable[[float, str], None] | None = None
    ) -> SystemStatus:
        return await self._all(SHUTTING_DOWN, lambda life: life.shutdown(force=force), progress)

    async def _all(self, name: str, call, progress) -> SystemStatus:
        lives = [self._lives[c] for c in self.components]
        done = 0
        refused: dict[str, str] = {}

        async def one(life: Lifecycle) -> None:
            nonlocal done
            try:
                await call(life)
            except ApiError as e:  # busy: not touched, the others carry on
                refused[life.component] = e.body.message
            done += 1
            if progress is not None:
                progress(done / len(lives), f"{life.component} done")

        with self._umbrellas[name].hold():
            await asyncio.gather(*(one(life) for life in lives))
        status = self.status()
        for component, why in refused.items():
            status.components[component].why_not_operational.append(why)
        return status

    def reset(self) -> None:
        """Forget every component (for tests)."""
        self._lives.clear()
        for umbrella in self._umbrellas.values():
            umbrella._running = umbrella._holds = 0


system = System()
