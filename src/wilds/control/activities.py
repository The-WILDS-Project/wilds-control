"""What each component is doing right now (plan §3.13).

A component owns an :class:`Activities`, built on its own ``Enum`` of things it can be doing
(``Slewing``, ``Exposing``, ``Cancelling``...). The state is the set of running members, kept in
start order. It goes on the wire as *names*, never as a bitmask, so adding, reordering or
removing a member cannot change what a consumer reads.

This module imports no FastAPI, so it can move into another repo if one ever needs to start
activities itself.
"""

import asyncio
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

log = logging.getLogger("wilds.control.activities")


@dataclass(frozen=True)
class ActivityEvent:
    component: str
    activity: str
    kind: str  # "start" or "end"
    details: tuple[str, ...] = ()
    duration_s: float | None = None  # on "end" only


@dataclass(frozen=True)
class ActivityInfo:
    name: str
    since_ms: int  # wall clock at start, ms since the Unix epoch, UTC
    details: tuple[str, ...] = ()


@dataclass
class _Running:
    started_wall_ms: int
    started_mono: float  # monotonic: an NTP step cannot skew or negate the duration
    details: tuple[str, ...]


class Activities:
    """The running activities of one component. Thread-safe."""

    def __init__(self, component: str, members: type[Enum]):
        self.component = component
        self.members = members
        self._lock = threading.Lock()
        self._running: dict[Enum, _Running] = {}  # insertion order = start order
        self._listeners: list[Callable[[ActivityEvent], None]] = []

    def on_change(self, listener: Callable[[ActivityEvent], None]) -> None:
        """Call ``listener`` on every start and end. It must not block; its exceptions are
        logged and swallowed, since an announcement can never fail what it announces."""
        self._listeners.append(listener)

    def _check(self, activity: Enum) -> None:
        if not isinstance(activity, self.members):
            raise TypeError(f"{activity!r} is not a {self.members.__name__} of {self.component}")

    def _emit(self, event: ActivityEvent) -> None:
        for listener in list(self._listeners):
            try:
                listener(event)
            except Exception:
                log.exception("%s: listener failed on %s", self.component, event)

    def start(
        self, activity: Enum, *, details: list[str] | None = None, existing_ok: bool = False
    ) -> None:
        """Mark ``activity`` as running. Already running: an error, unless ``existing_ok``,
        which leaves the original start time and details alone."""
        self._check(activity)
        with self._lock:
            if activity in self._running:
                if existing_ok:
                    return
                raise RuntimeError(f"{self.component}: {activity.name} is already running")
            self._running[activity] = _Running(
                time.time_ns() // 1_000_000, time.monotonic(), tuple(details or ())
            )
            info = self._running[activity].details
        log.info(
            "%s: started %s%s", self.component, activity.name, f" {list(info)}" if info else ""
        )
        self._emit(ActivityEvent(self.component, activity.name, "start", info))

    def end(self, activity: Enum) -> None:
        """Mark ``activity`` as finished. Not running: nothing happens."""
        self._check(activity)
        with self._lock:
            run = self._running.pop(activity, None)
        if run is None:
            return
        duration = time.monotonic() - run.started_mono
        log.info("%s: ended %s after %.3f s", self.component, activity.name, duration)
        self._emit(ActivityEvent(self.component, activity.name, "end", run.details, duration))

    def is_active(self, activity: Enum) -> bool:
        self._check(activity)
        with self._lock:
            return activity in self._running

    def is_idle(self) -> bool:
        with self._lock:
            return not self._running

    @property
    def names(self) -> list[str]:
        with self._lock:
            return [a.name for a in self._running]

    def snapshot(self) -> list[ActivityInfo]:
        with self._lock:
            return [
                ActivityInfo(a.name, r.started_wall_ms, r.details) for a, r in self._running.items()
            ]

    def wait_clear(self, activity: Enum, *, timeout: float, interval: float = 0.2) -> bool:
        """Wait, bounded, for ``activity`` to end. False on timeout, with the activity **left
        set**, so the component still reports it and the caller can report the failure."""
        deadline = time.monotonic() + timeout
        while self.is_active(activity):
            if time.monotonic() >= deadline:
                log.error(
                    "%s: %s did not clear within %s s", self.component, activity.name, timeout
                )
                return False
            time.sleep(interval)
        return True

    async def await_clear(self, activity: Enum, *, timeout: float, interval: float = 0.2) -> bool:
        """:meth:`wait_clear` for coroutines."""
        deadline = time.monotonic() + timeout
        while self.is_active(activity):
            if time.monotonic() >= deadline:
                log.error(
                    "%s: %s did not clear within %s s", self.component, activity.name, timeout
                )
                return False
            await asyncio.sleep(interval)
        return True


class Registry:
    """Every component's :class:`Activities`. Ships empty: a component registers with the code
    that sets its activities, so no member exists before something starts it."""

    def __init__(self):
        self._components: dict[str, Activities] = {}

    def register(self, component: str, members: type[Enum]) -> Activities:
        if component in self._components:
            raise ValueError(f"{component} already has activities registered")
        self._components[component] = Activities(component, members)
        return self._components[component]

    def get(self, component: str) -> Activities:
        return self._components[component]

    def snapshot(self) -> dict[str, list[ActivityInfo]]:
        """Only the components with something running."""
        out = {c: a.snapshot() for c, a in self._components.items()}
        return {c: infos for c, infos in out.items() if infos}

    def clear(self) -> None:
        """Forget every component (for tests)."""
        self._components.clear()


registry = Registry()
