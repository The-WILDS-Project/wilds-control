"""Is light reaching WILDS? (plan §3.17)

The guide camera's darks need the light path blocked, and only the Telescope Operator can do
that (close the M1 cover, move a fold mirror, whatever the telescope needs). So blocking is a
conversation between two people, kept here:

    lit  --request block-->  block_requested  --TO acknowledges-->  blocked
    blocked  --request restore-->  unblock_requested  --TO acknowledges-->  lit

The instrument operator (who holds control) asks and may cancel; the TO acknowledges, refuses
with a reason (the state goes back to what it was), or sets the state directly (they may need
the light back at any time). The state is only what people said: it starts ``unknown`` after a
restart and is never restored from an older claim. No telemetry feeds it yet (plan §3.17 lists
the M1 cover, the instrument cover and the fold mirrors as candidates).
"""

import time
from typing import Literal

from pydantic import BaseModel

from .errors import ApiError

LightState = Literal["lit", "block_requested", "blocked", "unblock_requested", "unknown"]


class LightPathStatus(BaseModel):
    state: LightState = "unknown"
    since_ms: int = 0  # when the state last changed, ms since the Unix epoch, UTC
    requested_by: str | None = None  # the name behind a pending request
    note: str | None = None  # the TO's note with an acknowledgement
    refusal: str | None = None  # why the TO turned the last request down (until the next one)


def _now_ms() -> int:
    return time.time_ns() // 1_000_000


class LightPath:
    def __init__(self) -> None:
        self._state: LightState = "unknown"
        self._before: LightState = "unknown"  # what a cancelled or refused request goes back to
        self._since = _now_ms()
        self._requested_by: str | None = None
        self._note: str | None = None
        self._refusal: str | None = None

    @property
    def state(self) -> LightState:
        return self._state

    @property
    def blocked(self) -> bool:
        return self._state == "blocked"

    def status(self) -> LightPathStatus:
        return LightPathStatus(
            state=self._state,
            since_ms=self._since,
            requested_by=self._requested_by,
            note=self._note,
            refusal=self._refusal,
        )

    def _go(self, state: LightState, *, by: str | None = None, note: str | None = None) -> None:
        self._state = state
        self._since = _now_ms()
        self._requested_by = by
        self._note = note

    @staticmethod
    def _conflict(message: str) -> ApiError:
        return ApiError(409, "conflict", message)

    def request_block(self, by: str) -> None:
        if self._state in ("block_requested", "unblock_requested"):
            raise self._conflict(f"a request is already waiting for the TO ({self._state})")
        if self._state == "blocked":
            raise self._conflict("the light path is already blocked")
        self._before = self._state
        self._refusal = None
        self._go("block_requested", by=by)

    def request_restore(self, by: str) -> None:
        if self._state in ("block_requested", "unblock_requested"):
            raise self._conflict(f"a request is already waiting for the TO ({self._state})")
        if self._state == "lit":
            raise self._conflict("the light path is already open")
        self._before = self._state
        self._refusal = None
        self._go("unblock_requested", by=by)

    def cancel(self) -> None:
        if self._state not in ("block_requested", "unblock_requested"):
            raise self._conflict("there is no request to cancel")
        self._go(self._before)

    def acknowledge(self, note: str | None = None) -> None:
        """The TO has done what was asked."""
        if self._state == "block_requested":
            self._go("blocked", note=note)
        elif self._state == "unblock_requested":
            self._go("lit", note=note)
        else:
            raise self._conflict("there is no request to acknowledge")

    def refuse(self, reason: str) -> None:
        if self._state not in ("block_requested", "unblock_requested"):
            raise self._conflict("there is no request to refuse")
        self._go(self._before)
        self._refusal = reason

    def set(self, state: Literal["lit", "blocked"], note: str | None = None) -> None:
        """The TO says how things are, with or without a request (they may need the light back)."""
        self._refusal = None
        self._go(state, note=note)


_light_path: LightPath | None = None


def get_light_path() -> LightPath:
    global _light_path
    if _light_path is None:
        _light_path = LightPath()
    return _light_path


def reset() -> None:
    """Forget the state (for tests)."""
    global _light_path
    _light_path = None
