"""A TCS that takes its time, with External Command Preview, so WILDS can be exercised without
the telescope (plan §3.12). Motion is worked out from the clock when the telemetry is read, so
nothing runs in the background."""

import threading
import time
from typing import ClassVar

from pydantic import PrivateAttr

from .base import (
    Offset,
    OffsetNum,
    OffsetType,
    Pending,
    RotatorFrame,
    Target,
    Telemetry,
    TelescopeBase,
)


def _shortest(delta: float) -> float:
    return (delta + 180.0) % 360.0 - 180.0


class SimulatedTelescope(TelescopeBase):
    kind: ClassVar[str] = "simulated"

    slew_s: float = 3.0  # a slew takes this long
    rotate_deg_s: float = 6.0
    offset_s: float = 0.6
    preview: bool = True  # External Command Preview: the TCS default
    locked_out: bool = False
    clock: object = time.monotonic

    _lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)
    _target: Target | None = PrivateAttr(default=None)
    _rot_from: float = PrivateAttr(default=0.0)
    _rot_goal: float = PrivateAttr(default=0.0)
    _rot_t0: float = PrivateAttr(default=0.0)
    _rot_frame: RotatorFrame = PrivateAttr(default="Target")
    _slew_until: float = PrivateAttr(default=0.0)
    _offset_until: float = PrivateAttr(default=0.0)
    _offsets: dict[str, Offset] = PrivateAttr(default_factory=dict)
    _pending: tuple[Pending, tuple] | None = PrivateAttr(default=None)

    def _now(self) -> float:
        return self.clock()  # type: ignore[operator]

    # -- the rotator moves at a fixed rate along the short way round ----------------------------

    def _rot_now(self) -> float:
        span = _shortest(self._rot_goal - self._rot_from)
        moved = min(abs(span), self.rotate_deg_s * max(0.0, self._now() - self._rot_t0))
        return (self._rot_from + moved * (1 if span >= 0 else -1)) % 360.0

    def _rotate_to(self, pa: float) -> None:
        self._rot_from = self._rot_now()
        self._rot_goal = pa % 360.0
        self._rot_t0 = self._now()

    # -- the TCS: commands, previewed or not --------------------------------------------------

    def _apply(self, what: tuple) -> None:
        kind = what[0]
        if kind == "target":
            _, target, pa, frame = what
            self._target = target
            self._rot_frame = frame
            self._rotate_to(pa)
            self._slew_until = self._now() + self.slew_s
        elif kind == "rotator":
            _, pa, frame = what
            if frame is not None:
                self._rot_frame = frame
            self._rotate_to(pa)

    def _send_target(self, target: Target, rot_pa: float, rot_frame: RotatorFrame) -> None:
        with self._lock:
            what = ("target", target, rot_pa, rot_frame)
            if self.preview:
                detail = (
                    f"RA {target.ra_hours:.4f} h, Dec {target.dec_deg:+.3f}°, slit PA {rot_pa:.1f}°"
                )
                self._pending = (Pending(command="target", what=target.name, detail=detail), what)
            else:
                self._apply(what)

    def _send_rotator(self, rot_pa: float, rot_frame: RotatorFrame | None) -> None:
        with self._lock:
            what = ("rotator", rot_pa, rot_frame)
            if self.preview:  # a re-sent target is previewed too (plan §3.12)
                pending = Pending(
                    command="rotator", what=f"{rot_pa:.2f}°", detail="same target, new slit PA"
                )
                self._pending = (pending, what)
            else:
                self._apply(what)

    def _send_offset(self, num: OffsetNum, offset: Offset, type: OffsetType) -> None:
        with self._lock:
            self._offsets[num] = offset  # the register's new total; not previewed
            self._offset_until = self._now() + self.offset_s

    def _send_clear(self, num: OffsetNum) -> None:
        with self._lock:
            self._offsets[num] = Offset()
            self._offset_until = self._now() + self.offset_s

    def _send_absorb(self, num: OffsetNum) -> None:
        self._send_clear(num)  # baked into the target; the register is empty again

    def _withdraw(self) -> None:
        with self._lock:
            self._pending = None

    def _read(self) -> Telemetry:
        with self._lock:
            now = self._now()
            rot = self._rot_now()
            moving = (
                now < self._slew_until
                or now < self._offset_until
                or abs(_shortest(self._rot_goal - rot)) > 0.01
            )
            return Telemetry(
                target=self._target,
                rot_pa=rot,
                rot_goal=self._rot_goal,
                rot_frame=self._rot_frame,
                offsets={k: v.model_copy() for k, v in self._offsets.items()},
                in_position=None if self._target is None else not moving,
                preview=self.preview,
                locked_out=self.locked_out,
                state="ENABLED",
                health="GOOD",
                access_mode="Operator",
                pending=self._pending[0] if self._pending else None,
            )

    # -- what the TO does in the TCS GUI (development only; /dev/telescope) --------------------

    def accept_pending(self) -> bool:
        with self._lock:
            if self._pending is None:
                return False
            _, what = self._pending
            self._pending = None
            self._apply(what)
            return True

    def reject_pending(self) -> bool:
        with self._lock:
            had, self._pending = self._pending is not None, None
            return had

    def set_preview(self, on: bool) -> None:
        with self._lock:
            self.preview = on
