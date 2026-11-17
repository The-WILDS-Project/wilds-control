"""An ADC that only takes its time, so the rest of WILDS can be exercised without hardware.

Two endless rotary stages (the SmarAct SR-12012 of the real ADC, closed loop, 5 micro-degrees of
resolution) that turn the short way round at a made-up 10 degrees a second, and read part of the way
there while they turn (the status shows the pairs moving). They start at 0 and are referenced at once (how the real stages are referenced is still to be agreed with SmarAct).
"""

import time
from typing import ClassVar

from pydantic import PrivateAttr

from .base import AdcBase


def _shortest(delta: float) -> float:
    return (delta + 180.0) % 360.0 - 180.0


class SimulatedAdc(AdcBase):
    kind: ClassVar[str] = "simulated"

    speed_deg_s: float = 10.0
    start_deg: tuple[float, float] = (0.0, 0.0)  # where the stages read when WILDS starts

    _at: tuple[float, float] = PrivateAttr(default=(0.0, 0.0))
    # a move in progress: (from 1, from 2, turn 1, turn 2, started, how long), or None
    _run: tuple[float, float, float, float, float, float] | None = PrivateAttr(default=None)

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        self._at = self.start_deg

    def _reference(self) -> None:
        pass

    def _read_angles(self) -> tuple[float, float] | None:
        """Where the stages are now: part of the way there while a move is in progress."""
        run = self._run
        if run is None:
            return self._at
        from1, from2, delta1, delta2, t0, travel_s = run
        done = 1.0 if travel_s <= 0 else min(1.0, (time.monotonic() - t0) / travel_s)
        return ((from1 + delta1 * done) % 360.0, (from2 + delta2 * done) % 360.0)

    def _move(self, raw1: float, raw2: float) -> None:
        delta1, delta2 = _shortest(raw1 - self._at[0]), _shortest(raw2 - self._at[1])
        travel_s = max(abs(delta1), abs(delta2)) / self.speed_deg_s
        self._run = (self._at[0], self._at[1], delta1, delta2, time.monotonic(), travel_s)
        try:
            time.sleep(travel_s + self._conf.current.settle_s)
        finally:
            self._at = (raw1 % 360.0, raw2 % 360.0)
            self._run = None
