"""An ADC that only takes its time, so the rest of WILDS can be exercised without hardware.

Two endless rotary stages (the SmarAct SR-12012 of the real ADC, closed loop, 5 micro-degrees of
resolution) that turn the short way round at a made-up 10 degrees a second. They start at 0 and
are referenced at once (how the real stages are referenced is still to be agreed with SmarAct).
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

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        self._at = self.start_deg

    def _reference(self) -> None:
        pass

    def _read_angles(self) -> tuple[float, float] | None:
        return self._at

    def _move(self, raw1: float, raw2: float) -> None:
        turn = max(abs(_shortest(raw1 - self._at[0])), abs(_shortest(raw2 - self._at[1])))
        time.sleep(turn / self.speed_deg_s + self._conf.current.settle_s)
        self._at = (raw1 % 360.0, raw2 % 360.0)
