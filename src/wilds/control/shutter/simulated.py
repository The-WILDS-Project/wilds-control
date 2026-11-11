"""A shutter that only takes its time, so the rest of WILDS can be exercised without hardware."""

import time
from typing import ClassVar, Literal

from pydantic import PrivateAttr

from .base import ShutterBase


class SimulatedShutter(ShutterBase):
    kind: ClassVar[str] = "simulated"

    travel_s: float = 0.15  # how long the blade takes
    start_open: bool = False  # where the blade is when WILDS starts

    _open_now: bool = PrivateAttr(default=False)

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        self._open_now = self.start_open

    def _read_state(self) -> Literal["open", "closed", "unknown"]:
        return "open" if self._open_now else "closed"

    def _move(self, open: bool) -> None:
        time.sleep(self.travel_s + self._conf.current.settle_s)
        self._open_now = open
