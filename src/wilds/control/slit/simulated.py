"""A slit stage that only takes its time, so the rest of WILDS can be exercised without hardware.
Its travel is the SmarAct CLS-3232's, 21 mm (SmarAct data sheet: >20 mm/s, 1 nm closed-loop
resolution with the MCS2); the speed is made up: 5 mm/s."""

import time
from typing import ClassVar

from pydantic import PrivateAttr

from .base import SlitBase


class SimulatedSlit(SlitBase):
    kind: ClassVar[str] = "simulated"

    min_um: float = 0.0
    max_um: float = 21000.0
    speed_um_s: float = 5000.0
    start_um: float = 2.0  # where the stage is when WILDS starts (closed)

    _at: float = PrivateAttr(default=0.0)

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        self._at = self.start_um

    def _travel(self) -> tuple[float, float]:
        return self.min_um, self.max_um

    def _read_position(self) -> float | None:
        return self._at

    def _move(self, um: float) -> None:
        time.sleep(abs(um - self._at) / self.speed_um_s + self._conf.current.settle_s)
        self._at = um
