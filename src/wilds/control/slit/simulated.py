"""A slit stage that only takes its time, so the rest of WILDS can be exercised without hardware.
Its range is the slit's, fixed by hard stops: 0.5 to 4 arcsec (WILDS PDR), 62.345 to 498.76 µm at
124.69 µm/arcsec. The slit cannot close completely: its narrowest gap is 0.5 arcsec. The speed is
made up: 5 mm/s (the SmarAct CLS-3232 does more than 20 mm/s)."""

import time
from typing import ClassVar

from pydantic import PrivateAttr

from .base import SlitBase


class SimulatedSlit(SlitBase):
    kind: ClassVar[str] = "simulated"

    min_um: float = 62.345  # the lower hard stop: 0.5 arcsec
    max_um: float = 498.76  # the upper hard stop: 4 arcsec
    speed_um_s: float = 5000.0
    start_um: float = 62.345  # where the stage is when WILDS starts (closed: the narrowest)

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
