"""A cover that behaves like the real thing without hardware, used in the ``development`` and
``ldt-simulated`` opmodes: nothing is ever sent to LDT.

It simulates the instrument cube too. A small model of the hardware (the cover takes ``travel_s``,
5 s, to move; the fold mirror that feeds WILDS is extended) publishes what the cube would, as
:class:`CubeReport`, once a second while the cover is started up. The cover then follows those
reports exactly as it follows the real telemetry: the state it shows is what the "sensor"
says. As on the real cube (as far as we know), the reported state is OPEN or CLOSED only, and it
changes when the cover has arrived. The strokes in mm are invented.
"""

import asyncio
import logging
import time
from typing import ClassVar

from pydantic import PrivateAttr

from .base import CoverBase, CoverState, CubeReport

log = logging.getLogger(__name__)

STROKE_MM = 250.0  # invented: the cover's position is 0 open, this closed
FM_EXTENDED_MM = 100.0  # invented


class SimulatedCover(CoverBase):
    kind: ClassVar[str] = "simulated"
    start_state: ClassVar[CoverState] = "open"

    telemetry_period_s: float = 1.0  # as often as the real cube reports

    _hw_closed: bool = PrivateAttr(default=False)  # where the simulated cover is, or is going
    _hw_from_mm: float = PrivateAttr(default=0.0)
    _hw_t0: float = PrivateAttr(default=0.0)
    _feed_task: asyncio.Task | None = PrivateAttr(default=None)

    # -- the simulated hardware ---------------------------------------------------------------

    def _hw_done(self) -> float:
        """How far through its move the cover is, 0 to 1."""
        return min(1.0, (time.monotonic() - self._hw_t0) / max(self.travel_s, 1e-6))

    def _hw_position(self) -> float:
        goal = STROKE_MM if self._hw_closed else 0.0
        return self._hw_from_mm + (goal - self._hw_from_mm) * self._hw_done()

    def _hw_report(self) -> CubeReport:
        """What the cube would say now: OPEN or CLOSED once the cover has arrived, and until then
        what it was before the move; the position follows the move."""
        position = self._hw_position()
        closed = self._hw_closed if self._hw_done() >= 1.0 else not self._hw_closed
        return CubeReport(
            cover_state="CLOSED" if closed else "OPEN",
            cover_position_mm=round(position, 1),
            fm_c_state="EXTENDED",
            fm_c_position_mm=FM_EXTENDED_MM,
        )

    async def _send(self, close: bool) -> None:
        self._hw_from_mm = self._hw_position()
        self._hw_closed, self._hw_t0 = close, time.monotonic()

    # -- the simulated reports ----------------------------------------------------------------

    async def _start_feed(self) -> None:
        if self._feed_task is None:
            self._feed_task = asyncio.create_task(self._publish(), name="cover-cube-telemetry")

    async def _stop_feed(self) -> None:
        task, self._feed_task = self._feed_task, None
        if task is not None and task is not asyncio.current_task():
            task.cancel()

    async def _publish(self) -> None:
        while True:
            self.feed(self._hw_report())
            await asyncio.sleep(self.telemetry_period_s)
