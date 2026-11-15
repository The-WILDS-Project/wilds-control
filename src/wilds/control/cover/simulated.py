"""A cover that only takes its time (5 s to open or close, made up until LDT says), so the
rest of WILDS can be exercised without hardware. Used in the ``development`` and
``ldt-simulated`` opmodes: nothing is ever sent to LDT."""

from typing import ClassVar

from .base import CoverBase, CoverState


class SimulatedCover(CoverBase):
    kind: ClassVar[str] = "simulated"
    start_state: ClassVar[CoverState] = "open"

    async def _send(self, close: bool) -> None:
        return None
