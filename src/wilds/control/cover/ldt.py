"""The LDT instrument cover, moved by XML ``<stage-command>`` messages to the instrument cube's
stage controller (``joeStage``) through the broker. Only the ``ldt-production`` opmode builds
this class (see :func:`wilds.control.cover.get_cover`).

The cube reports the cover (OPEN or CLOSED) and the fold mirrors on ``instrumentCube.loisTelemetry``
about once a second. At start-up the state is ``unknown`` until the first report; then the
reports are the truth (see :meth:`CoverBase.feed`). The cover takes 5 to 10 s to move (LDT, Nov
2026): ``travel_s`` is the long end, the time after which a move that has not been seen is judged.
"""

import logging
from collections.abc import Awaitable, Callable
from typing import ClassVar

from pydantic import Field

from .base import CoverBase, CubeReport

log = logging.getLogger(__name__)

# The axis AZ, function IC; HM closes the cover and HR opens it (LDT, Nov 2026).
_XML = (
    "<stage-command>\n"
    "    <axis>\n"
    "        <name>AZ</name>\n"
    "        <function>IC</function>\n"
    "    </axis>\n"
    "    <command>AZ; {code};</command>\n"
    "</stage-command>"
)


def stage_command_xml(*, close: bool) -> str:
    return _XML.format(code="HM" if close else "HR")


def cube_report(telemetry: object) -> CubeReport:
    """A broker ``InstrumentCubeTelemetry`` as a :class:`CubeReport`."""
    return CubeReport(
        cover_state=getattr(telemetry, "InstrumentCoverState", None),
        cover_position_mm=getattr(telemetry, "InstrumentCoverPosition", None),
        fm_c_state=getattr(telemetry, "FMCState", None),
        fm_c_position_mm=getattr(telemetry, "FMCPosition", None),
    )


class LdtCover(CoverBase):
    kind: ClassVar[str] = "ldt"
    travel_s: float = 10.0

    # How the XML is sent: the broker's ``send_stage_command``. Injected, so tests (and the
    # app) decide the connection.
    send: Callable[[str], Awaitable[None]] = Field(exclude=True)
    # Start listening: connect to the broker, whose cube telemetry goes to ``feed``.
    listen: Callable[[], Awaitable[None]] | None = Field(default=None, exclude=True)
    stop: Callable[[], Awaitable[None]] | None = Field(default=None, exclude=True)

    async def _start_feed(self) -> None:
        if self.listen is None:
            return
        try:
            await self.listen()
        except Exception as e:  # the commands still work, and connect again when sent
            log.warning("cover: not listening to the instrument cube yet: %s", e)

    async def _send(self, close: bool) -> None:
        await self.send(stage_command_xml(close=close))

    async def _close_hardware(self) -> None:
        if self.stop is not None:
            await self.stop()
