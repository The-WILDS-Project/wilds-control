"""The LDT instrument cover, moved by XML ``<stage-command>`` messages to the instrument cube's
stage controller (``joeStage``) through the broker. Only the ``ldt-production`` opmode builds
this class (see :func:`wilds.control.cover.get_cover`).

Nothing is known to sense the cover, so its state is what was last commanded, after the assumed
travel time; at start-up it is ``unknown``. Should LDT say otherwise, ``_sense`` is where it goes.
"""

from collections.abc import Awaitable, Callable
from typing import ClassVar

from pydantic import Field

from .base import CoverBase

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


class LdtCover(CoverBase):
    kind: ClassVar[str] = "ldt"

    # How the XML is sent: the broker's ``send_stage_command``. Injected, so tests (and the
    # app) decide the connection.
    send: Callable[[str], Awaitable[None]] = Field(exclude=True)
    stop: Callable[[], Awaitable[None]] | None = Field(default=None, exclude=True)

    async def _send(self, close: bool) -> None:
        await self.send(stage_command_xml(close=close))

    async def _close_hardware(self) -> None:
        if self.stop is not None:
            await self.stop()
