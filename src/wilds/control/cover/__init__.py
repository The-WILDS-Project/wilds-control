"""The instrument cover: an abstract base, a simulator, the LDT one, and the one instance the
app uses."""

import logging

from .. import config
from ..lifecycle import system
from .base import (
    COMPONENT,
    CoverActivity,
    CoverBase,
    CoverBusy,
    CoverError,
    CoverState,
    CoverStatus,
    CubeReport,
    CubeStatus,
)
from .ldt import LdtCover, cube_report, stage_command_xml
from .simulated import SimulatedCover

__all__ = [
    "COMPONENT",
    "CoverActivity",
    "CoverBase",
    "CoverBusy",
    "CoverError",
    "CoverState",
    "CoverStatus",
    "CubeReport",
    "CubeStatus",
    "LdtCover",
    "SimulatedCover",
    "close_cover",
    "cube_report",
    "get_cover",
    "reset",
    "stage_command_xml",
]

log = logging.getLogger(__name__)

_cover: CoverBase | None = None


def _ldt_cover() -> LdtCover:
    """The real cover: a broker connection made at start-up (to listen to the instrument cube),
    or on the first command, to ``[broker] host``."""
    from wilds.bridge.broker import LDTBroker
    from wilds.bridge.config import TOPIC_INSTRUMENT_CUBE, BrokerConfig

    broker: LDTBroker | None = None

    async def connect() -> LDTBroker:
        nonlocal broker
        if broker is None:
            made = LDTBroker(BrokerConfig(host=config.get().broker.host or ""))

            async def on_cube(telemetry: object) -> None:
                cover.feed(cube_report(telemetry))

            made.add_callback(TOPIC_INSTRUMENT_CUBE, on_cube)
            await made.start()
            broker = made
        return broker

    async def send(xml: str) -> None:
        await (await connect()).send_stage_command(xml)

    async def listen() -> None:
        await connect()

    async def stop() -> None:
        nonlocal broker
        if broker is not None:
            await broker.stop()
            broker = None

    cover = LdtCover(send=send, listen=listen, stop=stop)
    return cover


def get_cover() -> CoverBase:
    """The one cover, made on first use (the app does so at startup). Only ``ldt-production``
    sends anything to LDT; every other opmode gets the simulator. Use it as a FastAPI
    ``Depends``; tests override it through ``app.dependency_overrides``."""
    global _cover
    if _cover is None:
        cover = _ldt_cover() if config.get().opmode == "ldt-production" else SimulatedCover()
        system.register(cover.life)  # started up by the system (or by hand), not here
        _cover = cover
    return _cover


async def close_cover() -> None:
    global _cover
    cover, _cover = _cover, None
    if cover is not None:
        await cover.close_hardware()


def reset() -> None:
    """Forget the instance without closing it (for tests)."""
    global _cover
    _cover = None
