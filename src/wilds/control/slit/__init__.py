"""The slit stage: an abstract base, a simulator, and the one instance the app uses."""

from ..conf import confs
from ..lifecycle import system
from .base import (
    COMPONENT,
    Preset,
    SlitActivity,
    SlitBase,
    SlitBusy,
    SlitConf,
    SlitRange,
    SlitStatus,
)
from .simulated import SimulatedSlit

__all__ = [
    "COMPONENT",
    "Preset",
    "SimulatedSlit",
    "SlitActivity",
    "SlitBase",
    "SlitBusy",
    "SlitConf",
    "SlitRange",
    "SlitStatus",
    "close_slit",
    "get_slit",
    "reset",
]

_slit: SlitBase | None = None


def get_slit() -> SlitBase:
    """The one slit stage, made on first use (the app does so at startup). Use it as a FastAPI
    ``Depends``; tests override it through ``app.dependency_overrides``. Only the simulator
    exists until the hardware is chosen."""
    global _slit
    if _slit is None:
        slit = SimulatedSlit()
        confs.add(slit.conf)  # for the /config endpoints
        system.register(slit.life)  # started up by the system (or by hand), not here
        _slit = slit
    return _slit


async def close_slit() -> None:
    global _slit
    slit, _slit = _slit, None
    if slit is not None:
        await slit.close_hardware()


def reset() -> None:
    """Forget the instance without closing it (for tests)."""
    global _slit
    _slit = None
