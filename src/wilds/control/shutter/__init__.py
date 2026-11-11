"""The shutter: an abstract base, a simulator, and the one instance the app uses."""

from ..conf import confs
from ..lifecycle import system
from .base import (
    COMPONENT,
    ShutterActivity,
    ShutterBase,
    ShutterBusy,
    ShutterConf,
    ShutterState,
    ShutterStatus,
)
from .simulated import SimulatedShutter

__all__ = [
    "COMPONENT",
    "ShutterActivity",
    "ShutterBase",
    "ShutterBusy",
    "ShutterConf",
    "ShutterState",
    "ShutterStatus",
    "SimulatedShutter",
    "close_shutter",
    "get_shutter",
    "reset",
]

_shutter: ShutterBase | None = None


def get_shutter() -> ShutterBase:
    """The one shutter, made on first use (the app does so at startup). Use it as a FastAPI
    ``Depends``; tests override it through ``app.dependency_overrides``. Only the simulator
    exists until the hardware is chosen."""
    global _shutter
    if _shutter is None:
        shutter = SimulatedShutter()
        confs.add(shutter.conf)  # for the /config endpoints
        system.register(shutter.life)  # started up by the system (or by hand), not here
        _shutter = shutter
    return _shutter


async def close_shutter() -> None:
    global _shutter
    shutter, _shutter = _shutter, None
    if shutter is not None:
        await shutter.close_hardware()


def reset() -> None:
    """Forget the instance without closing it (for tests)."""
    global _shutter
    _shutter = None
