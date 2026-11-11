"""The telescope: an abstract base, a simulator, and the one instance the app uses."""

from ..conf import confs
from ..lifecycle import system
from .base import (
    COMPONENT,
    Offset,
    OffsetNum,
    OffsetType,
    Pending,
    RotatorFrame,
    Target,
    Telemetry,
    TelescopeActivity,
    TelescopeBase,
    TelescopeConf,
    TelescopeError,
    TelescopeStatus,
)
from .simulated import SimulatedTelescope

__all__ = [
    "COMPONENT",
    "Offset",
    "OffsetNum",
    "OffsetType",
    "Pending",
    "RotatorFrame",
    "SimulatedTelescope",
    "Target",
    "TelescopeActivity",
    "TelescopeBase",
    "TelescopeConf",
    "TelescopeError",
    "TelescopeStatus",
    "Telemetry",
    "close_telescope",
    "get_telescope",
    "reset",
]

_telescope: TelescopeBase | None = None


def get_telescope() -> TelescopeBase:
    """The one telescope, made on first use (the app does so at startup). Use it as a FastAPI
    ``Depends``; tests override it through ``app.dependency_overrides``. Only the simulator
    exists until the bridge is wired in."""
    global _telescope
    if _telescope is None:
        telescope = SimulatedTelescope()
        confs.add(telescope.conf)  # for the /config endpoints
        system.register(telescope.life)  # started up by the system (or by hand), not here
        _telescope = telescope
    return _telescope


async def close_telescope() -> None:
    global _telescope
    telescope, _telescope = _telescope, None
    if telescope is not None:
        await telescope.close_hardware()


def reset() -> None:
    """Forget the instance without closing it (for tests)."""
    global _telescope
    _telescope = None
