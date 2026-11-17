"""The ADC: an abstract base, a simulator, the prism-pair model, and the one instance the app uses."""

from ..conf import confs
from ..lifecycle import system
from . import model
from .base import (
    COMPONENT,
    AdcActivity,
    AdcBase,
    AdcBusy,
    AdcCal,
    AdcConf,
    AdcRange,
    AdcStatus,
    AdcTarget,
    SkyState,
)
from .simulated import SimulatedAdc

__all__ = [
    "COMPONENT",
    "AdcActivity",
    "AdcBase",
    "AdcBusy",
    "AdcCal",
    "AdcConf",
    "AdcRange",
    "AdcStatus",
    "AdcTarget",
    "SimulatedAdc",
    "SkyState",
    "close_adc",
    "get_adc",
    "model",
    "reset",
]

_adc: AdcBase | None = None


def telescope_sky() -> SkyState | None:
    """The telescope's target as the ADC needs it, or ``None`` while it has none."""
    from ..telescope import get_telescope

    t = get_telescope().telemetry()
    if t.zenith_distance_deg is None or t.parallactic_angle_deg is None:
        return None
    return SkyState(t.zenith_distance_deg, t.parallactic_angle_deg, t.rot_pa or 0.0)


def get_adc() -> AdcBase:
    """The one ADC, made on first use (the app does so at startup). Use it as a FastAPI
    ``Depends``; tests override it through ``app.dependency_overrides``. Only the simulator
    exists until the MCS2 is wired."""
    global _adc
    if _adc is None:
        adc = SimulatedAdc(sky=telescope_sky)  # the weather is the simulator's, see weather.py
        confs.add(adc.conf)  # for the /config endpoints
        confs.add(adc.cal)
        system.register(adc.life)  # started up by the system (or by hand), not here
        _adc = adc
    return _adc


async def close_adc() -> None:
    global _adc
    adc, _adc = _adc, None
    if adc is not None:
        await adc.close_hardware()


def reset() -> None:
    """Forget the instance without closing it (for tests)."""
    global _adc
    _adc = None
