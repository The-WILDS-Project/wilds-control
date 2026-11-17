"""The weather the ADC needs (pressure, temperature, humidity), as WRS gives it.

The real source is the weather station's broker topic (``WRS.WRSPubDataSV.WRSDataPacket``,
metric units); until the bridge is wired in, a simulator gives the LDT's site-typical conditions
at 2360 m. Nothing here talks to the network.
"""

from pydantic import BaseModel


class Weather(BaseModel):
    pressure_mbar: float
    temperature_c: float
    humidity_pct: float


# Typical for the LDT (Happy Jack, Arizona, 2360 m): about 770 mbar, mild, dry.
SITE_TYPICAL = Weather(pressure_mbar=770.0, temperature_c=5.0, humidity_pct=30.0)

_weather: Weather = SITE_TYPICAL


def get_weather() -> Weather:
    """The weather now. The simulator's is fixed; tests set it with :func:`set_weather`."""
    return _weather


def set_weather(weather: Weather) -> None:
    global _weather
    _weather = weather


def reset() -> None:
    set_weather(SITE_TYPICAL)
