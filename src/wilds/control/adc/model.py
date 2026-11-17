"""The atmosphere's dispersion, and the angles of the two prism pairs that cancel it.

The ADC is two pairs of Amici prisms in the telescope's converging beam (WILDS optics report
§4.1). Each pair makes a dispersion of the same size and turns on its own stage; the pairs
counter-rotate, so the dispersion they deliver together is

    D(phi) = Dmax * sin(phi)          (optics report, Appendix A.5)

in a direction set by where the pairs sit together. ``phi`` is half the angle between them: with
the direction ``theta`` of the dispersion to make, the pairs sit at ``theta + phi`` and
``theta - phi``. The atmosphere's dispersion between the band edges (305 and 915 nm) grows
roughly as tan(zenith distance); ``phi = asin(D_atm / Dmax)`` matches it. The report's design
puts ZD 60 degrees at the maximum, ``phi`` = 90 degrees (at sea-level pressure; at the LDT's
altitude the atmosphere disperses about 1.3 times less).

Which way the prism pairs have to point to make a dispersion opposite to the atmosphere's, and
where each stage reads zero, are hardware facts WILDS does not know yet: the direction is
``parallactic angle - slit PA`` plus a calibrated offset, and each stage has a calibrated zero
point (``AdcCal``). Everything here is plain functions of numbers, with no hardware and no I/O.
"""

import math
from dataclasses import dataclass

from ..weather import Weather

SHORT_NM = 305.0
LONG_NM = 915.0


def refractivity(wavelength_nm: float, weather: Weather) -> float:
    """``n - 1`` of air at this wavelength and these conditions (Edlén 1966, with the water-vapour
    term): the dispersion of the standard air, scaled for pressure, temperature and humidity."""
    sigma2 = (1000.0 / wavelength_nm) ** 2  # 1/lambda^2, lambda in micrometres
    n_std = (
        8342.13 + 2406030.0 / (130.0 - sigma2) + 15997.0 / (38.9 - sigma2)
    ) * 1e-8  # dry air at 15 C and 760 Torr
    p = weather.pressure_mbar * 0.750062  # Torr
    t = weather.temperature_c
    n_tp = n_std * p * (1.0 + (1.049 - 0.0157 * t) * 1e-6 * p) / (720.883 * (1.0 + 0.003661 * t))
    # water vapour pressure (Torr) from the relative humidity, Magnus formula
    es = 6.1094 * math.exp(17.625 * t / (t + 243.04)) * 0.750062
    f = weather.humidity_pct / 100.0 * es
    return n_tp - f * (5.722 - 0.0457 * sigma2) * 1e-8


def refraction_arcsec(zd_deg: float, n_minus_1: float, temperature_c: float) -> float:
    """The refraction at this zenith distance, to the second order in tan(zd):
    ``R = A tan z + B tan^3 z``, A = (n-1)(1-beta), B = -(n-1)(beta - (n-1)/2)."""
    beta = 0.001254 * (273.15 + temperature_c) / 273.15
    z = math.radians(zd_deg)
    r = (
        n_minus_1 * (1.0 - beta) * math.tan(z)
        - n_minus_1 * (beta - n_minus_1 / 2.0) * math.tan(z) ** 3
    )
    return math.degrees(r) * 3600.0


def atmospheric_dispersion_arcsec(
    zd_deg: float, weather: Weather, short_nm: float = SHORT_NM, long_nm: float = LONG_NM
) -> float:
    """How far apart the images of the two band edges fall at this zenith distance (arcsec)."""
    zd = min(max(zd_deg, 0.0), 89.0)
    t = weather.temperature_c
    return refraction_arcsec(zd, refractivity(short_nm, weather), t) - refraction_arcsec(
        zd, refractivity(long_nm, weather), t
    )


@dataclass(frozen=True)
class Pairs:
    """What the two pairs are to do: the dispersion they deliver, its direction, and where each
    pair sits (degrees in the instrument frame, 0 to 360)."""

    dispersion_arcsec: float
    direction_deg: float
    pair1_deg: float
    pair2_deg: float
    saturated: bool  # the dispersion asked for is more than the ADC can make (it makes its most)


def pairs_for(dispersion_arcsec: float, direction_deg: float, dmax_arcsec: float) -> Pairs:
    """The positions of the two pairs for a dispersion of this size and direction."""
    wanted = max(dispersion_arcsec, 0.0)
    saturated = wanted > dmax_arcsec
    phi = math.degrees(math.asin(min(1.0, wanted / dmax_arcsec)))
    given = dmax_arcsec * math.sin(math.radians(phi))
    return Pairs(
        dispersion_arcsec=given,
        direction_deg=direction_deg % 360.0,
        pair1_deg=(direction_deg + phi) % 360.0,
        pair2_deg=(direction_deg - phi) % 360.0,
        saturated=saturated,
    )


def delivered(pair1_deg: float, pair2_deg: float, dmax_arcsec: float) -> tuple[float, float]:
    """The dispersion (arcsec) and its direction (degrees) that two pairs at these positions
    deliver: the inverse of :func:`pairs_for`."""
    half = math.radians(_wrap(pair1_deg - pair2_deg) / 2.0)
    mid = pair2_deg + _wrap(pair1_deg - pair2_deg) / 2.0
    d = dmax_arcsec * math.sin(half)
    if d < 0:  # pair 1 is behind pair 2: the same dispersion the other way round
        return -d, (mid + 180.0) % 360.0
    return d, mid % 360.0


def _wrap(deg: float) -> float:
    """An angle in (-180, 180]."""
    d = (deg + 180.0) % 360.0 - 180.0
    return 180.0 if d == -180.0 else d
