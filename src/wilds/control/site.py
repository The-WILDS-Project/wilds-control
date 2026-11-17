"""The Lowell Discovery Telescope's site, and where a star is in its sky.

The simulated telescope has no mount: it works out the zenith distance and the parallactic angle
of its target from the site and the clock, for the ADC. The coordinates are treated as of date,
which is off by the precession since J2000 (a fraction of a degree): good for a simulator, not for
pointing. The real TCS reports both.
"""

import math
from dataclasses import dataclass

LATITUDE_DEG = 34.7443  # 34 44 40 N
LONGITUDE_DEG = -111.4222  # 111 25 20 W
ALTITUDE_M = 2360.0


@dataclass(frozen=True)
class Horizontal:
    zenith_distance_deg: float
    azimuth_deg: float  # from north, through east
    parallactic_angle_deg: float  # at the target, from north to the zenith, positive to the west


def local_sidereal_deg(unix_s: float, longitude_deg: float = LONGITUDE_DEG) -> float:
    """The local mean sidereal time in degrees at this Unix time (UT1 taken as UTC)."""
    jd = unix_s / 86400.0 + 2440587.5
    gmst = 280.46061837 + 360.98564736629 * (jd - 2451545.0)
    return (gmst + longitude_deg) % 360.0


def horizontal(
    ra_hours: float, dec_deg: float, unix_s: float, latitude_deg: float = LATITUDE_DEG
) -> Horizontal:
    """Where a target is in the sky of the site at this time."""
    ha = math.radians(local_sidereal_deg(unix_s) - ra_hours * 15.0)
    dec, lat = math.radians(dec_deg), math.radians(latitude_deg)
    sin_alt = math.sin(lat) * math.sin(dec) + math.cos(lat) * math.cos(dec) * math.cos(ha)
    alt = math.asin(max(-1.0, min(1.0, sin_alt)))
    az = math.atan2(
        -math.cos(dec) * math.sin(ha),
        math.sin(dec) * math.cos(lat) - math.cos(dec) * math.cos(ha) * math.sin(lat),
    )
    q = math.atan2(math.sin(ha), math.tan(lat) * math.cos(dec) - math.sin(dec) * math.cos(ha))
    return Horizontal(
        zenith_distance_deg=90.0 - math.degrees(alt),
        azimuth_deg=math.degrees(az) % 360.0,
        parallactic_angle_deg=math.degrees(q),
    )
