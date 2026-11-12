"""Real stars for the simulated guide camera (plan §3.3).

The simulator used to draw one invented star field whatever the telescope pointed at, which no
plate solver could match. This draws the real sky instead: the stars of a catalogue (Gaia DR3
through ``catalogs/catshtm.py``, a copy of MAST's reader), projected into the frame the way
the telescope, the rotator and the camera put them there.

**Geometry.** The telescope points at the target plus the pointing offsets, and the target sits on
the slit when the offsets are zero (the pointing origin is the slit pixel). A star ``(xi, eta)``
arcsec east and north of the pointing lands ``(xi E + eta N) / scale`` pixels from the slit pixel,
where ``E`` and ``N`` are the unit vectors of east and north in the image. The camera turns with
the rotator, so north is at the calibrated orientation (the angle at PA 0) minus parity times the
slit PA now: the same basis ``slit.js`` uses to draw the slit. Pixels are the unbinned
full-sensor pixels the simulated camera draws in.

**Catalogue.** A cone search is costly (files over SMB), so one is done around where the telescope
points with some margin and reused until the pointing leaves it. ``StarSource`` is anything that
answers a cone; ``CatsHtmSource`` is the real one, and tests use a few made-up stars.
"""

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

log = logging.getLogger(__name__)

# A star of this Gaia G magnitude gives ZERO_FLUX electrons per second in the simulator, the
# brightness of a typical star of the old invented field.
ZERO_MAG = 14.0
ZERO_FLUX = 3000.0


@dataclass
class Stars:
    ra_deg: np.ndarray
    dec_deg: np.ndarray
    mag: np.ndarray

    def __len__(self) -> int:
        return len(self.ra_deg)


class StarSource(Protocol):
    def cone(self, ra_deg: float, dec_deg: float, radius_deg: float) -> Stars: ...


class CatsHtmSource:
    """Gaia (or another catsHTM catalogue) as a :class:`StarSource`. ``h5py`` and ``scipy`` are
    imported when it is first read (the ``catalogs`` extra)."""

    def __init__(
        self,
        root: str,
        name: str = "GAIADR3",
        mag_column: str = "phot_g_mean_mag",
        mag_limit: float = 18.0,
        max_stars: int = 3000,
    ) -> None:
        from .catalogs.catshtm import CatsHtm

        self._cat = CatsHtm(root, name)
        self.mag_column, self.mag_limit, self.max_stars = mag_column, mag_limit, max_stars

    def cone(self, ra_deg: float, dec_deg: float, radius_deg: float) -> Stars:
        data = self._cat.cone_search(ra_deg, dec_deg, radius_deg, ["RA", "Dec", self.mag_column])
        ra, dec, mag = np.degrees(data["RA"]), np.degrees(data["Dec"]), data[self.mag_column]
        keep = np.isfinite(mag) & (mag <= self.mag_limit)
        ra, dec, mag = ra[keep], dec[keep], mag[keep]
        if len(mag) > self.max_stars:  # the brightest ones
            top = np.argsort(mag)[: self.max_stars]
            ra, dec, mag = ra[top], dec[top], mag[top]
        return Stars(ra, dec, mag)


class FieldFileSource:
    """A star field saved on local disk (:func:`save_field`): the data set the simulator uses, so a
    slow or absent catalogue share does not matter. Outside the saved cone it has no stars."""

    def __init__(self, path) -> None:
        with np.load(path) as f:
            self.centre = (float(f["centre_ra"]), float(f["centre_dec"]))
            self.radius_deg = float(f["radius_deg"])
            self._stars = Stars(f["ra_deg"], f["dec_deg"], f["mag"])

    def cone(self, ra_deg: float, dec_deg: float, radius_deg: float) -> Stars:
        s = self._stars
        cos_d = math.cos(math.radians(dec_deg))
        dra = (s.ra_deg - ra_deg + 180.0) % 360.0 - 180.0
        near = np.hypot(dra * cos_d, s.dec_deg - dec_deg) <= radius_deg
        return Stars(s.ra_deg[near], s.dec_deg[near], s.mag[near])


def save_field(path, centre_ra: float, centre_dec: float, radius_deg: float, stars: Stars) -> None:
    """Write the stars of a cone for :class:`FieldFileSource`."""
    np.savez_compressed(
        path,
        centre_ra=centre_ra,
        centre_dec=centre_dec,
        radius_deg=radius_deg,
        ra_deg=stars.ra_deg,
        dec_deg=stars.dec_deg,
        mag=stars.mag,
    )


def flux_from_mag(mag: np.ndarray) -> np.ndarray:
    """Electrons per second in the simulated camera for Gaia G magnitudes."""
    return ZERO_FLUX * 10.0 ** (-0.4 * (np.asarray(mag, dtype=float) - ZERO_MAG))


@dataclass
class Geometry:
    """How the sky maps onto the frame (the guide camera's calibration, and the rotator)."""

    plate_scale_arcsec_px: float
    slit_x: float
    slit_y: float
    orientation_deg: float  # angle from image up to sky north, counter-clockwise, at PA 0
    parity: int = 1
    pa_deg: float = 0.0  # the slit's sky PA now

    def basis(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Unit vectors of north and east in the image (x right, y down)."""
        a = -math.radians(self.orientation_deg - self.parity * self.pa_deg)  # clockwise
        north = (math.sin(a), -math.cos(a))
        east = (self.parity * -math.cos(a), self.parity * -math.sin(a))
        return north, east


def tangent_offsets(
    ra_deg: np.ndarray, dec_deg: np.ndarray, ra0_deg: float, dec0_deg: float
) -> tuple[np.ndarray, np.ndarray]:
    """Gnomonic projection: arcsec east (xi) and north (eta) of (ra0, dec0). Stars on the far
    side of the sky (``D <= 0``) come back as NaN."""
    a, d = np.radians(ra_deg), np.radians(dec_deg)
    a0, d0 = math.radians(ra0_deg), math.radians(dec0_deg)
    da = a - a0
    denom = math.sin(d0) * np.sin(d) + math.cos(d0) * np.cos(d) * np.cos(da)
    with np.errstate(divide="ignore", invalid="ignore"):
        xi = np.cos(d) * np.sin(da) / denom
        eta = (math.cos(d0) * np.sin(d) - math.sin(d0) * np.cos(d) * np.cos(da)) / denom
    xi, eta = np.where(denom > 0, xi, np.nan), np.where(denom > 0, eta, np.nan)
    return np.degrees(xi) * 3600.0, np.degrees(eta) * 3600.0


def pointing_after_offsets(
    ra_deg: float, dec_deg: float, east_arcsec: float, north_arcsec: float
) -> tuple[float, float]:
    """The sky position the telescope points at: the target moved by the tangent-plane offsets
    (small offsets, so a flat-sky step is exact enough)."""
    dec = dec_deg + north_arcsec / 3600.0
    ra = ra_deg + east_arcsec / 3600.0 / max(1e-6, math.cos(math.radians(dec_deg)))
    return ra % 360.0, max(-90.0, min(90.0, dec))


def project(
    stars: Stars, centre_ra: float, centre_dec: float, geo: Geometry
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pixel positions (full sensor, unbinned) and fluxes of ``stars`` for a telescope pointing at
    ``centre``; stars that cannot be in any frame are left in, the caller cuts to the frame."""
    xi, eta = tangent_offsets(stars.ra_deg, stars.dec_deg, centre_ra, centre_dec)
    north, east = geo.basis()
    x = geo.slit_x + (xi * east[0] + eta * north[0]) / geo.plate_scale_arcsec_px
    y = geo.slit_y + (xi * east[1] + eta * north[1]) / geo.plate_scale_arcsec_px
    ok = np.isfinite(x) & np.isfinite(y)
    return x[ok], y[ok], flux_from_mag(stars.mag[ok])


@dataclass
class Sky:
    """The stars around where the telescope points, ready to be drawn. One catalogue query covers a
    cone with margin; it is repeated only when the pointing leaves that margin."""

    source: StarSource
    frame_px: tuple[int, int]  # the sensor, unbinned
    margin_deg: float = 0.05
    _cone: tuple[float, float, float] | None = field(default=None, init=False, repr=False)
    _stars: Stars | None = field(default=None, init=False, repr=False)
    failed: str | None = field(default=None, init=False)  # why the catalogue could not be read
    retry_s: float = 30.0  # after a failure, do not ask again for this long (a dead share is slow)
    _retry_at: float = field(default=0.0, init=False, repr=False)

    def stars_for(
        self, centre_ra: float, centre_dec: float, geo: Geometry
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """``(x, y, flux)`` in the frame for a pointing, or ``None`` if there is no catalogue."""
        stars = self._load(centre_ra, centre_dec, geo)
        if stars is None:
            return None
        x, y, flux = project(stars, centre_ra, centre_dec, geo)
        w, h = self.frame_px
        near = (x > -50) & (x < w + 50) & (y > -50) & (y < h + 50)
        return x[near], y[near], flux[near]

    def _radius_deg(self, geo: Geometry) -> float:
        w, h = self.frame_px  # the frame's half-diagonal seen from the slit pixel, plus the margin
        reach = max(math.hypot(x - geo.slit_x, y - geo.slit_y) for x in (0, w) for y in (0, h))
        return reach * geo.plate_scale_arcsec_px / 3600.0 + self.margin_deg

    def _load(self, ra: float, dec: float, geo: Geometry) -> Stars | None:
        if self._cone is not None:
            c_ra, c_dec, c_radius = self._cone
            moved = math.hypot((ra - c_ra) * math.cos(math.radians(dec)), dec - c_dec)
            if (
                moved + self._radius_deg(geo) - self.margin_deg <= c_radius
                and self._stars is not None
            ):
                return self._stars
        if time.monotonic() < self._retry_at:
            return None
        radius = self._radius_deg(geo) + self.margin_deg
        try:
            self._stars = self.source.cone(ra, dec, radius)
        except Exception as e:  # noqa: BLE001 - an unreadable catalogue must not stop the camera
            if self.failed != str(e):
                log.warning("star catalogue not readable (%s); using the invented field", e)
            self.failed = str(e)
            self._cone, self._stars = None, None
            self._retry_at = time.monotonic() + self.retry_s
            return None
        self.failed = None
        self._cone = (ra, dec, radius)
        log.info(
            "catalogue cone at (%.4f, %.4f) r=%.3f deg: %d stars", ra, dec, radius, len(self._stars)
        )
        return self._stars
