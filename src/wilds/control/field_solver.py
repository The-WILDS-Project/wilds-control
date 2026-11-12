"""Plate solving a guide-camera frame (plan §3.3): where the sky lies in the image.

``FieldSolver`` is what the optics calibration needs from a solver: the frame's pixels and a rough
idea (where it points, what the plate scale is) in; the plate scale, the angle of north and the
parity out. :class:`AstrometrySolver` is the real one (``wilds-solving``: SEP source extraction
and astrometry.net index files); :class:`SimulatedSolver` answers from the simulated camera's own
hidden optics (see ``guide_camera/optics.py``) for a machine with no index files, so the
calibration can be exercised end to end. Nothing here knows about FastAPI.

Everything is in the pixels of the frame given: a binned frame has a coarser plate scale, and the
caller converts. Angles are counter-clockwise on the screen from the top of the image.
"""

import asyncio
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from . import config

log = logging.getLogger(__name__)

MAX_STARS = 50  # the brightest this many go to the solver: astrometry.net is fast with few


class SolveFailed(RuntimeError):
    """The frame could not be solved (no match, no stars, or no index files)."""


@dataclass(frozen=True)
class FieldSolution:
    scale_arcsec_px: float  # of the frame given (binned pixels if it is binned)
    north_deg: float  # angle of sky north counter-clockwise from the top, (-180, 180]
    parity: int  # +1 east is counter-clockwise from north on screen, -1 mirrored
    ra_deg: float  # the frame's centre
    dec_deg: float
    logodds: float
    confidence: str
    stars: int  # how many were used


class FieldSolver(Protocol):
    async def solve(
        self,
        pixels: np.ndarray,
        *,
        ra_deg: float,
        dec_deg: float,
        scale_arcsec_px: tuple[float, float],
    ) -> FieldSolution:
        """Solve a frame given roughly where it points and the range its scale is in (in the
        frame's own pixels). Raises :class:`SolveFailed`."""
        ...


class AstrometrySolver:
    """``wilds-solving`` on astrometry.net index files. The index files are loaded on first use
    and kept: loading takes a while, a solve does not."""

    def __init__(self, index_dir: Path | None = None, scales: list[int] | None = None) -> None:
        self.index_dir = Path(
            index_dir or os.environ.get("WILDS_ASTROMETRY_CACHE") or "~/.astrometry"
        ).expanduser()
        self.scales = set(scales if scales is not None else [0, 1, 2])
        self._solver = None

    def _open(self):
        if self._solver is None:
            from wilds.solving import PlateSolver

            if not any((self.index_dir / "5200").glob("*.fits")):
                raise SolveFailed(
                    f"no astrometry.net index files in {self.index_dir / '5200'}: "
                    "put the series 5200 files there (see [solver] in config.toml)"
                )
            self._solver = PlateSolver(self.index_dir, self.scales, download=False).__enter__()
        return self._solver

    def _solve(self, pixels, ra_deg, dec_deg, scale) -> FieldSolution:
        from wilds.solving import PositionHint, ScaleHint, extract_sources

        solver = self._open()
        sources = extract_sources(pixels.astype(np.float64), threshold=3.0)
        if len(sources) < 8:
            raise SolveFailed(f"only {len(sources)} stars found in the frame")
        if len(sources) > MAX_STARS:
            sources = sources[np.argsort(sources["flux"])[::-1][:MAX_STARS]]
        h, w = pixels.shape
        result = solver.solve(
            sources,
            image_width=w,
            image_height=h,
            scale_hint=ScaleHint(*scale),
            position_hint=PositionHint(ra_deg=ra_deg, dec_deg=dec_deg, radius_deg=1.0),
        )
        if result is None:
            raise SolveFailed(
                f"no match for {len(sources)} stars near RA {ra_deg:.3f} Dec {dec_deg:+.3f}"
            )
        return FieldSolution(
            scale_arcsec_px=result.pixel_scale_arcsec,
            north_deg=result.north_deg,
            parity=result.parity,
            ra_deg=result.ra_deg,
            dec_deg=result.dec_deg,
            logodds=result.logodds,
            confidence=result.confidence,
            stars=len(sources),
        )

    async def solve(self, pixels, *, ra_deg, dec_deg, scale_arcsec_px) -> FieldSolution:
        return await asyncio.to_thread(self._solve, pixels, ra_deg, dec_deg, scale_arcsec_px)


class SimulatedSolver:
    """What a perfect solver would say about the simulated camera, with a little noise: its true
    optics (plate scale, orientation, parity) and the rotator's angle now. ``sensor_width`` is the
    unbinned width the frame is a binning of."""

    def __init__(
        self,
        optics: Callable,
        pa_deg: Callable[[], float],
        centre: Callable[[], tuple[float, float]],
        sensor_width: int,
    ) -> None:
        self._optics, self._pa, self._centre, self._width = optics, pa_deg, centre, sensor_width
        self._rng = np.random.default_rng()

    async def solve(self, pixels, *, ra_deg, dec_deg, scale_arcsec_px) -> FieldSolution:
        o = self._optics()
        binning = max(1, round(self._width / pixels.shape[1]))
        north = (o.orientation_deg - o.parity * self._pa() + 180.0) % 360.0 - 180.0
        ra, dec = self._centre()
        await asyncio.sleep(0.05)
        return FieldSolution(
            scale_arcsec_px=o.plate_scale_arcsec_px
            * binning
            * (1 + 5e-4 * self._rng.standard_normal()),
            north_deg=north + 0.02 * float(self._rng.standard_normal()),
            parity=o.parity,
            ra_deg=ra,
            dec_deg=dec,
            logodds=100.0,
            confidence="high",
            stars=MAX_STARS,
        )


_solver: FieldSolver | None = None


def get_field_solver() -> FieldSolver:
    """The one solver, made on first use from ``[solver]`` in config.toml. Use it as a FastAPI
    ``Depends``; tests override it through ``app.dependency_overrides``."""
    global _solver
    if _solver is None:
        cfg = config.get().solver
        if cfg.kind == "simulated":
            from .guide_camera import SimulatedASICamera, get_guide_camera
            from .guide_camera.simulated import SENSOR
            from .telescope import get_telescope

            camera = get_guide_camera()
            if not isinstance(camera, SimulatedASICamera):
                raise RuntimeError("[solver] kind = simulated needs the simulated guide camera")

            def centre() -> tuple[float, float]:
                t = get_telescope().telemetry().target
                return (t.ra_hours * 15.0, t.dec_deg) if t else (0.0, 0.0)

            _solver = SimulatedSolver(
                lambda: camera.optics,
                lambda: get_telescope().telemetry().rot_pa or 0.0,
                centre,
                SENSOR[0],
            )
        else:
            _solver = AstrometrySolver(cfg.index_dir, cfg.scales)
    return _solver


def reset() -> None:
    """Forget the instance (for tests)."""
    global _solver
    _solver = None
