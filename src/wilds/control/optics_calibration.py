"""Calibrating the guide camera and the slit from the sky (plan §3.3).

One procedure measures what the live view's slit overlay and the acquisition need to know:

1. Close the slit and take a frame; **plate solve** it: the plate scale, the angle of sky north in
   the image, and whether the image is mirrored (parity).
2. Open the slit to a measuring width (a strip a few arcsec wide: a fully open slit is wider than
   it is long) and take a second frame of the same sky.
3. **Find the gap** in the difference of the two (``slit_detect``): where the slit is in the image,
   the direction it points, and its length and width in pixels.
4. Put them together: with the plate scale the slit's length and width are in arcsec, and the slit's
   direction against north at the rotator's angle says how the rotator's zero sits.

The result is only a measurement: ``Measurement`` says what was found and what looks wrong
(``warnings``); applying it to the calibration is a separate, deliberate step (``cal_values``).

Everything here is in unbinned full-sensor pixels unless it says otherwise, as the calibration is.
Nothing here knows about FastAPI: the router brings the camera, the stage, the telescope, the
solver, and ``expose`` (one exposure that a cancelled job can cut short).
"""

import logging
import time
from collections.abc import Awaitable, Callable

import numpy as np
from pydantic import BaseModel, Field

from .field_solver import FieldSolution, FieldSolver, SolveFailed
from .guide_camera.base import CaptureSettings, Frame
from .slit import SlitBase
from .slit_detect import SlitMeasurement, SlitNotFound, measure_slit
from .telescope import TelescopeBase

log = logging.getLogger(__name__)

Expose = Callable[[CaptureSettings], Awaitable[Frame]]
Progress = Callable[[float, str], None]

DEFAULT_WIDTH_ARCSEC = 8.0  # a strip: narrower than the slit is long, wide enough to find easily
SCALE_TOLERANCE = 0.3  # the solver is told the plate scale is the nominal one, within this
WIDTH_AGREEMENT = 0.2  # a measured width this far from the commanded one is suspicious
ZERO_ERROR_WARN_DEG = 1.5  # the slit this far from where the rotator says it points
LOW_SIGNIFICANCE = 15.0  # a gap that stands out by less than this many sigma is measured poorly


class Measurement(BaseModel):
    taken_ms: int
    by: str
    # how it was taken
    exposure_s: float
    gain: int
    binning: int
    rot_pa: float  # the rotator's slit PA during the frames
    slit_position_um: float  # the stage, for the open frame
    slit_width_commanded_arcsec: float
    # the plate solve (the closed frame)
    plate_scale_arcsec_px: float  # per unbinned pixel
    north_deg: float  # angle of sky north in the image at this PA, counter-clockwise from the top
    parity: int
    orientation_deg: float  # north at PA 0: north + parity * PA
    solve_confidence: str
    solve_logodds: float
    solve_stars: int
    # the slit (the difference of the two frames), unbinned sensor pixels
    slit_x: float
    slit_y: float
    slit_angle_deg: float  # the long axis, counter-clockwise from the top, (-90, 90]
    slit_length_px: float
    slit_width_px: float
    slit_snr: float  # the gap's contrast over the pixel noise, one pixel
    slit_significance: float  # how many sigma the whole gap stands out by
    # derived
    slit_length_arcsec: float
    slit_width_arcsec: float  # measured, to compare with the commanded one
    rotator_zero_error_deg: float  # slit angle minus the orientation, (-90, 90]
    warnings: list[str] = Field(default_factory=list)
    closed_frame_id: int
    open_frame_id: int


_last: Measurement | None = None


def last() -> Measurement | None:
    """The most recent measurement, not necessarily applied."""
    return _last


def remember(m: Measurement) -> None:
    global _last
    _last = m


def reset() -> None:
    """Forget the last measurement (for tests)."""
    global _last
    _last = None


def to_sensor(x: float, y: float, binning: int, roi_x: int = 0, roi_y: int = 0):
    """A position in a (binned, windowed) frame in unbinned sensor pixels: the centre of binned
    pixel ``i`` is the middle of the ``binning`` unbinned pixels it sums."""
    return x * binning + (binning - 1) / 2 + roi_x, y * binning + (binning - 1) / 2 + roi_y


def wrap_line(deg: float) -> float:
    """An angle of a line (not an arrow) in (-90, 90]."""
    d = (deg + 90.0) % 180.0 - 90.0
    return 90.0 if d == -90.0 else d


def wrap_arrow(deg: float) -> float:
    """An angle in (-180, 180]."""
    d = (deg + 180.0) % 360.0 - 180.0
    return 180.0 if d == -180.0 else d


def combine(
    solution: FieldSolution,
    slit: SlitMeasurement,
    *,
    binning: int,
    roi: tuple[int, int],
    rot_pa: float,
) -> dict:
    """The calibration numbers from a solve and a slit measurement (both in the frames' pixels):
    unbinned plate scale, orientation at PA 0, the slit's place, direction and size in pixels and
    arcsec, and how far the slit points from where the rotator says."""
    scale = solution.scale_arcsec_px / binning  # per unbinned pixel
    orientation = wrap_arrow(solution.north_deg + solution.parity * rot_pa)
    x, y = to_sensor(slit.x, slit.y, binning, *roi)
    length_px, width_px = slit.length_px * binning, slit.width_px * binning
    return {
        "plate_scale_arcsec_px": scale,
        "orientation_deg": orientation,
        "parity": solution.parity,
        "slit_x": x,
        "slit_y": y,
        "slit_angle_deg": slit.angle_deg,
        "slit_length_px": length_px,
        "slit_width_px": width_px,
        "slit_length_arcsec": length_px * scale,
        "slit_width_arcsec": width_px * scale,
        "rotator_zero_error_deg": wrap_line(slit.angle_deg - orientation),
    }


def judge(m: dict, commanded_arcsec: float, significance: float) -> list[str]:
    """What looks wrong in a measurement, in words for the operator."""
    out = []
    if significance < LOW_SIGNIFICANCE:
        out.append(
            f"the slit is faint in the frames (it stands out by {significance:.0f} sigma): a "
            "longer exposure, a higher gain or a wider gap would measure it better"
        )
    if abs(m["slit_width_arcsec"] - commanded_arcsec) > WIDTH_AGREEMENT * commanded_arcsec:
        out.append(
            f"the gap measures {m['slit_width_arcsec']:.2f} arcsec but the stage was told "
            f"{commanded_arcsec:.2f}: check the microns-per-arcsec setting and the plate scale"
        )
    if abs(m["rotator_zero_error_deg"]) > ZERO_ERROR_WARN_DEG:
        out.append(
            f"the slit points {m['rotator_zero_error_deg']:+.2f} degrees from where the rotator "
            "says north is at PA 0: the rotator's zero may need correcting"
        )
    return out


async def measure(
    *,
    expose: Expose,
    slit: SlitBase,
    telescope: TelescopeBase,
    solver: FieldSolver,
    settings: CaptureSettings,
    width_arcsec: float,
    nominal_scale_arcsec_px: float,
    who: str,
    progress: Progress = lambda f, m: None,
) -> Measurement:
    """Run the procedure (see the module docstring). Raises :class:`SolveFailed` or
    :class:`SlitNotFound` (the router turns them into an error) and puts the slit stage back
    where it was, whatever happens."""
    t = telescope.telemetry()
    if t.target is None:
        raise SolveFailed("the telescope has no target: slew to a star field first")
    rot_pa = t.rot_pa or 0.0
    binning = settings.binning
    roi = (settings.roi.x, settings.roi.y) if settings.roi else (0, 0)
    um = slit.um_for_arcsec(width_arcsec)

    async with slit.holding():
        progress(0.05, "closing the slit")
        await slit.move_to_preset("closed")
        progress(0.12, f"closed-slit frame ({settings.exposure_s:g} s)")
        closed = await expose(settings)

        progress(0.4, "plate solving")
        nominal = nominal_scale_arcsec_px * binning
        solution = await solver.solve(
            closed.pixels,
            ra_deg=t.target.ra_hours * 15.0,
            dec_deg=t.target.dec_deg,
            scale_arcsec_px=(nominal * (1 - SCALE_TOLERANCE), nominal * (1 + SCALE_TOLERANCE)),
        )

        progress(0.55, f"opening the slit to {width_arcsec:g} arcsec")
        await slit.move_to(um)
        progress(0.62, f"open-slit frame ({settings.exposure_s:g} s)")
        opened = await expose(settings)
        position = slit.position_um if slit.position_um is not None else um

    progress(0.9, "finding the slit")
    found = measure_slit(closed.pixels.astype(np.float64), opened.pixels.astype(np.float64))
    values = combine(solution, found, binning=binning, roi=roi, rot_pa=rot_pa)
    commanded = position / slit.conf.current.um_per_arcsec
    progress(1.0, "done")
    return Measurement(
        taken_ms=int(time.time() * 1000),
        by=who,
        exposure_s=settings.exposure_s,
        gain=settings.gain,
        binning=binning,
        rot_pa=rot_pa,
        slit_position_um=position,
        slit_width_commanded_arcsec=commanded,
        north_deg=solution.north_deg,
        solve_confidence=solution.confidence,
        solve_logodds=solution.logodds,
        solve_stars=solution.stars,
        slit_snr=found.snr,
        slit_significance=found.significance,
        warnings=judge(values, commanded, found.significance),
        closed_frame_id=closed.meta.id,
        open_frame_id=opened.meta.id,
        **values,
    )


def cal_values(m: Measurement) -> dict:
    """What applying a measurement sets in the guide camera's calibration. Not the slit's width:
    that was measured at the measuring width, not the working one, and the overlay follows the
    stage."""
    return {
        "plate_scale_arcsec_px": round(m.plate_scale_arcsec_px, 5),
        "orientation_deg": round(m.orientation_deg, 3),
        "parity": m.parity,
        "slit_pixel_x": round(m.slit_x, 2),
        "slit_pixel_y": round(m.slit_y, 2),
        "slit_angle_deg": round(m.slit_angle_deg, 3),
        "slit_length_arcsec": round(m.slit_length_arcsec, 2),
    }


__all__ = [
    "DEFAULT_WIDTH_ARCSEC",
    "Measurement",
    "SlitNotFound",
    "SolveFailed",
    "cal_values",
    "combine",
    "judge",
    "last",
    "measure",
    "remember",
    "reset",
    "to_sensor",
    "wrap_arrow",
    "wrap_line",
]
