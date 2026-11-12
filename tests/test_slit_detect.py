"""The slit shadow in the simulated camera, and finding it again (``slit_detect``)."""

import numpy as np
import pytest

from wilds.control.guide_camera import SimulatedASICamera
from wilds.control.guide_camera.base import CaptureSettings
from wilds.control.guide_camera.optics import SimulatedOptics
from wilds.control.slit_detect import SlitNotFound, measure_slit

EXPOSURE = 2.0  # seconds: a bright sky, 20 e/s, so a wide gap is clear in a short frame
# The 1 arcsec gap is only ~4 px wide: per pixel it is a few sigma at 2 s, which pins its ends to
# a few pixels along a 110 px strip. A longer exposure (or stacked frames) is what the working
# slit needs.
NARROW_EXPOSURE = 8.0


def camera(optics: SimulatedOptics | None = None, width: list | None = None) -> SimulatedASICamera:
    """A still camera (no jitter or drift between the two frames) whose slit gap is ``width[0]``
    arcsec, so a test can change it between frames."""
    width = width if width is not None else [None]
    cam = SimulatedASICamera(
        jitter_px=0.0,
        drift_x_px_s=0.0,
        drift_y_px_s=0.0,
        optics=optics or SimulatedOptics(),
        slit_width=lambda: width[0],
    )
    cam._rng = np.random.default_rng(3)
    return cam


def frame(cam: SimulatedASICamera, rng, exposure_s: float = EXPOSURE) -> np.ndarray:
    """A frame as a sensor records it: shot noise and read noise on the electrons."""
    e = cam._electrons(CaptureSettings(exposure_s=exposure_s))
    return rng.poisson(e).astype(np.float64) + rng.normal(0, 3.0, e.shape)


def pair(
    width_arcsec: float,
    optics: SimulatedOptics | None = None,
    seed: int = 11,
    exposure_s: float = EXPOSURE,
):
    """A closed frame and a frame with the gap ``width_arcsec`` wide, of the same sky."""
    gap = [0.01]
    cam = camera(optics, gap)
    rng = np.random.default_rng(seed)
    closed = frame(cam, rng, exposure_s)
    gap[0] = width_arcsec
    return closed, frame(cam, rng, exposure_s)


# -- the shadow ---------------------------------------------------------------------------


def test_an_open_slit_casts_a_dark_gap_and_a_closed_one_almost_none():
    o = SimulatedOptics()
    wide = camera(o, [15.7])._electrons(CaptureSettings(exposure_s=EXPOSURE))
    shut = camera(o, [0.016])._electrons(CaptureSettings(exposure_s=EXPOSURE))
    x, y = round(o.slit_x_px), round(o.slit_y_px)
    assert wide[y, x] < 0.1 * shut[y, x]  # the gap is dark where the slit is
    assert wide[10, 10] == pytest.approx(shut[10, 10], rel=0.05)  # and nothing changes far away


def test_with_no_width_there_is_no_shadow():
    a = camera(width=[None])._electrons(CaptureSettings(exposure_s=1.0))
    b = camera(width=[0.0])._electrons(CaptureSettings(exposure_s=1.0))
    assert np.allclose(a, b)


def test_the_shadow_does_not_dim_the_dark_current():
    cam = camera(width=[15.7])
    cam.light_blocked = lambda: False
    o = cam.optics
    e = cam._electrons(CaptureSettings(exposure_s=10.0))
    # inside the gap what is left is the leak of the light plus the (untouched) dark current
    dark = 0.4 * 10.0
    assert e[round(o.slit_y_px), round(o.slit_x_px)] >= dark


# -- finding it ---------------------------------------------------------------------------


@pytest.mark.parametrize("angle", [31.4, -40.0, 0.0, 88.0])
def test_the_wide_open_gap_is_found_with_its_direction_and_size(angle):
    o = SimulatedOptics(slit_angle_deg=angle)
    m = measure_slit(*pair(15.7, o))
    assert (m.x, m.y) == pytest.approx((o.slit_x_px, o.slit_y_px), abs=0.5)
    assert m.angle_deg == pytest.approx(angle, abs=0.4)
    assert m.length_px == pytest.approx(o.slit_length_arcsec / o.plate_scale_arcsec_px, abs=2.0)
    assert m.width_px == pytest.approx(15.7 / o.plate_scale_arcsec_px, abs=2.0)


@pytest.mark.parametrize("angle", [31.4, -62.0])
def test_the_working_one_arcsec_slit_is_found_too(angle):
    o = SimulatedOptics(slit_angle_deg=angle)
    m = measure_slit(*pair(1.0, o, exposure_s=NARROW_EXPOSURE))
    assert m.angle_deg == pytest.approx(angle, abs=0.5)
    assert m.width_px == pytest.approx(1.0 / o.plate_scale_arcsec_px, abs=0.5)
    # Across the strip the position is tight; along it the ends are looser: a very bright star
    # at one end leaves a residual (the seeing differs a little between the two frames), which
    # moves that end by a few pixels of 110.
    (ux, uy), (vx, vy) = (
        (-np.sin(np.radians(angle)), -np.cos(np.radians(angle))),
        (np.cos(np.radians(angle)), -np.sin(np.radians(angle))),
    )
    dx, dy = m.x - o.slit_x_px, m.y - o.slit_y_px
    assert abs(dx * vx + dy * vy) < 0.6  # across
    assert abs(dx * ux + dy * uy) < 2.5  # along
    assert m.length_px == pytest.approx(o.slit_length_arcsec / o.plate_scale_arcsec_px, abs=5.0)


def test_two_closed_frames_have_no_gap_to_find():
    closed, again = pair(0.01)
    with pytest.raises(SlitNotFound):
        measure_slit(closed, again)


def test_frames_of_different_sizes_are_refused():
    with pytest.raises(ValueError, match="differ in size"):
        measure_slit(np.zeros((10, 10)), np.zeros((10, 11)))
