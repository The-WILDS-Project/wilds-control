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
    x, y = (round(c) for c in o.gap_centre(15.7))
    assert wide[y, x] < 0.1 * shut[y, x]  # the gap is dark where the slit is
    assert wide[10, 10] == pytest.approx(shut[10, 10], rel=0.05)  # and nothing changes far away


@pytest.mark.parametrize("side", [1, -1])
def test_only_one_jaw_moves_so_the_gap_grows_from_a_fixed_edge(side):
    o = SimulatedOptics(slit_side=side, slit_angle_deg=0.0)  # the slit is vertical: across is x
    shut = camera(o, [0.016])._electrons(CaptureSettings(exposure_s=EXPOSURE))
    x0, y = round(o.slit_x_px), round(o.slit_y_px)
    for width in (4.0, 16.0):
        e = camera(o, [width])._electrons(CaptureSettings(exposure_s=EXPOSURE))
        edge_px = width / o.plate_scale_arcsec_px
        inside = x0 + side * round(edge_px / 2)  # half way across the gap
        beyond = x0 - side * 6  # just the other side of the fixed edge: jaw, not gap
        far = x0 + side * round(edge_px + 6)  # past the moving edge: jaw again
        assert e[y, inside] < 0.1 * shut[y, inside]
        assert e[y, beyond] == pytest.approx(shut[y, beyond], rel=0.05)
        assert e[y, far] == pytest.approx(shut[y, far], rel=0.05)
    cx, cy = o.gap_centre(16.0)
    assert (cx - x0) * side == pytest.approx(16.0 / o.plate_scale_arcsec_px / 2)  # half the width
    assert cy == pytest.approx(o.slit_y_px)


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
    x, y = (round(c) for c in o.gap_centre(15.7))
    assert e[y, x] >= dark


# -- finding it ---------------------------------------------------------------------------


@pytest.mark.parametrize("angle", [31.4, -40.0, 0.0, 88.0])
def test_the_wide_open_gap_is_found_with_its_direction_and_size(angle):
    o = SimulatedOptics(slit_angle_deg=angle)
    m = measure_slit(*pair(15.7, o))
    assert (m.x, m.y) == pytest.approx(o.gap_centre(15.7), abs=0.5)
    assert m.angle_deg == pytest.approx(angle, abs=0.4)
    assert m.length_px == pytest.approx(o.slit_length_arcsec / o.plate_scale_arcsec_px, abs=2.0)
    assert m.width_px == pytest.approx(15.7 / o.plate_scale_arcsec_px, abs=2.0)


@pytest.mark.parametrize("angle", [31.4, -62.0])
def test_the_working_one_arcsec_slit_is_found_too_if_cut_short(angle):
    o = SimulatedOptics(slit_angle_deg=angle)
    m = measure_slit(*pair(1.0, o, exposure_s=NARROW_EXPOSURE))
    assert m.angle_deg == pytest.approx(angle, abs=0.5)
    assert m.width_px == pytest.approx(1.0 / o.plate_scale_arcsec_px, abs=0.5)
    # Across the strip the position is tight. Along it a narrow gap is found less well: the
    # detector cuts a 90 arcsec strip short at one or both ends (a fifth to a quarter of it, the
    # same for any exposure), which moves its middle. That is why the calibration takes where the
    # slit is along its length from the wide gap alone (see optics_calibration.combine).
    (ux, uy), (vx, vy) = (
        (-np.sin(np.radians(angle)), -np.cos(np.radians(angle))),
        (np.cos(np.radians(angle)), -np.sin(np.radians(angle))),
    )
    dx, dy = m.x - o.gap_centre(1.0)[0], m.y - o.gap_centre(1.0)[1]
    assert abs(dx * vx + dy * vy) < 0.6  # across
    true_length = o.slit_length_arcsec / o.plate_scale_arcsec_px
    assert abs(dx * ux + dy * uy) < 0.15 * true_length  # along
    assert 0.7 * true_length < m.length_px <= 1.02 * true_length


def test_two_closed_frames_have_no_gap_to_find():
    closed, again = pair(0.01)
    with pytest.raises(SlitNotFound):
        measure_slit(closed, again)


def test_frames_of_different_sizes_are_refused():
    with pytest.raises(ValueError, match="differ in size"):
        measure_slit(np.zeros((10, 10)), np.zeros((10, 11)))
