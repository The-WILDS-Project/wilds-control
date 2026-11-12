"""Real stars in the simulated guide camera: the projection, the cone cache and the catalogue reader."""

import math

import numpy as np
import pytest

from wilds.control import sky
from wilds.control.guide_camera import SimulatedASICamera
from wilds.control.guide_camera.base import CaptureSettings
from wilds.control.guide_camera.simulated import SENSOR
from wilds.control.sky import (
    Geometry,
    Sky,
    Stars,
    pointing_after_offsets,
    project,
    tangent_offsets,
)

RA0, DEC0 = 52.5, 21.25  # the target


def stars_at(*offsets_arcsec: tuple[float, float], mag: float = 12.0) -> Stars:
    """Stars east and north of the target by these arcsec."""
    ra = [RA0 + e / 3600.0 / math.cos(math.radians(DEC0 + n / 3600.0)) for e, n in offsets_arcsec]
    dec = [DEC0 + n / 3600.0 for _, n in offsets_arcsec]
    return Stars(np.array(ra), np.array(dec), np.full(len(ra), mag))


def geo(**kw) -> Geometry:
    values = {
        "plate_scale_arcsec_px": 0.25,
        "slit_x": 648.0,
        "slit_y": 441.0,
        "orientation_deg": 0.0,
        "parity": 1,
    }
    return Geometry(**{**values, **kw})


# -- the projection ---------------------------------------------------------------------------


def test_tangent_offsets_are_east_and_north_arcsec_and_far_side_is_nan():
    xi, eta = tangent_offsets(
        np.array([RA0 + 1 / 3600 / math.cos(math.radians(DEC0))]), np.array([DEC0]), RA0, DEC0
    )
    assert xi[0] == pytest.approx(1.0, abs=1e-3) and eta[0] == pytest.approx(0.0, abs=1e-3)
    xi, eta = tangent_offsets(np.array([RA0 + 180.0]), np.array([-DEC0]), RA0, DEC0)
    assert np.isnan(xi[0]) and np.isnan(eta[0])


def test_the_target_lands_on_the_slit_pixel():
    x, y, _ = project(stars_at((0, 0)), RA0, DEC0, geo())
    assert (x[0], y[0]) == pytest.approx((648.0, 441.0), abs=1e-3)


def test_north_is_up_and_east_is_left_without_rotation():
    x, y, _ = project(stars_at((0, 10), (10, 0)), RA0, DEC0, geo())
    assert (x[0], y[0]) == pytest.approx((648.0, 441.0 - 40.0), abs=0.01)  # 10" north = 40 px up
    assert (x[1], y[1]) == pytest.approx((648.0 - 40.0, 441.0), abs=0.01)  # 10" east = 40 px left
    x, y, _ = project(stars_at((10, 0)), RA0, DEC0, geo(parity=-1))  # a mirrored image
    assert x[0] == pytest.approx(648.0 + 40.0, abs=0.01)


def test_the_orientation_and_the_rotator_turn_the_sky_in_the_image():
    # north 90 degrees counter-clockwise from up is to the left
    x, y, _ = project(stars_at((0, 10)), RA0, DEC0, geo(orientation_deg=90.0))
    assert (x[0], y[0]) == pytest.approx((648.0 - 40.0, 441.0), abs=0.01)
    # the camera turns with the rotator: at PA 90 the sky has turned 90 degrees clockwise
    x, y, _ = project(stars_at((0, 10)), RA0, DEC0, geo(pa_deg=90.0))
    assert (x[0], y[0]) == pytest.approx((648.0 + 40.0, 441.0), abs=0.01)
    # ... and the slit, which stays put, now points 90 degrees east of north: along image east
    north, east = geo(pa_deg=90.0).basis()
    assert math.hypot(*north) == pytest.approx(1.0) and math.hypot(*east) == pytest.approx(1.0)


def test_an_offset_moves_the_pointing_and_so_the_stars_the_other_way():
    ra, dec = pointing_after_offsets(RA0, DEC0, 5.0, 0.0)  # 5" east
    x, y, _ = project(stars_at((0, 0)), ra, dec, geo())
    # the target is now 5" west of the pointing: to the right in the image (east is left)
    assert x[0] == pytest.approx(648.0 + 20.0, abs=0.01) and y[0] == pytest.approx(441.0, abs=0.01)
    ra, dec = pointing_after_offsets(RA0, DEC0, 0.0, 5.0)
    x, y, _ = project(stars_at((0, 0)), ra, dec, geo())
    assert y[0] == pytest.approx(441.0 + 20.0, abs=0.01)  # north: the target moves down


def test_brighter_stars_give_more_flux():
    f = sky.flux_from_mag(np.array([10.0, 14.0, 15.0]))
    assert f[1] == pytest.approx(sky.ZERO_FLUX) and f[0] > f[1] > f[2]
    assert f[0] / f[1] == pytest.approx(10 ** (0.4 * 4))


# -- the cone cache -------------------------------------------------------------------------------


class CountingSource:
    def __init__(self, fail: bool = False):
        self.calls: list[tuple[float, float, float]] = []
        self.fail = fail

    def cone(self, ra_deg, dec_deg, radius_deg):
        self.calls.append((ra_deg, dec_deg, radius_deg))
        if self.fail:
            raise OSError("share unreachable")
        return stars_at((0, 0), (30, 20), (-40, 10))


def test_one_cone_serves_many_frames_until_the_pointing_leaves_it():
    source = CountingSource()
    s = Sky(source, SENSOR)
    assert len(s.stars_for(RA0, DEC0, geo())[0]) == 3
    s.stars_for(RA0, DEC0, geo())
    ra, dec = pointing_after_offsets(RA0, DEC0, 10.0, -10.0)  # a nudge: well inside the margin
    s.stars_for(ra, dec, geo())
    assert len(source.calls) == 1
    s.stars_for(RA0, DEC0 + 1.0, geo())  # a degree away: a new cone
    assert len(source.calls) == 2


def test_an_unreadable_catalogue_gives_none_and_is_not_asked_again_at_once(monkeypatch):
    source = CountingSource(fail=True)
    s = Sky(source, SENSOR, retry_s=30.0)
    now = [100.0]
    monkeypatch.setattr(sky.time, "monotonic", lambda: now[0])
    assert s.stars_for(RA0, DEC0, geo()) is None and "unreachable" in s.failed
    assert s.stars_for(RA0, DEC0, geo()) is None
    assert len(source.calls) == 1  # a dead share is slow: not once per frame
    now[0] += 31.0
    source.fail = False
    assert s.stars_for(RA0, DEC0, geo()) is not None and s.failed is None
    assert len(source.calls) == 2


# -- the catsHTM reader's adapter -------------------------------------------------------------------


def test_the_catshtm_source_converts_radians_and_keeps_the_brightest(monkeypatch):
    from wilds.control.catalogs import catshtm

    asked = {}

    class Fake:
        def __init__(self, root, name):
            asked["open"] = (root, name)

        def cone_search(self, ra, dec, radius, columns):
            asked["cone"] = (ra, dec, radius, list(columns))
            return {
                "RA": np.radians([10.0, 10.1, 10.2, 10.3]),
                "Dec": np.radians([5.0, 5.1, 5.2, 5.3]),
                "phot_g_mean_mag": np.array([12.0, np.nan, 19.5, 15.0]),
            }

    monkeypatch.setattr(catshtm, "CatsHtm", Fake)
    source = sky.CatsHtmSource("/data/GAIA/DR3", "GAIADR3", mag_limit=18.0, max_stars=10)
    got = source.cone(10.0, 5.0, 0.1)
    assert asked["open"] == ("/data/GAIA/DR3", "GAIADR3")
    assert asked["cone"] == (10.0, 5.0, 0.1, ["RA", "Dec", "phot_g_mean_mag"])
    assert got.ra_deg.tolist() == pytest.approx([10.0, 10.3]) and got.mag.tolist() == [
        12.0,
        15.0,
    ]  # no NaN, no faint
    assert sky.CatsHtmSource("/x", max_stars=1).cone(10.0, 5.0, 0.1).mag.tolist() == [12.0]


# -- the simulated camera draws them ------------------------------------------------------------------


def test_the_simulated_camera_draws_the_real_stars_where_they_project():
    stars = stars_at((0, 0), mag=10.0)  # one bright star on the target, among faint ones
    x, y, flux = project(stars, RA0, DEC0, geo())
    camera = SimulatedASICamera(
        jitter_px=0.0, drift_x_px_s=0.0, drift_y_px_s=0.0, sky=lambda: (x, y, flux)
    )
    camera.seeing_fwhm_px = 3.0
    image = camera._electrons(CaptureSettings(exposure_s=0.5))
    py, px = np.unravel_index(np.argmax(image), image.shape)
    assert (px, py) == (648, 441)


def test_the_camera_without_a_sky_still_draws_its_invented_field():
    camera = SimulatedASICamera(jitter_px=0.0, drift_x_px_s=0.0, drift_y_px_s=0.0)
    assert camera.sky() is None
    assert camera._electrons(CaptureSettings(exposure_s=0.5)).max() > 100


def test_the_cameras_real_sky_follows_the_telescopes_target_pointing_and_rotator():
    from wilds.control import guide_camera, telescope
    from wilds.control.guide_camera import real_sky
    from wilds.control.guide_camera.optics import SimulatedOptics
    from wilds.control.telescope import Offset, Target

    camera = guide_camera.get_guide_camera()
    camera.optics = SimulatedOptics(slit_x_px=648.0, slit_y_px=441.0)  # mid-frame, as the stars
    s = Sky(CountingSource(), SENSOR)
    tel = telescope.get_telescope()
    tel.set_preview(False)
    assert real_sky(camera, s) is None  # no target yet: the invented field
    tel._send_target(Target(name="x", ra_hours=RA0 / 15.0, dec_deg=DEC0), 0.0, "Target")
    x, y, _ = real_sky(camera, s)
    here = (x[np.argmin(np.hypot(x - 648, y - 441))], y[np.argmin(np.hypot(x - 648, y - 441))])
    assert here == pytest.approx((648.0, 441.0), abs=0.05)  # the star on the target: on the slit
    tel._send_offset("Handset", Offset(d1_arcsec=5.0, d2_arcsec=0.0), "TPLANE")
    x2, _, _ = real_sky(camera, s)
    assert x2.max() > x.max() - 1e-9 and not np.allclose(np.sort(x2), np.sort(x))  # the field moved


def test_a_saved_field_is_read_back_and_answers_cones_inside_it(tmp_path):
    stars = sky.Stars(
        np.array([250.40, 250.42, 251.5]),
        np.array([36.45, 36.46, 36.46]),
        np.array([12.0, 15.0, 9.0]),
    )
    path = tmp_path / "m13.npz"
    sky.save_field(path, 250.42, 36.46, 0.5, stars)
    source = sky.FieldFileSource(path)
    assert source.centre == (250.42, 36.46) and source.radius_deg == 0.5
    got = source.cone(250.41, 36.455, 0.1)
    assert sorted(got.mag.tolist()) == [12.0, 15.0]  # the one at 251.5 is outside the cone
    assert len(source.cone(10.0, -20.0, 0.1)) == 0  # nothing outside the saved field


def test_a_relative_field_file_is_found_from_the_checkout_wherever_the_server_starts(
    config_file, tmp_path, monkeypatch
):
    from wilds.control import guide_camera

    config_file('[sky]\nfield_file = "data/sky/m13.npz"\n')
    monkeypatch.chdir(tmp_path)  # not the control folder
    assert guide_camera._make_sky() is not None
