"""The plate solver the optics calibration uses: choosing one, and its failures; and, when the
astrometry.net index files for M13's patch of sky are on this machine, the real thing against
frames of the simulated M13 field.

The real solver needs series-5200 index files (``$WILDS_ASTROMETRY_CACHE`` or ``~/.astrometry``,
folder ``5200``): the tile that holds M13 is number 10 in astrometry.net's own numbering (a
HEALPix nside-2 tiling whose sub-tile bits are swapped against astropy's), scale 1 (2.8-4
arcmin quads, ``index-5201-10.fits``) and scale 0 (``index-5200-10.fits``). Without them those
tests are skipped."""

import asyncio
import os
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from wilds.control import guide_camera, lease, telescope
from wilds.control.activities import registry
from wilds.control.app import create_app
from wilds.control.field_solver import (
    AstrometrySolver,
    FieldSolution,
    SimulatedSolver,
    SolveFailed,
    get_field_solver,
)
from wilds.control.guide_camera.base import CaptureSettings
from wilds.control.guide_camera.optics import SimulatedOptics
from wilds.control.jobs import manager
from wilds.control.telescope import Target

TO = {"X-TO-Passphrase": "to-pass"}
M13_RA_H, M13_DEC = 16.6949, 36.4613
M13_FIELD = Path(__file__).resolve().parents[1] / "data" / "sky" / "m13.npz"
INDEX_DIR = Path(os.environ.get("WILDS_ASTROMETRY_CACHE") or "~/.astrometry").expanduser()


def _complete(scale: int, tile: int = 10) -> bool:
    """Is that index file here in full? (One still being copied does not count.)"""
    from astrometry import series_5200

    path = INDEX_DIR / "5200" / f"index-52{scale:02d}-{tile:02d}.fits"
    try:
        return path.stat().st_size == series_5200.scale_to_sizes[scale][tile]
    except OSError:
        return False


SCALES = [s for s in (0, 1) if _complete(s)]  # the scales of M13's tile that are here
HAVE_INDEXES = bool(SCALES)
needs_indexes = pytest.mark.skipif(
    not HAVE_INDEXES,
    reason=f"no astrometry.net index files for M13's tile in {INDEX_DIR / '5200'}",
)


# -- always: choosing a solver, and failing clearly -----------------------------------------


def test_the_solver_comes_from_the_config(config_file):
    config_file('[solver]\nkind = "astrometry"\nindex_dir = "/somewhere/indexes"\nscales = [1]\n')
    solver = get_field_solver()
    assert isinstance(solver, AstrometrySolver)
    assert solver.index_dir == Path("/somewhere/indexes") and solver.scales == {1}


def test_the_simulated_kind_needs_the_simulated_camera_and_is_the_simulated_solver(config_file):
    config_file('[solver]\nkind = "simulated"\n')
    assert isinstance(get_field_solver(), SimulatedSolver)


def test_without_index_files_a_solve_fails_clearly_and_does_not_download(tmp_path):
    solver = AstrometrySolver(tmp_path, [0, 1])
    with pytest.raises(SolveFailed, match="no astrometry.net index files"):
        asyncio.run(
            solver.solve(
                np.zeros((20, 20)), ra_deg=250.0, dec_deg=36.0, scale_arcsec_px=(0.2, 0.35)
            )
        )
    assert not (tmp_path / "5200").exists() or not any((tmp_path / "5200").iterdir())


def test_the_simulated_solver_answers_with_the_cameras_true_optics_at_the_rotators_angle():
    o = SimulatedOptics(plate_scale_arcsec_px=0.3, orientation_deg=40.0, parity=-1)
    solver = SimulatedSolver(lambda: o, lambda: 25.0, lambda: (250.0, 36.0), 1296)
    frame = np.zeros((441, 648))  # binned 2x2
    s = asyncio.run(solver.solve(frame, ra_deg=0, dec_deg=0, scale_arcsec_px=(0.1, 0.9)))
    assert isinstance(s, FieldSolution)
    assert s.scale_arcsec_px == pytest.approx(0.6, rel=0.01)  # per binned pixel
    assert s.parity == -1
    assert s.north_deg == pytest.approx(40.0 - -1 * 25.0, abs=0.2)  # orientation - parity * PA


# -- with the index files: the real solver -----------------------------------------------------


def simulated_m13_frame(optics: SimulatedOptics, pa: float, binning: int = 1) -> np.ndarray:
    """A frame of the simulated camera looking at M13 with the rotator at ``pa``, as a sensor
    records it (shot and read noise), through the real M13 star data."""
    cam = guide_camera.get_guide_camera()
    cam.optics = optics
    tel = telescope.get_telescope()
    tel.set_preview(False)
    tel._send_target(Target(name="M13", ra_hours=M13_RA_H, dec_deg=M13_DEC), pa, "Target")
    later = tel.clock() + 10_000.0
    tel.clock = lambda: later  # long after: the rotator has arrived
    e = cam._electrons(CaptureSettings(exposure_s=2.0, gain=100, binning=binning))
    rng = np.random.default_rng(5)
    return (rng.poisson(e) + rng.normal(0, 3.0, e.shape)).astype(np.float64)


@pytest.fixture
def m13(config_file):
    config_file(f'[sky]\nfield_file = "{M13_FIELD}"\n')


@needs_indexes
@pytest.mark.parametrize("parity", [1, -1])
@pytest.mark.parametrize("pa", [0.0, 20.0, 200.0])
def test_the_real_solver_solves_the_simulated_m13_field(m13, parity, pa):
    o = SimulatedOptics(parity=parity)
    frame = simulated_m13_frame(o, pa)
    solver = AstrometrySolver(INDEX_DIR, SCALES)
    started = time.monotonic()
    s = asyncio.run(
        solver.solve(frame, ra_deg=M13_RA_H * 15, dec_deg=M13_DEC, scale_arcsec_px=(0.19, 0.35))
    )
    assert time.monotonic() - started < 60
    assert s.confidence in ("good", "high", "absolute")
    assert s.scale_arcsec_px == pytest.approx(o.plate_scale_arcsec_px, rel=0.01)
    assert s.parity == parity
    expect = o.orientation_deg - parity * pa  # north, counter-clockwise from the top
    assert (s.north_deg - expect + 180.0) % 360.0 - 180.0 == pytest.approx(0.0, abs=0.5)
    assert (s.ra_deg, s.dec_deg) == pytest.approx((M13_RA_H * 15, M13_DEC), abs=0.05)


@needs_indexes
def test_the_real_solver_solves_a_binned_frame_in_its_own_pixels(m13):
    o = SimulatedOptics()
    frame = simulated_m13_frame(o, 20.0, binning=2)
    solver = AstrometrySolver(INDEX_DIR, SCALES)
    s = asyncio.run(
        solver.solve(frame, ra_deg=M13_RA_H * 15, dec_deg=M13_DEC, scale_arcsec_px=(0.38, 0.7))
    )
    assert s.scale_arcsec_px == pytest.approx(2 * o.plate_scale_arcsec_px, rel=0.01)


@needs_indexes
def test_the_real_solver_says_so_when_the_sky_is_not_in_the_frame(m13):
    rng = np.random.default_rng(2)
    frame = rng.normal(100, 5, (882, 1296))  # no stars at all
    solver = AstrometrySolver(INDEX_DIR, SCALES)
    with pytest.raises(SolveFailed):
        asyncio.run(
            solver.solve(frame, ra_deg=M13_RA_H * 15, dec_deg=M13_DEC, scale_arcsec_px=(0.19, 0.35))
        )


# -- with the index files: the calibration job through the real solver -----------------------


@pytest.fixture
def real_app(config_file, passphrases_file):
    config_file(
        f'[sky]\nfield_file = "{M13_FIELD}"\n'
        f'[solver]\nkind = "astrometry"\nindex_dir = "{INDEX_DIR}"\nscales = {SCALES}\n'
    )
    passphrases_file('[passphrases]\nto = "to-pass"\nengineering = "eng-pass"\n')
    lease.state.sessions.clear()
    lease.state.holder_id = lease.state.token = None
    lease.state.engineering = False
    manager.reset()
    registry.clear()
    yield create_app()
    manager.reset()
    registry.clear()


def hold_lease(client) -> dict[str, str]:
    s = client.post("/sessions", json={"name": "Arie"}).json()
    sid = {"X-Session-Id": s["session_id"]}
    client.post("/control/requests", headers=sid)
    client.post(f"/control/requests/{s['code']}/grant", headers=TO)
    token = client.post(f"/sessions/{s['session_id']}/heartbeat").json()["lease_token"]
    return {**sid, "X-Lease-Token": token}


def finished(client, id, seconds=120.0) -> dict:
    deadline = time.time() + seconds
    while time.time() < deadline:
        job = client.get(f"/jobs/{id}").json()
        if job["state"] not in ("queued", "running"):
            return job
        time.sleep(0.05)
    raise AssertionError("the job did not finish")


@needs_indexes
def test_the_calibration_job_recovers_the_optics_through_the_real_solver(real_app):
    with TestClient(real_app) as client:
        sim = telescope.get_telescope()
        sim.slew_s, sim.rotate_deg_s = 0.1, 400.0
        h = hold_lease(client)
        client.put("/dev/telescope/preview", json={"on": False})
        slew = {"name": "M13", "ra_hours": M13_RA_H, "dec_deg": M13_DEC, "rot_pa": 20.0}
        assert (
            finished(client, client.post("/telescope/slew", json=slew, headers=h).json()["id"])[
                "state"
            ]
            == "done"
        )
        o = guide_camera.get_guide_camera().optics
        job = client.post(
            "/calibration/optics",
            # no calibration yet, and the design scale (0.037"/px) is not the simulator's
            json={"exposure_s": 2.0, "gain": 100, "nominal_scale_arcsec_px": 0.27},
            headers=h,
        ).json()
        done = finished(client, job["id"])
        assert done["state"] == "done", done.get("error")
        m = done["result"]
        assert m["solve_confidence"] in ("good", "high", "absolute")
        assert m["plate_scale_arcsec_px"] == pytest.approx(o.plate_scale_arcsec_px, rel=0.01)
        assert m["parity"] == o.parity
        assert m["orientation_deg"] == pytest.approx(o.orientation_deg, abs=0.5)
        assert (m["slit_x"], m["slit_y"]) == pytest.approx((o.slit_x_px, o.slit_y_px), abs=2.0)
        assert m["slit_angle_deg"] == pytest.approx(o.slit_angle_deg, abs=0.7)
        assert m["slit_width_arcsec"] == pytest.approx(8.0, rel=0.08)
        assert m["warnings"] == []
