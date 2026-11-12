"""Calibrating the guide camera and the slit from the sky: the arithmetic, and the job end to end
against the simulated camera, whose hidden true optics it has to recover."""

import time

import pytest
from fastapi.testclient import TestClient

from wilds.control import guide_camera, lease, telescope
from wilds.control import optics_calibration as oc
from wilds.control.activities import registry
from wilds.control.app import create_app
from wilds.control.field_solver import FieldSolution, SolveFailed, get_field_solver
from wilds.control.jobs import manager
from wilds.control.slit_detect import SlitMeasurement

TO = {"X-TO-Passphrase": "to-pass"}
M13 = {"name": "M13", "ra_hours": 16.6949, "dec_deg": 36.4613, "rot_pa": 20.0}
QUICK = {"exposure_s": 0.3, "gain": 200}  # a bright simulated sky: short frames will do


# -- the arithmetic -----------------------------------------------------------------------


def test_a_binned_windowed_position_is_in_the_middle_of_the_pixels_it_sums():
    assert oc.to_sensor(10, 20, 1) == (10, 20)
    assert oc.to_sensor(10, 20, 2) == (20.5, 40.5)  # binned pixel 10 sums sensor pixels 20 and 21
    assert oc.to_sensor(10, 20, 2, 100, 50) == (120.5, 90.5)


@pytest.mark.parametrize(
    ("deg", "line", "arrow"),
    [
        (0, 0, 0),
        (91, -89, 91),
        (-91, 89, -91),
        (180, 0, 180),
        (-180, 0, 180),
        (270, -90 + 180, -90),
    ],
)
def test_angles_wrap_as_lines_and_as_arrows(deg, line, arrow):
    assert oc.wrap_line(deg) == pytest.approx(oc.wrap_line(line))
    assert oc.wrap_arrow(deg) == pytest.approx(arrow)


def test_combine_converts_binning_scale_and_orientation():
    solution = FieldSolution(0.544, 11.0, -1, 250.0, 36.0, 100.0, "high", 50)  # binned 2x2
    slit = SlitMeasurement(
        x=100.0, y=60.0, angle_deg=30.0, length_px=50.0, width_px=10.0, snr=9.0, significance=80.0
    )
    v = oc.combine(solution, slit, binning=2, roi=(0, 0), rot_pa=20.0)
    assert v["plate_scale_arcsec_px"] == pytest.approx(0.272)
    assert v["orientation_deg"] == pytest.approx(11.0 + -1 * 20.0)  # north + parity * PA
    assert (v["slit_x"], v["slit_y"]) == (200.5, 120.5)
    assert (v["slit_length_px"], v["slit_width_px"]) == (100.0, 20.0)
    assert v["slit_length_arcsec"] == pytest.approx(27.2) and v[
        "slit_width_arcsec"
    ] == pytest.approx(5.44)
    assert v["rotator_zero_error_deg"] == pytest.approx(oc.wrap_line(30.0 - (-9.0)))


def test_judge_says_what_looks_wrong():
    ok = {"slit_width_arcsec": 8.1, "rotator_zero_error_deg": 0.4}
    assert oc.judge(ok, 8.0, 80.0) == []
    words = " ".join(oc.judge({"slit_width_arcsec": 4.0, "rotator_zero_error_deg": 3.0}, 8.0, 4.0))
    assert "faint" in words and "4.00 arcsec" in words and "rotator" in words


# -- the job ------------------------------------------------------------------------------


@pytest.fixture
def app(config_file, passphrases_file):
    config_file('[solver]\nkind = "simulated"\n')
    passphrases_file('[passphrases]\nto = "to-pass"\nengineering = "eng-pass"\n')
    lease.state.sessions.clear()
    lease.state.holder_id = lease.state.token = None
    lease.state.engineering = False
    manager.reset()
    registry.clear()
    yield create_app()
    manager.reset()
    registry.clear()


@pytest.fixture
def client(app):
    with TestClient(app) as c:
        sim = telescope.get_telescope()
        sim.slew_s, sim.rotate_deg_s, sim.offset_s = 0.1, 400.0, 0.1  # quick
        yield c


def hold_lease(client) -> dict[str, str]:
    s = client.post("/sessions", json={"name": "Arie"}).json()
    sid = {"X-Session-Id": s["session_id"]}
    client.post("/control/requests", headers=sid)
    client.post(f"/control/requests/{s['code']}/grant", headers=TO)
    token = client.post(f"/sessions/{s['session_id']}/heartbeat").json()["lease_token"]
    return {**sid, "X-Lease-Token": token}


def wait_for(predicate, seconds=20.0):
    deadline = time.time() + seconds
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.03)
    return False


def finished(client, id) -> dict:
    assert wait_for(lambda: client.get(f"/jobs/{id}").json()["state"] not in ("queued", "running"))
    return client.get(f"/jobs/{id}").json()


def on_target(client, h):
    client.put("/dev/telescope/preview", json={"on": False})
    done = finished(client, client.post("/telescope/slew", json=M13, headers=h).json()["id"])
    assert done["state"] == "done"


@pytest.mark.parametrize("binning", [1, 2])
def test_the_job_recovers_the_cameras_true_optics(client, binning):
    h = hold_lease(client)
    on_target(client, h)
    o = guide_camera.get_guide_camera().optics
    j = client.post(
        "/calibration/optics",
        json={**QUICK, "binning": binning, "slit_width_arcsec": 8.0},
        headers=h,
    )
    assert j.status_code == 202, j.text
    done = finished(client, j.json()["id"])
    assert done["state"] == "done", done.get("error")
    m = done["result"]
    assert m["plate_scale_arcsec_px"] == pytest.approx(o.plate_scale_arcsec_px, rel=0.005)
    assert m["parity"] == o.parity
    assert m["orientation_deg"] == pytest.approx(o.orientation_deg, abs=0.2)  # at PA 0, not PA 20
    assert m["rot_pa"] == pytest.approx(20.0)
    assert (m["slit_x"], m["slit_y"]) == pytest.approx((o.slit_x_px, o.slit_y_px), abs=1.5)
    assert m["slit_angle_deg"] == pytest.approx(o.slit_angle_deg, abs=0.6)
    assert m["slit_length_arcsec"] == pytest.approx(o.slit_length_arcsec, rel=0.06)
    assert m["slit_width_arcsec"] == pytest.approx(8.0, rel=0.06)
    assert m["rotator_zero_error_deg"] == pytest.approx(
        o.slit_angle_deg - o.orientation_deg, abs=0.7
    )
    assert m["warnings"] == []
    assert client.get("/calibration/optics/last").json()["taken_ms"] == m["taken_ms"]


def test_the_slit_goes_back_where_it_was(client):
    h = hold_lease(client)
    on_target(client, h)
    before = client.get("/slit").json()["position_um"]
    done = finished(client, client.post("/calibration/optics", json=QUICK, headers=h).json()["id"])
    assert done["state"] == "done"
    assert client.get("/slit").json()["position_um"] == before


def test_applying_writes_the_calibration_with_who_and_why(client):
    h = hold_lease(client)
    on_target(client, h)
    m = finished(client, client.post("/calibration/optics", json=QUICK, headers=h).json()["id"])[
        "result"
    ]
    assert client.get("/config/guide-camera/cal").json()["values"]["plate_scale_arcsec_px"] is None
    stale = client.post(
        "/calibration/optics/apply", json={"taken_ms": m["taken_ms"] - 1}, headers=h
    )
    assert stale.status_code == 409 and stale.json()["code"] == "conflict"
    r = client.post("/calibration/optics/apply", json={"taken_ms": m["taken_ms"]}, headers=h)
    assert r.status_code == 200
    values = client.get("/config/guide-camera/cal").json()["values"]
    assert values["plate_scale_arcsec_px"] == pytest.approx(m["plate_scale_arcsec_px"], abs=1e-5)
    assert values["slit_pixel_x"] == pytest.approx(m["slit_x"], abs=0.01)
    assert values["slit_angle_deg"] == pytest.approx(m["slit_angle_deg"], abs=0.001)
    assert values["parity"] == m["parity"]
    assert values["slit_width_arcsec"] is None  # the working width is not the measuring width


def test_it_needs_the_lease_and_a_target_and_the_light(client):
    assert client.post("/calibration/optics", json=QUICK).status_code == 403
    h = hold_lease(client)
    no_target = client.post("/calibration/optics", json=QUICK, headers=h)
    assert no_target.status_code == 409 and no_target.json()["code"] == "conflict"
    assert "target" in no_target.json()["message"]
    on_target(client, h)
    client.put("/light-path/state", json={"state": "blocked"}, headers=TO)
    blocked = client.post("/calibration/optics", json=QUICK, headers=h)
    assert blocked.status_code == 409 and "light path" in blocked.json()["message"]


def test_a_gap_outside_the_slits_travel_is_422(client):
    h = hold_lease(client)
    on_target(client, h)
    r = client.post("/calibration/optics", json={**QUICK, "slit_width_arcsec": 500}, headers=h)
    assert r.status_code == 422 and r.json()["code"] == "invalid"


def test_no_gap_is_a_failed_job_that_says_so(client):
    h = hold_lease(client)
    on_target(client, h)
    done = finished(
        client,
        client.post(
            "/calibration/optics", json={**QUICK, "slit_width_arcsec": 0.01}, headers=h
        ).json()["id"],
    )
    assert done["state"] == "failed" and done["error"]["code"] == "slit_not_found"
    assert client.get("/calibration/optics/last").status_code == 404


def test_a_failed_solve_is_a_failed_job_and_the_slit_is_put_back(app):
    class Failing:
        async def solve(self, pixels, **kw):
            raise SolveFailed("no match for 50 stars")

    app.dependency_overrides[get_field_solver] = lambda: Failing()
    with TestClient(app) as client:
        sim = telescope.get_telescope()
        sim.slew_s, sim.rotate_deg_s = 0.1, 400.0
        h = hold_lease(client)
        on_target(client, h)
        before = client.get("/slit").json()["position_um"]
        done = finished(
            client, client.post("/calibration/optics", json=QUICK, headers=h).json()["id"]
        )
        assert done["state"] == "failed" and done["error"]["code"] == "solve_failed"
        assert "no match" in done["error"]["message"]
        assert client.get("/slit").json()["position_um"] == before
