"""Calibrating the guide camera and the slit from the sky: the arithmetic, and the job end to end
against the simulated camera, whose hidden true optics it has to recover."""

import math
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


def gap(x, y, width_px, angle_deg=30.0, length_px=50.0):
    return SlitMeasurement(
        x=x, y=y, angle_deg=angle_deg, length_px=length_px, width_px=width_px, snr=9.0,
        significance=80.0,
    )  # fmt: skip


def test_combine_converts_binning_scale_and_orientation():
    solution = FieldSolution(0.544, 11.0, -1, 250.0, 36.0, 100.0, "high", 50)  # binned 2x2
    v = oc.combine(
        solution, gap(100.0, 60.0, 10.0), gap(100.0, 60.0, 5.0), binning=2, roi=(0, 0), rot_pa=20.0
    )
    assert v["plate_scale_arcsec_px"] == pytest.approx(0.272)
    assert v["orientation_deg"] == pytest.approx(11.0 + -1 * 20.0)  # north + parity * PA
    assert (v["gap_x"], v["gap_y"]) == (200.5, 120.5)
    assert (v["slit_length_px"], v["slit_width_px"]) == (100.0, 20.0)
    assert v["slit_length_arcsec"] == pytest.approx(27.2) and v[
        "slit_width_arcsec"
    ] == pytest.approx(5.44)
    assert v["rotator_zero_error_deg"] == pytest.approx(oc.wrap_line(30.0 - (-9.0)))


@pytest.mark.parametrize("side", [1, -1])
def test_the_fixed_edge_and_the_side_come_from_where_the_gap_centre_moves(side):
    # a slit at 30 degrees: across (to the right of the long axis) is (cos a, -sin a)
    a = math.radians(30.0)
    across = (math.cos(a), -math.sin(a))
    edge = (500.0, 400.0)
    solution = FieldSolution(0.27, 11.0, 1, 250.0, 36.0, 100.0, "high", 50)

    def at(width):  # one jaw: the gap's centre is half the width from the fixed edge
        return gap(
            edge[0] + side * width / 2 * across[0], edge[1] + side * width / 2 * across[1], width
        )

    v = oc.combine(solution, at(40.0), at(20.0), binning=1, roi=(0, 0), rot_pa=0.0)
    assert v["slit_side"] == side
    assert (v["slit_x"], v["slit_y"]) == pytest.approx(edge, abs=1e-6)
    assert v["centre_shift_ratio"] == pytest.approx(1.0)
    words = " ".join(oc.judge({**v, "slit_width_arcsec": 40 * 0.27}, 40 * 0.27, 80.0))
    assert "jaw" not in words  # one moving jaw is what is expected


def test_a_gap_that_opens_symmetrically_is_flagged():
    solution = FieldSolution(0.27, 11.0, 1, 250.0, 36.0, 100.0, "high", 50)
    v = oc.combine(  # the centre does not move: both jaws move
        solution, gap(500.0, 400.0, 40.0), gap(500.0, 400.0, 20.0), binning=1, roi=(0, 0), rot_pa=0
    )
    assert v["centre_shift_ratio"] == pytest.approx(0.0)
    words = " ".join(oc.judge({**v, "slit_width_arcsec": 40 * 0.27}, 40 * 0.27, 80.0))
    assert "both jaws" in words


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
    done = finished(client, client.post("/telescope/slew", json=M13, headers=h).json()["id"])
    assert done["state"] == "done"


@pytest.mark.parametrize(("binning", "side"), [(1, 1), (2, 1), (1, -1)])
def test_the_job_recovers_the_cameras_true_optics(client, binning, side):
    h = hold_lease(client)
    on_target(client, h)
    camera = guide_camera.get_guide_camera()
    camera.optics = camera.optics.model_copy(update={"slit_side": side})
    o = camera.optics
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
    # the fixed jaw's edge, and which way the gap grows from it (not the gap's own middle)
    assert (m["slit_x"], m["slit_y"]) == pytest.approx((o.slit_x_px, o.slit_y_px), abs=1.5)
    assert m["slit_side"] == side
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
    assert values["slit_side"] == m["slit_side"]
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
    client.post("/cover/close", headers=h)
    blocked = client.post("/calibration/optics", json=QUICK, headers=h)
    assert blocked.status_code == 409 and "cover" in blocked.json()["message"]


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
