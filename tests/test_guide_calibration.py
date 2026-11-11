"""The guide camera's dark and bias frames: the store, the job, and the HTTP side."""

import asyncio
import io
import time

import numpy as np
import pytest
from astropy.io import fits
from fastapi.testclient import TestClient

from wilds.control import guide_camera, lease, paths
from wilds.control.activities import registry
from wilds.control.app import create_app
from wilds.control.guide_camera import SimulatedASICamera
from wilds.control.guide_camera.base import CaptureSettings, Frame, FrameMeta
from wilds.control.guide_camera.calibration_frames import CalibrationStore, store
from wilds.control.jobs import manager

TO = {"X-TO-Passphrase": "to-pass"}
URL = "/guide-camera/calibration"


# -- the camera in the dark ---------------------------------------------------------------


def test_a_blocked_light_path_gives_a_flat_dark_frame_and_an_open_one_has_stars():
    settings = CaptureSettings(exposure_s=0.5)
    lit = SimulatedASICamera()
    dark = SimulatedASICamera(light_blocked=lambda: True)
    for cam in (lit, dark):
        asyncio.run(cam.startup())
    star = asyncio.run(lit.calibration_frame("dark", settings)).pixels  # light: nobody closed it
    flat = asyncio.run(dark.calibration_frame("dark", settings)).pixels
    assert star.max() > 3 * np.median(star)
    assert flat.max() < 1.5 * np.median(flat) + 50  # noise only, no stars


# -- the store ----------------------------------------------------------------------------


def frame(value: float, *, exposure=1.0, gain=100, temp=-10.0, roi=None, n=1) -> Frame:
    meta = FrameMeta(
        id=n,
        mode="dark",
        taken_ms=1000 + n,
        exposure_s=exposure,
        gain=gain,
        offset=10,
        binning=1,
        roi=roi,
        width=4,
        height=3,
        temperature_c=temp,
    )
    return Frame(meta=meta, jpeg=b"", pixels=np.full((3, 4), value, np.uint16))


@pytest.fixture
def cal(tmp_path) -> CalibrationStore:
    return CalibrationStore(tmp_path / "cal")


def test_sessions_are_time_stamped_and_one_is_active(cal):
    assert cal.sessions() == [] and cal.active_id() is None
    a = cal.new_session("first night")
    time.sleep(1.1)
    b = cal.new_session()
    assert a.id < b.id and cal.active_id() == b.id
    assert [s.id for s in cal.sessions()] == [b.id, a.id]
    assert [s.active for s in cal.sessions()] == [True, False]
    cal.activate(a.id)
    assert cal.active_id() == a.id and cal.sessions()[1].active is True
    assert cal.sessions()[1].label == "first night"


def test_two_sessions_in_one_second_get_distinct_ids(cal):
    ids = {cal.new_session().id for _ in range(3)}
    assert len(ids) == 3


def test_activating_an_unknown_session_is_404(cal):
    with pytest.raises(Exception, match="no calibration session"):
        cal.activate("nope")


def test_ensure_session_starts_one_only_when_there_is_none(cal):
    first = cal.ensure_session()
    assert cal.ensure_session() == first


def test_an_entry_has_its_frames_and_their_median(cal):
    sid = cal.new_session().id
    entry = cal.add(sid, "dark", [frame(10, n=1), frame(20, n=2), frame(90, n=3)], match="video")
    assert (entry.id, entry.count, entry.match) == ("dark-0001", 3, "video")
    master = fits.getdata(cal.path(sid, entry.id, "master.fits"))
    assert (
        master.dtype.kind == "f" and master.dtype.itemsize == 4 and (master == 20).all()
    )  # the median, not the mean
    header = fits.getheader(cal.path(sid, entry.id, "master.fits"))
    assert header["NCOMBINE"] == 3 and header["IMAGETYP"] == "Dark Frame"
    assert fits.getdata(cal.path(sid, entry.id, "frame-02.fits")).max() == 20
    assert cal.add(sid, "bias", [frame(5)], match="single_frame").id == "bias-0002"


def test_only_listed_files_are_served(cal):
    sid = cal.new_session().id
    entry = cal.add(sid, "dark", [frame(1)], match="single_frame")
    for bad in ("../index.json", "index.json", "frame-99.fits"):
        with pytest.raises(Exception, match="no file"):
            cal.path(sid, entry.id, bad)
    with pytest.raises(Exception, match="no file"):
        cal.path(sid, "dark-0009", "master.fits")


def test_find_master_matches_settings_and_takes_the_nearest_temperature(cal):
    sid = cal.new_session().id
    cal.add(sid, "dark", [frame(1, temp=-10.0)], match="single_frame")
    cal.add(sid, "dark", [frame(2, temp=-14.0, n=2)], match="single_frame")
    cal.add(sid, "dark", [frame(3, exposure=2.0, temp=-10.0, n=3)], match="single_frame")
    want = CaptureSettings(exposure_s=1.0, gain=100, offset=10)
    entry, path = cal.find_master("dark", want, temperature_c=-13.0)
    assert entry.id == "dark-0002" and path.name == "master.fits"
    assert cal.find_master("dark", want, temperature_c=-10.4)[0].id == "dark-0001"
    assert cal.find_master("dark", want, temperature_c=10.0) is None  # too far off
    assert cal.find_master("dark", want.model_copy(update={"gain": 200})) is None
    assert cal.find_master("dark", want.model_copy(update={"exposure_s": 3.0})) is None
    assert cal.find_master("bias", want) is None


def test_a_bias_matches_whatever_the_exposure(cal):
    sid = cal.new_session().id
    cal.add(sid, "bias", [frame(7, exposure=0.001)], match="single_frame")
    entry, _ = cal.find_master("bias", CaptureSettings(exposure_s=30.0, gain=100, offset=10))
    assert entry.kind == "bias"


def test_the_masters_of_another_session_are_not_offered(cal):
    old = cal.new_session().id
    cal.add(old, "dark", [frame(1)], match="single_frame")
    cal.new_session()
    assert cal.find_master("dark", CaptureSettings(exposure_s=1.0, gain=100, offset=10)) is None
    cal.activate(old)
    assert cal.find_master("dark", CaptureSettings(exposure_s=1.0, gain=100, offset=10))


# -- the job and HTTP ---------------------------------------------------------------------


@pytest.fixture
def client(config_file, passphrases_file):
    passphrases_file('[passphrases]\nto = "to-pass"\n')
    lease.state.sessions.clear()
    lease.state.holder_id = lease.state.token = None
    manager.reset()
    registry.clear()
    with TestClient(create_app()) as c:
        yield c
    manager.reset()
    registry.clear()


def hold_lease(client) -> dict[str, str]:
    s = client.post("/sessions", json={"name": "Arie"}).json()
    sid = {"X-Session-Id": s["session_id"]}
    client.post("/control/requests", headers=sid)
    client.post(f"/control/requests/{s['code']}/grant", headers=TO)
    token = client.post(f"/sessions/{s['session_id']}/heartbeat").json()["lease_token"]
    return {**sid, "X-Lease-Token": token}


def wait_for(predicate, seconds=15.0):
    deadline = time.time() + seconds
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


def run(client, h, **body) -> dict:
    body = {"require_stable_cooler": False, **body}
    r = client.post(f"{URL}/frames", json=body, headers=h)
    assert r.status_code == 202, r.text
    job = r.json()
    assert wait_for(
        lambda: client.get(f"/jobs/{job['id']}").json()["state"] not in ("queued", "running")
    )
    return client.get(f"/jobs/{job['id']}").json()


def set_group(client, h, group, **values):
    assert client.put("/guide-camera/settings", json={group: values}, headers=h).status_code == 200


def test_a_dark_run_files_frames_and_a_median_and_restores_the_shutter(client):
    h = hold_lease(client)
    set_group(client, h, "single_frame", exposure_s=0.02)
    assert client.post("/shutter/open", headers=h).json()["state"] == "open"
    job = run(client, h, kind="dark", count=3)
    assert job["state"] == "done", job
    assert job["result"]["count"] == 3 and job["result"]["entry"] == "dark-0001"
    assert client.get("/shutter").json()["state"] == "open"  # put back as found

    (session,) = client.get(f"{URL}/sessions").json()
    assert session["active"] is True and session["id"] == job["result"]["session"]
    (entry,) = session["entries"]
    assert (entry["kind"], entry["count"], entry["exposure_s"], entry["match"]) == (
        "dark",
        3,
        0.02,
        "single_frame",
    )
    raw = client.get(f"{URL}/sessions/{session['id']}/entries/dark-0001/master.fits")
    hdul = fits.open(io.BytesIO(raw.content))
    assert hdul[0].header["IMAGETYP"] == "Dark Frame" and hdul[0].header["NCOMBINE"] == 3
    assert hdul[0].header["EXPTIME"] == 0.02
    assert hdul[0].data.max() < 1.5 * np.median(hdul[0].data) + 50  # no stars: the shutter was shut
    assert paths.guide_camera_calibration().is_dir()


def test_a_bias_is_the_shortest_exposure_and_takes_no_exposure(client):
    h = hold_lease(client)
    job = run(client, h, kind="bias", count=2)
    assert job["state"] == "done", job
    (entry,) = client.get(f"{URL}/sessions").json()[0]["entries"]
    assert entry["kind"] == "bias" and entry["exposure_s"] == 0.001
    r = client.post(f"{URL}/frames", json={"kind": "bias", "exposure_s": 1.0}, headers=h)
    assert r.status_code == 422


def test_the_frames_follow_the_chosen_settings_group(client):
    h = hold_lease(client)
    set_group(client, h, "video", exposure_s=0.01, gain=250, binning=2)
    set_group(client, h, "single_frame", exposure_s=0.5)
    job = run(client, h, kind="dark", count=1, match="video")
    assert job["state"] == "done", job
    (entry,) = client.get(f"{URL}/sessions").json()[0]["entries"]
    assert (entry["gain"], entry["binning"], entry["exposure_s"], entry["match"]) == (
        250,
        2,
        0.01,
        "video",
    )
    job = run(client, h, kind="dark", count=1, match="video", exposure_s=0.03)
    newest, _ = client.get(f"{URL}/sessions").json()[0]["entries"][::-1]
    assert newest["exposure_s"] == 0.03


def test_the_store_finds_what_the_job_filed(client):
    h = hold_lease(client)
    set_group(client, h, "single_frame", exposure_s=0.02)
    run(client, h, kind="dark", count=2)
    camera_settings = guide_camera.get_guide_camera().settings.single_frame
    entry, path = store().find_master("dark", camera_settings)
    assert entry.id == "dark-0001" and path.is_file()


def test_a_dark_waits_for_a_stable_cooler_unless_told_not_to(client):
    h = hold_lease(client)
    r = client.post(f"{URL}/frames", json={"kind": "dark"}, headers=h)  # development: cooling
    assert r.status_code == 409 and r.json()["code"] == "cooler_not_stable"


def test_it_is_refused_while_streaming_or_not_started_up_or_without_the_lease(client):
    assert client.post(f"{URL}/frames", json={"kind": "dark"}).status_code == 403
    h = hold_lease(client)
    client.post("/guide-camera/start", headers=h)
    r = client.post(
        f"{URL}/frames", json={"kind": "dark", "require_stable_cooler": False}, headers=h
    )
    assert r.status_code == 409 and r.json()["code"] == "busy"
    client.post("/guide-camera/stop", headers=h)
    client.post("/system/shutter/shutdown", headers=TO)
    assert wait_for(lambda: client.get("/shutter").json()["lifecycle"]["opstate"] == "shutdown")
    r = client.post(
        f"{URL}/frames", json={"kind": "dark", "require_stable_cooler": False}, headers=h
    )
    assert r.status_code == 409 and r.json()["code"] == "not_operational"


def test_cancelling_a_run_ends_the_exposure_and_still_restores_the_shutter(client):
    h = hold_lease(client)
    set_group(client, h, "single_frame", exposure_s=20)
    client.post("/shutter/open", headers=h)
    r = client.post(
        f"{URL}/frames",
        json={"kind": "dark", "count": 5, "require_stable_cooler": False},
        headers=h,
    )
    job = r.json()
    assert wait_for(lambda: client.get("/shutter").json()["state"] == "closed")  # it is exposing
    client.post(f"/jobs/{job['id']}/cancel", headers=h)
    assert wait_for(lambda: client.get(f"/jobs/{job['id']}").json()["state"] == "cancelled")
    assert wait_for(lambda: client.get("/shutter").json()["state"] == "open")
    assert (
        client.get(f"{URL}/sessions").json() == []
        or not client.get(f"{URL}/sessions").json()[0]["entries"]
    )


def test_sessions_can_be_started_and_switched(client):
    h = hold_lease(client)
    assert client.post(f"{URL}/sessions", json={"label": "x"}).status_code == 403
    first = client.post(f"{URL}/sessions", json={"label": "before the move"}, headers=h).json()
    time.sleep(1.1)
    second = client.post(f"{URL}/sessions", headers=h).json()
    assert second["active"] and first["id"] < second["id"]
    back = client.post(f"{URL}/sessions/{first['id']}/activate", headers=h).json()
    assert back["active"] and back["label"] == "before the move"
    assert [s["active"] for s in client.get(f"{URL}/sessions").json()] == [False, True]
    assert client.post(f"{URL}/sessions/nope/activate", headers=h).status_code == 404


def test_a_calibration_frame_is_downloadable_while_the_run_goes_on(client):
    h = hold_lease(client)
    set_group(client, h, "single_frame", exposure_s=0.02)
    assert client.get("/guide-camera").json()["last_calibration_frame_id"] is None
    job = run(client, h, kind="dark", count=2)
    assert job["state"] == "done", job
    fid = client.get("/guide-camera").json()["last_calibration_frame_id"]
    assert fid is not None
    r = client.get(f"/guide-camera/frames/{fid}")
    assert r.status_code == 200
    header = fits.open(io.BytesIO(r.content))[0].header
    assert header["IMAGETYP"] == "Dark Frame" and header["EXPTIME"] == 0.02
    # it is not the live view's frame
    assert client.get("/guide-camera").json()["last_frame_id"] is None
    assert client.get("/status").json()["guide_camera"]["last_calibration_frame_id"] == fid


# -- discarding an entry ------------------------------------------------------------------


def test_a_discarded_entry_is_gone_from_the_disk_and_the_index(cal):
    sid = cal.new_session().id
    entry = cal.add(sid, "dark", [frame(1), frame(2, n=2)], match="single_frame")
    folder = cal.root / sid / entry.id
    assert folder.is_dir()
    info = cal.delete_entry(sid, entry.id)
    assert info.entries == [] and not folder.exists() and info.active is True
    assert cal.sessions()[0].entries == []
    assert cal.find_master("dark", CaptureSettings(exposure_s=1.0, gain=100, offset=10)) is None
    with pytest.raises(Exception, match="no file"):
        cal.path(sid, entry.id, "master.fits")


def test_discarding_an_unknown_entry_is_404(cal):
    sid = cal.new_session().id
    for session, entry in ((sid, "dark-0009"), ("nope", "dark-0001")):
        with pytest.raises(Exception, match="no entry"):
            cal.delete_entry(session, entry)


def test_a_new_entry_never_takes_the_number_of_one_still_there(cal):
    sid = cal.new_session().id
    first = cal.add(sid, "dark", [frame(1)], match="single_frame")
    second = cal.add(sid, "dark", [frame(2, n=2)], match="single_frame")
    cal.delete_entry(sid, first.id)
    third = cal.add(sid, "bias", [frame(3, n=3)], match="single_frame")
    assert (first.id, second.id, third.id) == ("dark-0001", "dark-0002", "bias-0003")
    assert (cal.root / sid / second.id / "master.fits").is_file()  # untouched


def test_discard_over_http_needs_the_lease_and_reports_the_session(client):
    h = hold_lease(client)
    set_group(client, h, "single_frame", exposure_s=0.02)
    run(client, h, kind="dark", count=1)
    run(client, h, kind="bias", count=1)
    (session,) = client.get(f"{URL}/sessions").json()
    url = f"{URL}/sessions/{session['id']}/entries/dark-0001"
    assert client.delete(url).status_code == 403
    r = client.delete(url, headers=h)
    assert r.status_code == 200 and [e["id"] for e in r.json()["entries"]] == ["bias-0002"]
    assert client.get(f"{url}/master.fits").status_code == 404
    assert client.delete(url, headers=h).status_code == 404
    assert (
        client.get(f"{URL}/sessions/{session['id']}/entries/bias-0002/master.fits").status_code
        == 200
    )
