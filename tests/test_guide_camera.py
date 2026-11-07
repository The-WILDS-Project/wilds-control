import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from wilds.control import guide_camera, lease
from wilds.control.activities import registry
from wilds.control.app import create_app
from wilds.control.guide_camera import (
    CoolerRequest,
    SettingError,
    SettingsUpdate,
    SimulatedASICamera,
)
from wilds.control.jobs import manager


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


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
    client.post(f"/control/requests/{s['code']}/grant", headers={"X-TO-Passphrase": "to-pass"})
    token = client.post(f"/sessions/{s['session_id']}/heartbeat").json()["lease_token"]
    return {**sid, "X-Lease-Token": token}


def wait_for(predicate, seconds=5.0):
    deadline = time.time() + seconds
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


def finished(client, job_id):
    return wait_for(
        lambda: client.get(f"/jobs/{job_id}").json()["state"] not in ("queued", "running")
    )


# -- settings -------------------------------------------------------------------------


def test_status_carries_the_camera_with_its_settings_and_cooler(client):
    cam = client.get("/status").json()["guide_camera"]
    assert cam["kind"] == "simulated" and cam["streaming"] is False
    assert cam["settings"]["binning"] == 1 and cam["settings"]["roi"] is None
    assert cam["cooler"]["supported"] is True and cam["cooler"]["state"] == "off"
    assert cam["cooler"]["temperature_c"] == 20.0


def test_a_partial_settings_update_changes_only_what_was_sent(client):
    h = hold_lease(client)
    r = client.put("/guide-camera/settings", json={"exposure_s": 2, "binning": 2}, headers=h)
    assert r.status_code == 200
    assert r.json()["exposure_s"] == 2 and r.json()["binning"] == 2 and r.json()["gain"] == 100
    assert client.get("/guide-camera/settings").json() == r.json()


@pytest.mark.parametrize(
    "body",
    [
        {"exposure_s": 0},
        {"exposure_s": 1000},
        {"gain": -1},
        {"binning": 3},
        {"nonsense": 1},
        {"roi": {"x": 1200, "y": 0, "width": 200, "height": 100}},  # off the sensor
        {"roi": {"x": 0, "y": 0, "width": 8, "height": 100}},  # too small
    ],
)
def test_bad_settings_are_refused_with_the_standard_error_and_change_nothing(client, body):
    h = hold_lease(client)
    before = client.get("/guide-camera/settings").json()
    r = client.put("/guide-camera/settings", json=body, headers=h)
    assert r.status_code == 422
    assert client.get("/guide-camera/settings").json() == before
    assert r.json()["code"] == "invalid" and r.json()["message"]


def test_roi_must_fit_the_binning(client):
    h = hold_lease(client)
    roi = {"x": 0, "y": 0, "width": 101, "height": 100}
    assert (
        client.put("/guide-camera/settings", json={"roi": roi, "binning": 2}, headers=h).status_code
        == 422
    )
    assert client.put("/guide-camera/settings", json={"roi": roi}, headers=h).status_code == 200


def test_changing_anything_needs_the_lease(client):
    assert client.put("/guide-camera/settings", json={"gain": 1}).status_code == 403
    assert client.put("/guide-camera/cooler", json={"enabled": True}).status_code == 403
    assert client.post("/guide-camera/start").status_code == 403
    assert client.post("/guide-camera/frames").status_code == 403


# -- cooler ---------------------------------------------------------------------------


def test_the_cooler_is_switched_on_with_a_setpoint_and_reported(client):
    h = hold_lease(client)
    r = client.put("/guide-camera/cooler", json={"enabled": True, "setpoint_c": -15}, headers=h)
    assert r.status_code == 200
    body = r.json()
    assert body["enabled"] is True and body["setpoint_c"] == -15 and body["state"] == "cooling"
    assert client.get("/status").json()["guide_camera"]["cooler"]["setpoint_c"] == -15


def test_switching_on_without_a_setpoint_uses_the_default_then_the_last_one(client):
    h = hold_lease(client)
    on = client.put("/guide-camera/cooler", json={"enabled": True}, headers=h).json()
    assert on["setpoint_c"] == -10.0
    client.put("/guide-camera/cooler", json={"setpoint_c": -20}, headers=h)
    client.put("/guide-camera/cooler", json={"enabled": False}, headers=h)
    again = client.put("/guide-camera/cooler", json={"enabled": True}, headers=h).json()
    assert again["setpoint_c"] == -20 and again["enabled"] is True


@pytest.mark.parametrize("setpoint", [-41, 31])
def test_a_setpoint_outside_the_range_is_refused(client, setpoint):
    h = hold_lease(client)
    r = client.put("/guide-camera/cooler", json={"setpoint_c": setpoint}, headers=h)
    assert r.status_code == 422 and r.json()["code"] == "invalid"
    assert client.get("/guide-camera/cooler").json()["enabled"] is False


def test_the_sensor_follows_the_setpoint_and_settles_within_the_tolerance():
    clock = Clock()
    cam = SimulatedASICamera(clock=clock)
    cam.set_cooler(CoolerRequest(enabled=True, setpoint_c=-10))
    assert cam.cooler().state == "cooling"
    clock.now += 45  # one time constant: about 63 % of the way
    mid = cam.cooler()
    assert -2 < mid.temperature_c < 5 and mid.state == "cooling" and mid.power_pct > 0
    clock.now += 600
    end = cam.cooler()
    assert end.state == "stable" and abs(end.temperature_c + 10) <= 1.0


def test_the_sensor_cannot_go_further_below_ambient_than_the_cooler_can():
    clock = Clock()
    cam = SimulatedASICamera(clock=clock, ambient_c=20.0)
    cam.set_cooler(CoolerRequest(enabled=True, setpoint_c=-40))
    clock.now += 3000
    assert cam.cooler().temperature_c == pytest.approx(-15.0, abs=0.2)
    assert cam.cooler().state == "cooling"  # never reaches a setpoint it cannot hold


def test_switched_off_the_sensor_drifts_back_to_ambient():
    clock = Clock()
    cam = SimulatedASICamera(clock=clock)
    cam.set_cooler(CoolerRequest(enabled=True, setpoint_c=-10))
    clock.now += 600
    cam.set_cooler(CoolerRequest(enabled=False))
    assert cam.cooler().state == "off" and cam.cooler().power_pct == 0.0
    clock.now += 600
    assert cam.cooler().temperature_c == pytest.approx(20.0, abs=0.2)


def test_the_hardware_is_read_at_most_once_a_second():
    clock = Clock()
    cam = SimulatedASICamera(clock=clock)
    reads = []
    real = cam._read_temperature
    cam._read_temperature = lambda: reads.append(1) or real()
    for _ in range(10):
        cam.cooler()
    assert len(reads) == 1
    clock.now += 1.5
    cam.cooler()
    assert len(reads) == 2


def test_a_colder_sensor_gives_a_cleaner_dark_frame():
    clock = Clock()
    cam = SimulatedASICamera(clock=clock)
    cam.update_settings(SettingsUpdate(exposure_s=0.01, gain=300))
    warm = cam._expose(cam.settings).mean()
    cam.set_cooler(CoolerRequest(enabled=True, setpoint_c=-15))
    clock.now += 600
    cold = cam._expose(cam.settings).mean()
    assert cold <= warm


def test_a_camera_without_a_cooler_refuses_cooler_requests():
    class NoCooler(SimulatedASICamera):
        cooler_supported = False

    cam = NoCooler()
    assert cam.cooler().supported is False and cam.cooler().state == "unsupported"
    assert cam.cooler().temperature_c == 20.0
    with pytest.raises(SettingError):
        cam.set_cooler(CoolerRequest(enabled=True))


# -- frames ---------------------------------------------------------------------------


def test_a_single_frame_is_a_job_that_ends_with_a_frame_id_and_downloads_as_fits(client):
    h = hold_lease(client)
    client.put("/guide-camera/settings", json={"exposure_s": 0.05}, headers=h)
    r = client.post("/guide-camera/frames", headers=h)
    assert r.status_code == 202 and r.json()["components"] == ["guide-camera"]
    job_id = r.json()["id"]
    assert finished(client, job_id)
    job = client.get(f"/jobs/{job_id}").json()
    assert job["state"] == "done"
    fid = job["result"]["frame_id"]
    fits = client.get(f"/guide-camera/frames/{fid}")
    assert fits.status_code == 200 and fits.content.startswith(b"SIMPLE")
    assert client.get("/status").json()["guide_camera"]["last_frame_id"] == fid
    assert client.get("/guide-camera/frames/999").json()["code"] == "not_found"


def test_roi_and_binning_set_the_frame_size(client):
    h = hold_lease(client)
    roi = {"x": 100, "y": 100, "width": 400, "height": 200}
    client.put(
        "/guide-camera/settings", json={"exposure_s": 0.01, "roi": roi, "binning": 2}, headers=h
    )
    job = client.post("/guide-camera/frames", headers=h).json()
    assert finished(client, job["id"])
    fid = client.get(f"/jobs/{job['id']}").json()["result"]["frame_id"]
    camera = guide_camera.get_guide_camera()
    meta = camera.frame(fid).meta
    assert (meta.width, meta.height, meta.binning) == (200, 100, 2)


def test_a_frame_has_stars_and_noise_not_a_flat_field():
    cam = SimulatedASICamera()
    img = cam._expose(cam.settings.model_copy(update={"exposure_s": 0.5}))
    assert img.dtype == np.uint16 and img.shape == (882, 1296)
    assert img.max() > 3 * np.median(img)


def test_cancelling_a_long_exposure_ends_it_at_once(client):
    h = hold_lease(client)
    client.put("/guide-camera/settings", json={"exposure_s": 30}, headers=h)
    job = client.post("/guide-camera/frames", headers=h).json()
    assert wait_for(
        lambda: (
            "Exposing"
            in client.get("/status")
            .json()["activities"]
            .get("guide-camera", [{}])[0]
            .get("name", "")
        )
    )
    assert client.post(f"/jobs/{job['id']}/cancel", headers=h).status_code == 200
    assert finished(client, job["id"])
    assert client.get(f"/jobs/{job['id']}").json()["state"] == "cancelled"
    assert wait_for(lambda: client.get("/status").json()["activities"] == {})


# -- streaming ------------------------------------------------------------------------


def test_streaming_raises_its_activity_and_excludes_single_frames(client):
    h = hold_lease(client)
    client.put("/guide-camera/settings", json={"exposure_s": 0.02}, headers=h)
    assert client.post("/guide-camera/start", headers=h).json()["streaming"] is True
    assert client.post("/guide-camera/start", headers=h).status_code == 200  # idempotent
    names = {a["name"] for a in client.get("/status").json()["activities"]["guide-camera"]}
    assert "Streaming" in names
    r = client.post("/guide-camera/frames", headers=h)
    assert r.status_code == 409 and r.json()["code"] == "busy"
    assert wait_for(lambda: client.get("/guide-camera").json()["last_frame_id"] is not None)
    assert client.post("/guide-camera/stop", headers=h).json()["streaming"] is False
    assert client.get("/status").json()["activities"] == {}


def test_viewers_get_metadata_then_a_jpeg_per_frame(client):
    h = hold_lease(client)
    client.put("/guide-camera/settings", json={"exposure_s": 0.02}, headers=h)
    client.post("/guide-camera/start", headers=h)
    try:
        with client.websocket_connect("/guide-camera/stream") as ws:
            meta = ws.receive_json()
            jpeg = ws.receive_bytes()
            assert meta["width"] == 1296 and meta["temperature_c"] == 20.0
            assert jpeg[:2] == b"\xff\xd8"
            newer = ws.receive_json()
            assert ws.receive_bytes()[:2] == b"\xff\xd8" and newer["id"] > meta["id"]
    finally:
        client.post("/guide-camera/stop", headers=h)


def test_a_subscriber_holds_only_the_newest_frame():
    cam = SimulatedASICamera()
    import asyncio

    async def go():
        q = cam.subscribe()
        cam.update_settings(SettingsUpdate(exposure_s=0.01))
        ids = [(await cam.single_frame()).meta.id for _ in range(3)]
        assert q.qsize() == 1 and (await q.get()).meta.id == ids[-1]

    asyncio.run(go())


def test_config_picks_the_implementation(client, config_file):
    assert client.get("/guide-camera").json()["kind"] == "simulated"
    config_file('[guide_camera]\nkind = "asi"\n')
    guide_camera.reset()
    with pytest.raises(NotImplementedError):
        guide_camera.get_guide_camera()
