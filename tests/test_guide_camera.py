import asyncio
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from wilds.control import guide_camera, lease
from wilds.control.activities import registry
from wilds.control.app import create_app
from wilds.control.guide_camera import (
    GuideCameraConf,
    SettingError,
    SimulatedASICamera,
)
from wilds.control.jobs import StartedBy, manager

ARIE = StartedBy(name="Arie", code="K7-42")


def up(cam) -> None:
    """Start a directly built camera up (the app does this at boot in development)."""
    asyncio.run(cam.startup())


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
    assert (
        cam["settings"]["single_frame"]["binning"] == 1 and cam["settings"]["video"]["roi"] is None
    )
    assert cam["cooler"]["supported"] is True
    assert cam["conf_version"] == 0 and cam["cal_version"] == 0  # nothing saved: the defaults
    # In development the cooler starts on (the default); production is off until decided.
    assert cam["settings"]["cooler_enabled"] is True and cam["cooler"]["state"] == "cooling"
    assert cam["cooler"]["setpoint_c"] == -10.0


def test_a_partial_settings_update_changes_only_what_was_sent(client):
    h = hold_lease(client)
    r = client.put(
        "/guide-camera/settings", json={"single_frame": {"exposure_s": 2, "binning": 2}}, headers=h
    )
    assert r.status_code == 200
    one, video = r.json()["single_frame"], r.json()["video"]
    assert one["exposure_s"] == 2 and one["binning"] == 2 and one["gain"] == 100
    assert video == GuideCameraConf().video.model_dump()  # the other group is untouched
    assert client.get("/guide-camera/settings").json() == r.json()


@pytest.mark.parametrize(
    "body",
    [
        {"single_frame": {"exposure_s": 0}},
        {"single_frame": {"exposure_s": 1000}},
        {"single_frame": {"gain": -1}},
        {"single_frame": {"binning": 3}},
        {"nonsense": 1},
        {"single_frame": {"nonsense": 1}},
        {"single_frame": {"roi": {"x": 1200, "y": 0, "width": 200, "height": 100}}},  # off it
        {"video": {"roi": {"x": 1200, "y": 0, "width": 200, "height": 100}}},
        {"single_frame": {"roi": {"x": 0, "y": 0, "width": 8, "height": 100}}},  # too small
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
        client.put(
            "/guide-camera/settings", json={"single_frame": {"roi": roi, "binning": 2}}, headers=h
        ).status_code
        == 422
    )
    assert (
        client.put(
            "/guide-camera/settings", json={"single_frame": {"roi": roi}}, headers=h
        ).status_code
        == 200
    )


def test_changing_anything_needs_the_lease(client):
    assert (
        client.put("/guide-camera/settings", json={"single_frame": {"gain": 1}}).status_code == 403
    )
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
    cooler = client.get("/guide-camera/cooler").json()
    assert cooler["setpoint_c"] == -10.0 and cooler["enabled"] is True  # untouched
    assert guide_camera.get_guide_camera().conf.version == 0  # and nothing was recorded


def test_the_sensor_follows_the_setpoint_and_settles_within_the_tolerance():
    clock = Clock()
    cam = SimulatedASICamera(clock=clock)
    up(cam)  # startup puts the settings on the hardware: the default is on, at -10 °C
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
    up(cam)
    cam.conf.patch({"cooler_enabled": True, "cooler_setpoint_c": -40}, who=ARIE)
    clock.now += 3000
    assert cam.cooler().temperature_c == pytest.approx(-15.0, abs=0.2)
    assert cam.cooler().state == "cooling"  # never reaches a setpoint it cannot hold


def test_switched_off_the_sensor_drifts_back_to_ambient():
    clock = Clock()
    cam = SimulatedASICamera(clock=clock)
    up(cam)
    clock.now += 600
    cam.conf.patch({"cooler_enabled": False}, who=ARIE)
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
    up(cam)
    cam.conf.patch({"single_frame": {"exposure_s": 0.5, "gain": 300}}, who=ARIE)
    warm = cam._expose(cam.settings.single_frame).mean()
    cam.conf.patch({"cooler_enabled": True, "cooler_setpoint_c": -15}, who=ARIE)
    clock.now += 600
    cold = cam._expose(cam.settings.single_frame).mean()
    assert cold < warm - 1  # about 6 counts at this exposure and gain, noise is ~0.05


def test_a_camera_without_a_cooler_refuses_cooler_requests():
    class NoCooler(SimulatedASICamera):
        cooler_supported = False

    cam = NoCooler()
    assert cam.cooler().supported is False and cam.cooler().state == "unsupported"
    assert cam.cooler().temperature_c == 20.0
    with pytest.raises(SettingError):
        cam.conf.patch({"cooler_setpoint_c": -20}, who=ARIE)
    assert cam.conf.version == 0


# -- frames ---------------------------------------------------------------------------


def test_a_single_frame_is_a_job_that_ends_with_a_frame_id_and_downloads_as_fits(client):
    h = hold_lease(client)
    client.put("/guide-camera/settings", json={"single_frame": {"exposure_s": 0.05}}, headers=h)
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
        "/guide-camera/settings",
        json={"single_frame": {"exposure_s": 0.01, "roi": roi, "binning": 2}},
        headers=h,
    )
    job = client.post("/guide-camera/frames", headers=h).json()
    assert finished(client, job["id"])
    fid = client.get(f"/jobs/{job['id']}").json()["result"]["frame_id"]
    camera = guide_camera.get_guide_camera()
    meta = camera.frame(fid).meta
    assert (meta.width, meta.height, meta.binning) == (200, 100, 2)


def test_a_frame_has_stars_and_noise_not_a_flat_field():
    cam = SimulatedASICamera()
    img = cam._expose(cam.settings.single_frame.model_copy(update={"exposure_s": 0.5}))
    assert img.dtype == np.uint16 and img.shape == (882, 1296)
    assert img.max() > 3 * np.median(img)


def test_cancelling_a_long_exposure_ends_it_at_once(client):
    h = hold_lease(client)
    client.put("/guide-camera/settings", json={"single_frame": {"exposure_s": 30}}, headers=h)
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
    client.put("/guide-camera/settings", json={"single_frame": {"exposure_s": 0.02}}, headers=h)
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
    client.put("/guide-camera/settings", json={"single_frame": {"exposure_s": 0.02}}, headers=h)
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

    async def go():
        await cam.startup()
        q = cam.subscribe()
        cam.conf.patch({"single_frame": {"exposure_s": 0.01}}, who=ARIE)
        ids = [(await cam.single_frame()).meta.id for _ in range(3)]
        assert q.qsize() == 1 and (await q.get()).meta.id == ids[-1]

    asyncio.run(go())


def test_config_picks_the_implementation(client, config_file):
    assert client.get("/guide-camera").json()["kind"] == "simulated"
    config_file('[guide_camera]\nkind = "asi"\n')
    guide_camera.reset()
    with pytest.raises(NotImplementedError):
        guide_camera.get_guide_camera()


# -- the simulated sky ------------------------------------------------------------------


def still_camera(**kw) -> SimulatedASICamera:
    """No jitter and no drift: the only motion is what a test asks for."""
    cam = SimulatedASICamera(jitter_px=0.0, drift_x_px_s=0.0, drift_y_px_s=0.0, **kw)
    cam.conf.patch({"single_frame": {"exposure_s": 0.001}}, who=ARIE)
    return cam


def centroid(electrons: np.ndarray) -> tuple[float, float]:
    stars = np.clip(electrons - np.median(electrons), 0, None)
    ys, xs = np.indices(stars.shape)
    return float((stars * xs).sum() / stars.sum()), float((stars * ys).sum() / stars.sum())


def test_a_telescope_offset_moves_the_stars_by_that_much():
    cam = still_camera()
    x0, y0 = centroid(cam._electrons(cam.settings.single_frame))
    cam.nudge(6.0, -4.0)
    x1, y1 = centroid(cam._electrons(cam.settings.single_frame))
    assert x1 - x0 == pytest.approx(6.0, abs=0.3) and y1 - y0 == pytest.approx(-4.0, abs=0.3)


def test_the_field_drifts_with_time():
    clock = Clock()
    cam = still_camera(clock=clock)
    cam.drift_x_px_s = 0.5
    x0, _ = centroid(cam._electrons(cam.settings.single_frame))
    clock.now += 10
    x1, _ = centroid(cam._electrons(cam.settings.single_frame))
    assert x1 - x0 == pytest.approx(5.0, abs=0.3)


def test_image_motion_jitters_the_field_from_frame_to_frame_by_about_jitter_px():
    cam = still_camera()
    cam.jitter_px = 0.5
    xs = [centroid(cam._electrons(cam.settings.single_frame))[0] for _ in range(60)]
    assert 0.25 < np.std(xs) < 0.9


def test_defocus_spreads_the_light_without_losing_it():
    cam = still_camera()
    cam.seeing_fwhm_px = 4.0
    sharp = cam._electrons(cam.settings.single_frame)
    cam.defocus_px = 6.0
    soft = cam._electrons(cam.settings.single_frame)
    assert soft.max() < 0.5 * sharp.max()
    assert soft.sum() == pytest.approx(sharp.sum(), rel=0.05)


def test_the_seeing_changes_a_little_from_frame_to_frame():
    cam = still_camera()
    peaks = [cam._electrons(cam.settings.single_frame).max() for _ in range(30)]
    assert np.std(peaks) / np.mean(peaks) > 0.02


def test_a_frame_takes_its_exposure_time_and_no_less():
    cam = SimulatedASICamera()
    cam.conf.patch({"single_frame": {"exposure_s": 0.3}}, who=ARIE)
    start = time.monotonic()
    cam._expose(cam.settings.single_frame)
    assert 0.3 <= time.monotonic() - start < 0.6


def test_the_shortest_exposures_give_video_rates():
    cam = SimulatedASICamera()
    cam.conf.patch({"single_frame": {"exposure_s": 0.001}}, who=ARIE)
    cam._expose(cam.settings.single_frame)
    start = time.monotonic()
    for _ in range(10):
        cam._make_frame(
            cam._expose(cam.settings.single_frame), cam.settings.single_frame, "single_frame"
        )
    assert (time.monotonic() - start) / 10 < 0.1  # better than 10 fps, even on a slow machine


def test_binning_sums_the_light_and_a_small_roi_draws_less():
    cam = still_camera()
    whole = cam._electrons(cam.settings.single_frame)
    cam.conf.patch({"single_frame": {"binning": 2}}, who=ARIE)
    binned = cam._electrons(cam.settings.single_frame)
    assert binned.shape == (441, 648) and binned.sum() == pytest.approx(whole.sum(), rel=0.02)


# -- the settings are kept, and the camera starts with them ---------------------------------


def test_a_settings_change_is_recorded_under_the_holders_name_and_survives_a_restart(client):
    h = hold_lease(client)
    assert (
        client.put(
            "/guide-camera/settings", json={"single_frame": {"gain": 250}}, headers=h
        ).status_code
        == 200
    )
    conf = guide_camera.get_guide_camera().conf
    assert conf.version == 1
    entry = conf.history()[0]
    assert (
        entry.who == "Arie"
        and "single_frame" in entry.changed
        and entry.values["single_frame"]["gain"] == 250
    )
    assert client.get("/status").json()["guide_camera"]["conf_version"] == 1

    guide_camera.reset()  # a restart: a new camera loads what the store holds
    assert guide_camera.get_guide_camera().conf.current.single_frame.gain == 250
    assert client.get("/guide-camera/settings").json()["single_frame"]["gain"] == 250


def test_the_cooler_choice_is_kept_and_put_back_on_the_hardware_at_startup(client):
    h = hold_lease(client)
    client.put("/guide-camera/cooler", json={"enabled": False, "setpoint_c": -15}, headers=h)
    guide_camera.reset()
    cam = guide_camera.get_guide_camera()
    assert cam.conf.current.cooler_enabled is False and cam.conf.current.cooler_setpoint_c == -15
    assert cam.cooler().state == "off" and cam.cooler().setpoint_c == -15


def test_in_production_the_cooler_starts_off_until_someone_switches_it_on(config_file):
    config_file('opmode = "ldt-production"\n[broker]\nhost = "joe"\n')
    cam = guide_camera.get_guide_camera()
    assert cam.conf.current.cooler_enabled is False and cam.cooler().state == "off"


def test_a_window_outside_the_sensor_is_refused_before_anything_is_recorded(client):
    h = hold_lease(client)
    roi = {"x": 1200, "y": 0, "width": 200, "height": 100}
    assert client.put("/guide-camera/settings", json={"roi": roi}, headers=h).status_code == 422
    assert guide_camera.get_guide_camera().conf.version == 0


def test_a_cooler_that_fails_to_switch_is_not_recorded_as_switched(client, monkeypatch):
    h = hold_lease(client)
    cam = guide_camera.get_guide_camera()

    def broken(enabled, setpoint_c):
        raise OSError("the camera does not answer")

    monkeypatch.setattr(cam, "_write_cooler", broken)
    with pytest.raises(OSError, match="does not answer"):
        client.put("/guide-camera/cooler", json={"enabled": False}, headers=h)
    assert cam.conf.version == 0 and cam.conf.current.cooler_enabled is True  # store == device


def test_calibration_values_start_unset_and_a_procedure_that_needs_one_is_refused(client):
    cam = guide_camera.get_guide_camera()
    assert cam.cal.current.plate_scale_arcsec_px is None and cam.cal.version == 0
    from wilds.control.errors import ApiError

    with pytest.raises(ApiError) as e:
        cam.cal.require("plate_scale_arcsec_px")
    assert e.value.body.code == "not_calibrated" and e.value.status == 409
    cam.cal.save(
        type(cam.cal.current)(plate_scale_arcsec_px=0.304),
        who=StartedBy(name="Noa", code="B2-07"),
        reason="astrometry on M13",
    )
    assert cam.cal.require("plate_scale_arcsec_px") == 0.304
    assert client.get("/status").json()["guide_camera"]["cal_version"] == 1


def test_the_handles_are_in_the_registry_for_the_config_endpoints(client):
    from wilds.control.conf import confs

    assert confs.components() == ["guide-camera", "shutter", "slit", "telescope"]
    assert confs.slices("guide-camera") == [
        "cal",
        "conf",
    ]
    assert confs.get("guide-camera").schema()["properties"]["cooler_setpoint_c"]["unit"] == "°C"


def test_the_models_describe_their_fields_for_the_gui(client):
    schema = guide_camera.GuideCameraConf.model_json_schema()
    props = schema["properties"]
    assert props["single_frame"]["title"] == "Single frame" and props["video"]["title"] == "Video"
    group = schema["$defs"]["CaptureSettings"]["properties"]
    assert group["exposure_s"]["unit"] == "s" and group["exposure_s"]["minimum"] == 0.001
    assert (
        props["cooler_setpoint_c"]["minimum"] == -40 and props["cooler_setpoint_c"]["maximum"] == 30
    )
    assert props["cooler_enabled"]["title"] == "Cooler"


# -- single frames and video have their own settings ----------------------------------------


def test_video_and_single_frames_take_their_own_settings(client):
    h = hold_lease(client)
    body = {"single_frame": {"exposure_s": 0.05, "gain": 200}, "video": {"exposure_s": 0.01}}
    assert client.put("/guide-camera/settings", json=body, headers=h).status_code == 200
    cam = guide_camera.get_guide_camera()
    s = cam.settings
    assert (s.single_frame.exposure_s, s.single_frame.gain) == (0.05, 200)
    assert (s.video.exposure_s, s.video.gain) == (0.01, 100)

    job = client.post("/guide-camera/frames", headers=h).json()
    assert finished(client, job["id"])
    one = cam.frame(client.get(f"/jobs/{job['id']}").json()["result"]["frame_id"]).meta
    assert (one.mode, one.exposure_s, one.gain) == ("single_frame", 0.05, 200)

    client.post("/guide-camera/start", headers=h)
    try:
        deadline = time.monotonic() + 5
        while (cam.video_frame is None or cam.video_frame.meta.mode != "video") and (
            time.monotonic() < deadline
        ):
            time.sleep(0.02)
        live = cam.video_frame.meta
    finally:
        client.post("/guide-camera/stop", headers=h)
    assert (live.mode, live.exposure_s, live.gain) == ("video", 0.01, 100)


def test_each_group_has_its_own_roi_and_binning(client):
    h = hold_lease(client)
    roi = {"x": 0, "y": 0, "width": 400, "height": 200}
    body = {"video": {"roi": roi, "binning": 2}}
    assert client.put("/guide-camera/settings", json=body, headers=h).status_code == 200
    s = guide_camera.get_guide_camera().settings
    assert s.video.binning == 2 and s.video.roi.width == 400
    assert s.single_frame.binning == 1 and s.single_frame.roi is None
    # one group's ROI is checked against that group's own binning
    bad = {"video": {"roi": {"x": 0, "y": 0, "width": 101, "height": 100}}}
    assert client.put("/guide-camera/settings", json=bad, headers=h).status_code == 422


def test_the_simulated_camera_seeds_a_slit_only_when_asked_and_never_over_a_value(
    config_file, client
):
    from wilds.control.guide_camera import _seed_simulated_calibration

    camera = guide_camera.get_guide_camera()
    assert camera.cal.current.plate_scale_arcsec_px is None  # not by default
    _seed_simulated_calibration(camera)
    cal = camera.cal.current
    assert cal.plate_scale_arcsec_px == 0.25 and cal.slit_length_arcsec == 30.0
    assert (cal.slit_pixel_x, cal.slit_pixel_y) == (648.0, 441.0) and cal.parity == 1
    camera.cal.patch({"plate_scale_arcsec_px": 0.3}, who=ARIE)
    _seed_simulated_calibration(camera)  # already set: left alone
    assert camera.cal.current.plate_scale_arcsec_px == 0.3


# -- the simulated field follows the simulated telescope ---------------------------------------


def _brightest(camera) -> tuple[int, int]:
    image = camera._electrons(guide_camera.base.CaptureSettings(exposure_s=0.5))
    y, x = np.unravel_index(np.argmax(image), image.shape)
    return int(x), int(y)


def _still_camera(**kwargs):
    camera = SimulatedASICamera(jitter_px=0.0, drift_x_px_s=0.0, drift_y_px_s=0.0, **kwargs)
    camera.seeing_fwhm_px = 3.0
    camera._rng = np.random.default_rng(7)  # the same seeing every time: no two stars swap places
    return camera


def test_the_simulated_field_shifts_and_rotates_as_the_hook_says():
    x0, y0 = _brightest(_still_camera())
    shifted = _brightest(_still_camera(field=lambda: (30.0, -12.0, 0.0, 0.0, 0.0)))
    assert abs(shifted[0] - (x0 + 30)) <= 1 and abs(shifted[1] - (y0 - 12)) <= 1
    cx, cy = 648.0, 441.0
    turned = _brightest(_still_camera(field=lambda: (0.0, 0.0, 90.0, cx, cy)))
    # counter-clockwise on the screen with y down: (rx, ry) -> (ry, -rx)
    expect = (cx + (y0 - cy), cy - (x0 - cx))
    assert abs(turned[0] - expect[0]) <= 1 and abs(turned[1] - expect[1]) <= 1


def test_the_telescopes_offsets_and_rotator_move_the_simulated_camera_field(client):
    import math

    from wilds.control import telescope
    from wilds.control.guide_camera import simulated_field
    from wilds.control.guide_camera.optics import SimulatedOptics
    from wilds.control.telescope import Offset, Target

    camera = guide_camera.get_guide_camera()
    camera.optics = SimulatedOptics(  # 0.25"/px, north 25 degrees CCW, parity +1, slit mid-frame
        plate_scale_arcsec_px=0.25, orientation_deg=25.0, slit_x_px=648.0, slit_y_px=441.0
    )
    tel = telescope.get_telescope()
    assert simulated_field(camera)[:3] == (0.0, 0.0, 0.0)  # nothing moved yet
    tel.set_preview(False)
    tel._send_offset("Handset", Offset(d1_arcsec=5.0, d2_arcsec=0.0), "TPLANE")
    dx, dy, rot, cx, cy = simulated_field(camera)
    # 5" east moves the stars west by 20 px: against the direction of east in the image
    a = math.radians(25.0)  # north is 25 degrees counter-clockwise from up
    east = (-math.cos(a), math.sin(a))
    assert (dx, dy) == pytest.approx((-20 * east[0], -20 * east[1]))
    assert rot == 0.0 and (cx, cy) == (648.0, 441.0)
    tel._send_target(Target(name="x", ra_hours=3.0, dec_deg=20.0), 40.0, "Target")
    later = tel.clock() + 10_000.0
    tel.clock = lambda: later  # long after: the rotator has arrived
    assert simulated_field(camera)[2] == pytest.approx(-40.0)
