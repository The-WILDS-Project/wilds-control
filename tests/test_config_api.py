"""The /config endpoints: typed forms over the conf registry, with stale-version protection."""

import pytest
from fastapi.testclient import TestClient

from wilds.control import lease
from wilds.control.activities import registry
from wilds.control.app import create_app
from wilds.control.jobs import manager


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


URL = "/config/guide-camera/conf"


def test_index_lists_components_and_slices(client):
    body = client.get("/config").json()
    assert body["components"]["guide-camera"] == ["cal", "conf"]


def test_get_gives_values_schema_and_version(client):
    body = client.get(URL).json()
    assert body["version"] == 0
    assert isinstance(body["values"]["single_frame"]["gain"], int)
    assert {"single_frame", "video"} <= set(body["json_schema"]["properties"])


def test_unknown_component_and_slice_are_404(client):
    assert client.get("/config/nope/conf").status_code == 404
    assert client.get("/config/guide-camera/other").json()["code"] == "not_found"


def test_put_needs_the_lease(client):
    r = client.put(URL, json={"values": {"single_frame": {"gain": 150}}, "version": 0})
    assert r.status_code == 403


def test_put_saves_under_the_holder_and_bumps_version(client):
    h = hold_lease(client)
    r = client.put(
        URL,
        json={"values": {"single_frame": {"gain": 150}}, "version": 0, "reason": "test"},
        headers=h,
    )
    assert r.status_code == 200, r.text
    assert r.json()["version"] == 1 and r.json()["values"]["single_frame"]["gain"] == 150
    hist = client.get(f"{URL}/history").json()
    assert hist[0]["who"] == "Arie" and hist[0]["reason"] == "test"
    assert client.get("/guide-camera/settings").json()["single_frame"]["gain"] == 150


def test_stale_version_is_409_stale_and_changes_nothing(client):
    h = hold_lease(client)
    client.put(URL, json={"values": {"single_frame": {"gain": 150}}, "version": 0}, headers=h)
    r = client.put(URL, json={"values": {"single_frame": {"gain": 200}}, "version": 0}, headers=h)
    assert r.status_code == 409
    assert r.json()["code"] == "stale" and r.json()["details"]["version"] == 1
    assert client.get(URL).json()["values"]["single_frame"]["gain"] == 150


def test_out_of_limits_is_422(client):
    h = hold_lease(client)
    r = client.put(URL, json={"values": {"single_frame": {"gain": 99999}}, "version": 0}, headers=h)
    assert r.status_code == 422 and r.json()["code"] == "invalid"


def test_component_refusal_is_422_and_nothing_is_recorded(client):
    h = hold_lease(client)
    roi = {"x": 0, "y": 0, "width": 99999, "height": 99999}
    r = client.put(URL, json={"values": {"single_frame": {"roi": roi}}, "version": 0}, headers=h)
    assert r.status_code == 422 and r.json()["code"] == "invalid"
    assert client.get(URL).json()["version"] == 0


def test_revert_restores_an_earlier_version_as_a_new_one(client):
    h = hold_lease(client)
    client.put(URL, json={"values": {"single_frame": {"gain": 150}}, "version": 0}, headers=h)
    client.put(URL, json={"values": {"single_frame": {"gain": 200}}, "version": 1}, headers=h)
    r = client.post(f"{URL}/revert/1", headers=h)
    assert r.status_code == 200 and r.json()["values"]["single_frame"]["gain"] == 150
    assert r.json()["version"] == 3
    assert client.post(f"{URL}/revert/99", headers=h).status_code == 404


def test_calibration_slice_starts_uncalibrated(client):
    body = client.get("/config/guide-camera/cal").json()
    assert body["values"]["plate_scale_arcsec_px"] is None
    h = hold_lease(client)
    r = client.put(
        "/config/guide-camera/cal",
        json={"values": {"plate_scale_arcsec_px": 0.42}, "version": 0},
        headers=h,
    )
    assert r.status_code == 200 and r.json()["values"]["plate_scale_arcsec_px"] == 0.42


def test_status_carries_every_sections_version(client):
    h = hold_lease(client)
    assert client.get("/status").json()["conf_versions"]["guide-camera/conf"] == 0
    client.put(URL, json={"values": {"single_frame": {"gain": 150}}, "version": 0}, headers=h)
    versions = client.get("/status").json()["conf_versions"]
    assert versions["guide-camera/conf"] == 1 and versions["guide-camera/cal"] == 0
