"""The telescope: targets, offsets and the rotator, sent with nobody to approve them (plan §3.12)."""

import time

import pytest
from fastapi.testclient import TestClient

from wilds.control import lease, telescope
from wilds.control.activities import registry
from wilds.control.app import create_app
from wilds.control.jobs import manager

TO = {"X-TO-Passphrase": "to-pass"}
TARGET = {"name": "HD 12345", "ra_hours": 3.5, "dec_deg": 21.25, "rot_pa": 40.0}


@pytest.fixture
def client(config_file, passphrases_file):
    passphrases_file('[passphrases]\nto = "to-pass"\nengineering = "eng-pass"\n')
    lease.state.sessions.clear()
    lease.state.holder_id = lease.state.token = None
    lease.state.engineering = False
    manager.reset()
    registry.clear()
    with TestClient(create_app()) as c:
        sim = telescope.get_telescope()
        sim.slew_s, sim.rotate_deg_s, sim.offset_s = 0.2, 400.0, 0.1  # quick
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


def wait_for(predicate, seconds=10.0):
    deadline = time.time() + seconds
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.03)
    return False


def job(client, id) -> dict:
    return client.get(f"/jobs/{id}").json()


def finished(client, id) -> dict:
    assert wait_for(lambda: job(client, id)["state"] not in ("queued", "running"))
    return job(client, id)


def test_the_status_carries_the_telescope_and_feeds_the_tcs_pill(client):
    body = client.get("/status").json()
    assert body["telescope"]["kind"] == "simulated"
    assert body["telescope"]["telemetry"]["accepts_commands"] is True
    assert body["tcs"]["stale"] is False and body["tcs"]["acceptsCommands"] is True
    assert body["tcs"]["state"] == "ENABLED" and body["tcs"]["health"] == "GOOD"
    assert body["actions"]["slew"]["ok"] is True
    no_target = {"ok": False, "why": "No target yet: slew to one first"}
    assert body["actions"]["rotator"] == no_target and body["actions"]["offset"] == no_target


def test_a_target_is_sent_and_the_telescope_arrives(client):
    h = hold_lease(client)
    r = client.post("/telescope/slew", json=TARGET, headers=h)
    assert r.status_code == 202, r.text
    done = finished(client, r.json()["id"])
    assert done["state"] == "done" and done["result"] == {"target": "HD 12345"}
    t = client.get("/telescope").json()["telemetry"]
    assert t["target"]["name"] == "HD 12345" and t["in_position"] is True
    assert t["rot_pa"] == pytest.approx(40.0)


def test_the_rotator_needs_a_target(client):
    h = hold_lease(client)
    r = client.post("/telescope/rotator", json={"pa": 90}, headers=h)
    assert r.status_code == 409 and r.json()["code"] == "conflict"
    finished(client, client.post("/telescope/slew", json=TARGET, headers=h).json()["id"])
    j = client.post("/telescope/rotator", json={"pa": 370.0}, headers=h).json()  # 10 degrees
    done = finished(client, j["id"])
    assert done["state"] == "done" and done["result"] == {"pa": 10.0}
    assert client.get("/telescope").json()["telemetry"]["rot_pa"] == pytest.approx(10.0)


def test_offsets_are_set_cleared_and_limited(client):
    h = hold_lease(client)
    finished(client, client.post("/telescope/slew", json=TARGET, headers=h).json()["id"])
    j = client.post(
        "/telescope/offset", json={"d1_arcsec": 3.5, "d2_arcsec": -2.0}, headers=h
    ).json()
    assert finished(client, j["id"])["state"] == "done"
    assert client.get("/telescope").json()["telemetry"]["offsets"]["Handset"] == {
        "d1_arcsec": 3.5,
        "d2_arcsec": -2.0,
    }
    r = client.post("/telescope/offset", json={"d1_arcsec": 500, "d2_arcsec": 0}, headers=h)
    assert r.status_code == 422 and r.json()["code"] == "invalid"
    j = client.post("/telescope/offset/clear", headers=h).json()
    assert finished(client, j["id"])["state"] == "done"
    assert client.get("/telescope").json()["telemetry"]["offsets"]["Handset"] == {
        "d1_arcsec": 0.0,
        "d2_arcsec": 0.0,
    }
    j = client.post("/telescope/offset/absorb", headers=h).json()
    assert finished(client, j["id"])["state"] == "done"


def test_only_a_to_granted_lease_commands_the_telescope(client):
    assert client.post("/telescope/slew", json=TARGET).status_code == 403
    s = client.post("/sessions", json={"name": "Eve"}).json()
    sid = {"X-Session-Id": s["session_id"]}
    assert (
        client.post(
            "/control/engineering", json={"passphrase": "eng-pass"}, headers=sid
        ).status_code
        == 200
    )
    token = client.post(f"/sessions/{s['session_id']}/heartbeat").json()["lease_token"]
    r = client.post("/telescope/slew", json=TARGET, headers={**sid, "X-Lease-Token": token})
    assert r.status_code == 403 and "Engineering mode" in r.json()["message"]
    assert client.get("/status").json()["actions"]["slew"]["ok"] is False


def test_a_tcs_that_does_not_accept_commands_greys_the_actions_and_refuses_with_the_reason(client):
    h = hold_lease(client)
    telescope.get_telescope().accepts_commands = False
    why = "The TCS is not accepting external commands: the Telescope Operator switches that on in the TCS"
    assert client.get("/status").json()["actions"]["slew"] == {"ok": False, "why": why}
    r = client.post("/telescope/slew", json=TARGET, headers=h)
    assert r.status_code == 409 and r.json()["code"] == "not_operational"
    assert client.get("/status").json()["tcs"]["acceptsCommands"] is False


def test_one_command_at_a_time(client):
    h = hold_lease(client)
    client.post("/telescope/slew", json=TARGET, headers=h)
    r = client.post("/telescope/slew", json=TARGET, headers=h)
    assert r.status_code == 409 and r.json()["code"] == "busy"


def test_a_target_without_a_name_is_called_by_its_coordinates(client):
    from wilds.control.routers.telescope import coordinate_name

    assert coordinate_name(3.5, 21.25) == "J033000.0+211500"
    assert coordinate_name(23.999999, -0.5) == "J000000.0-003000"
    h = hold_lease(client)
    body = {"ra_hours": 3.5, "dec_deg": 21.25, "rot_pa": 10.0}
    j = client.post("/telescope/slew", json=body, headers=h).json()
    done = finished(client, j["id"])
    assert done["state"] == "done" and done["result"] == {"target": "J033000.0+211500"}
    assert client.get("/telescope").json()["telemetry"]["target"]["name"] == "J033000.0+211500"


def test_a_target_the_tcs_does_not_take_up_ends_refused(client, monkeypatch):
    from wilds.control.telescope import operations

    monkeypatch.setattr(operations, "EFFECT_GRACE_S", 0.5)
    sim = telescope.get_telescope()
    monkeypatch.setattr(sim, "_send_target", lambda *a: None)  # invalid, or out of range
    h = hold_lease(client)
    j = client.post("/telescope/slew", json=TARGET, headers=h).json()
    done = finished(client, j["id"])
    assert done["state"] == "failed" and done["error"]["code"] == "refused"
    assert "did not take it up" in done["error"]["message"]
    assert client.get("/telescope").json()["telemetry"]["target"] is None


def test_a_rotation_the_tcs_does_not_take_up_ends_refused_too(client, monkeypatch):
    from wilds.control.telescope import operations

    monkeypatch.setattr(operations, "EFFECT_GRACE_S", 0.5)
    h = hold_lease(client)
    finished(client, client.post("/telescope/slew", json=TARGET, headers=h).json()["id"])
    monkeypatch.setattr(telescope.get_telescope(), "_send_rotator", lambda *a: None)
    j = client.post("/telescope/rotator", json={"pa": 200}, headers=h).json()
    done = finished(client, j["id"])
    assert done["state"] == "failed" and "did not take it up" in done["error"]["message"]
    assert client.get("/telescope").json()["telemetry"]["rot_pa"] == pytest.approx(40.0)


def test_adjustments_wait_until_the_telescope_is_on_target(client):
    h = hold_lease(client)
    sim = telescope.get_telescope()
    sim.slew_s = 1.5
    offset = {"d1_arcsec": 1, "d2_arcsec": 1}
    for path, body in (("offset", offset), ("rotator", {"pa": 90}), ("offset/clear", None)):
        r = client.post(f"/telescope/{path}", json=body, headers=h)
        assert r.status_code == 409 and r.json()["code"] == "conflict", path
    j = client.post("/telescope/slew", json=TARGET, headers=h).json()
    assert wait_for(lambda: client.get("/telescope").json()["telemetry"]["target"] is not None)
    assert client.get("/status").json()["actions"]["offset"] == {
        "ok": False,
        "why": "The telescope is not on target yet",
    }
    r = client.post("/telescope/offset", json=offset, headers=h)  # slewing: not on target
    assert r.status_code == 409 and r.json()["code"] in ("conflict", "busy")
    assert finished(client, j["id"])["state"] == "done"
    assert client.get("/status").json()["actions"]["offset"]["ok"] is True
    assert client.get("/status").json()["actions"]["rotator"]["ok"] is True
    j = client.post("/telescope/offset", json=offset, headers=h).json()
    assert finished(client, j["id"])["state"] == "done"
