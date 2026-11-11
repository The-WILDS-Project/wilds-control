"""The light path: ask the TO to block it, the TO acknowledges (plan §3.17)."""

import pytest
from fastapi.testclient import TestClient

from wilds.control import lease
from wilds.control.activities import registry
from wilds.control.app import create_app
from wilds.control.jobs import manager

TO = {"X-TO-Passphrase": "to-pass"}


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


def state(client) -> dict:
    return client.get("/light-path").json()


def test_it_starts_unknown_and_is_in_the_status(client):
    assert state(client)["state"] == "unknown"
    assert client.get("/status").json()["light_path"]["state"] == "unknown"


def test_a_block_request_the_tos_acknowledgement_and_a_restore(client):
    h = hold_lease(client)
    r = client.post("/light-path/block-request", headers=h)
    assert r.status_code == 200 and r.json()["state"] == "block_requested"
    assert r.json()["requested_by"] == "Arie"
    r = client.post("/light-path/acknowledge", json={"note": "M1 cover closed"}, headers=TO)
    assert r.json()["state"] == "blocked" and r.json()["note"] == "M1 cover closed"
    assert r.json()["requested_by"] is None
    assert (
        client.post("/light-path/restore-request", headers=h).json()["state"] == "unblock_requested"
    )
    assert client.post("/light-path/acknowledge", headers=TO).json()["state"] == "lit"


def test_only_the_lease_holder_asks_and_only_the_to_answers(client):
    assert client.post("/light-path/block-request").status_code == 403
    h = hold_lease(client)
    client.post("/light-path/block-request", headers=h)
    assert client.post("/light-path/acknowledge", headers=h).status_code == 403
    assert client.post("/light-path/refuse", json={"reason": "x"}, headers=h).status_code == 403
    assert client.put("/light-path/state", json={"state": "lit"}, headers=h).status_code == 403
    assert client.get("/light-path").status_code == 200  # anyone reads


def test_a_refusal_puts_the_state_back_and_shows_why(client):
    h = hold_lease(client)
    client.post("/light-path/block-request", headers=h)
    r = client.post("/light-path/refuse", json={"reason": "observing with LMI"}, headers=TO)
    assert r.json()["state"] == "unknown" and r.json()["refusal"] == "observing with LMI"
    client.post("/light-path/block-request", headers=h)  # a new request clears the refusal
    assert state(client)["refusal"] is None


def test_the_requester_can_cancel(client):
    h = hold_lease(client)
    client.put("/light-path/state", json={"state": "lit"}, headers=TO)
    client.post("/light-path/block-request", headers=h)
    assert client.post("/light-path/cancel", headers=h).json()["state"] == "lit"
    client.put("/light-path/state", json={"state": "blocked"}, headers=TO)
    client.post("/light-path/restore-request", headers=h)
    assert client.post("/light-path/cancel", headers=h).json()["state"] == "blocked"


def test_requests_that_do_not_fit_are_conflicts(client):
    h = hold_lease(client)
    for path in ("cancel",):
        assert client.post(f"/light-path/{path}", headers=h).json()["code"] == "conflict"
    assert client.post("/light-path/acknowledge", headers=TO).status_code == 409
    assert client.post("/light-path/refuse", json={"reason": "x"}, headers=TO).status_code == 409
    client.put("/light-path/state", json={"state": "lit"}, headers=TO)
    assert client.post("/light-path/restore-request", headers=h).status_code == 409  # already lit
    client.post("/light-path/block-request", headers=h)
    assert client.post("/light-path/block-request", headers=h).status_code == 409  # one at a time
    client.put("/light-path/state", json={"state": "blocked"}, headers=TO)
    assert client.post("/light-path/block-request", headers=h).status_code == 409  # already


def test_the_to_can_state_how_things_are_at_any_time(client):
    r = client.put("/light-path/state", json={"state": "blocked", "note": "cover"}, headers=TO)
    assert r.json()["state"] == "blocked" and r.json()["note"] == "cover"
    assert (
        client.put("/light-path/state", json={"state": "lit"}, headers=TO).json()["state"] == "lit"
    )
    assert client.put("/light-path/state", json={"state": "bogus"}, headers=TO).status_code == 422
