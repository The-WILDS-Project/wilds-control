import time

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


def test_status_reports_the_opmode_development_by_default(client):
    assert client.get("/status").json()["opmode"] == "development"


def test_the_demo_job_needs_the_lease(client):
    assert client.post("/dev/jobs/demo").status_code == 403


def test_the_demo_job_runs_and_raises_activities_on_three_components(client):
    r = client.post("/dev/jobs/demo", headers=hold_lease(client), json={"seconds": 3})
    assert r.status_code == 202
    job = r.json()
    assert job["kind"] == "testing-123" and job["started_by"]["name"] == "Arie"
    assert sorted(job["components"]) == ["adc", "lamps", "slit"]
    time.sleep(0.4)
    status = client.get("/status").json()
    assert set(status["activities"]) == {"slit", "adc"}
    assert [j["id"] for j in status["jobs"]] == [job["id"]]


def test_a_second_demo_job_is_busy_unless_it_queues(client):
    h = hold_lease(client)
    first = client.post("/dev/jobs/demo", headers=h).json()
    r = client.post("/dev/jobs/demo", headers=h)
    assert r.status_code == 409 and r.json()["code"] == "busy"
    assert r.json()["details"] == {"job_id": first["id"]}
    queued = client.post("/dev/jobs/demo", headers=h, json={"queue": True}).json()
    assert queued["state"] == "queued"


def test_the_failing_demo_job_ends_failed_with_the_standard_error(client):
    h = hold_lease(client)
    job = client.post("/dev/jobs/demo", headers=h, json={"outcome": "fail", "seconds": 3}).json()
    deadline = time.time() + 5
    while time.time() < deadline:
        got = client.get(f"/jobs/{job['id']}").json()
        if got["state"] != "running":
            break
        time.sleep(0.1)
    assert got["state"] == "failed" and got["error"]["code"] == "refused"
    assert got["error"]["details"] == {"phase": 3}
    assert client.get("/status").json()["activities"] == {}  # every activity was ended


def test_production_refuses_the_dev_helpers_without_a_restart(client, config_file):
    h = hold_lease(client)
    config_file('opmode = "ldt-production"\n')
    assert client.get("/status").json()["opmode"] == "ldt-production"
    r = client.post("/dev/jobs/demo", headers=h)
    assert r.status_code == 404 and r.json()["code"] == "not_found"


def test_an_unknown_opmode_rejects_the_file_and_keeps_the_running_one(client, config_file):
    assert client.get("/status").json()["opmode"] == "development"  # a valid file is running
    config_file('opmode = "sandbox"\n')
    assert client.get("/status").json()["opmode"] == "development"  # the bad one is rejected


def test_the_dev_helpers_exist_only_in_development_not_against_ldts_simulators(client, config_file):
    h = hold_lease(client)
    config_file('opmode = "ldt-simulated"\n[broker]\nhost = "tanagra"\n')
    r = client.post("/dev/jobs/demo", headers=h)
    assert r.status_code == 404 and r.json()["code"] == "not_found"
    assert client.get("/status").json()["opmode"] == "ldt-simulated"
    assert any("in-process simulator" in w for w in client.get("/status").json()["warnings"])
