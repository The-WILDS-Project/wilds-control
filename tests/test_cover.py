"""The instrument cover: simulated outside ldt-production, XML to the broker in it (plan §3.17)."""

import asyncio
import time

import pytest
from fastapi.testclient import TestClient

from wilds.control import config, cover, lease
from wilds.control.activities import registry
from wilds.control.app import create_app
from wilds.control.cover import LdtCover, SimulatedCover, stage_command_xml
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


def wait_for(cond, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.02)
    return False


def state(client) -> str:
    return client.get("/cover").json()["state"]


def test_the_simulated_cover_takes_5_s_by_default_and_starts_open():
    c = SimulatedCover()
    assert c.travel_s == 5.0 and c.state == "open"


def test_it_is_in_the_status_with_its_state(client):
    assert state(client) == "open"
    body = client.get("/status").json()["cover"]
    assert body["state"] == "open" and body["kind"] == "simulated" and body["moving"] is False
    assert body["lifecycle"]["operational"] is True


def test_closing_goes_through_closing_and_opening_back_through_opening(client):
    h = hold_lease(client)
    r = client.post("/cover/close", headers=h)
    assert r.status_code == 200 and r.json()["state"] == "closing" and r.json()["moving"] is True
    assert wait_for(lambda: state(client) == "closed")
    assert client.post("/cover/open", headers=h).json()["state"] == "opening"
    assert wait_for(lambda: state(client) == "open")


def test_a_move_while_moving_is_busy_and_where_it_is_already_is_not_an_error(client):
    h = hold_lease(client)
    client.post("/cover/close", headers=h)
    r = client.post("/cover/open", headers=h)
    assert r.status_code == 409 and r.json()["code"] == "busy"
    assert wait_for(lambda: state(client) == "closed")
    assert client.post("/cover/close", headers=h).json()["state"] == "closed"


def test_moving_it_needs_the_lease_but_reading_does_not(client):
    assert client.get("/cover").status_code == 200
    assert client.post("/cover/close").status_code == 403


def test_engineering_may_move_it_too(client, passphrases_file):
    passphrases_file('[passphrases]\nto = "to-pass"\nengineering = "eng-pass"\n')
    s = client.post("/sessions", json={"name": "Eve"}).json()
    sid = {"X-Session-Id": s["session_id"]}
    client.post("/control/engineering", json={"passphrase": "eng-pass"}, headers=sid)
    token = client.post(f"/sessions/{s['session_id']}/heartbeat").json()["lease_token"]
    assert client.post("/cover/close", headers={**sid, "X-Lease-Token": token}).status_code == 200


def test_it_is_not_moved_until_started_up(client):
    h = hold_lease(client)
    client.post("/system/cover/shutdown", headers=TO)
    assert wait_for(lambda: client.get("/cover").json()["lifecycle"]["opstate"] == "shutdown")
    r = client.post("/cover/close", headers=h)
    assert r.status_code == 409 and r.json()["code"] == "not_operational"


# -- the real one -------------------------------------------------------------------------


def test_the_xml_is_what_ldt_gave_us():
    assert "<command>AZ; HM;</command>" in stage_command_xml(close=True)
    assert "<command>AZ; HR;</command>" in stage_command_xml(close=False)
    for xml in (stage_command_xml(close=True), stage_command_xml(close=False)):
        assert "<name>AZ</name>" in xml and "<function>IC</function>" in xml


def test_the_ldt_cover_sends_the_xml_and_starts_unknown():
    sent = []

    async def send(xml):
        sent.append(xml)

    async def go():
        c = LdtCover(send=send, travel_s=0.05)
        await c.startup()
        assert c.state == "unknown"
        await c.close()
        assert c.state == "closing" and "HM" in sent[0]
        await asyncio.sleep(0.15)
        assert c.state == "closed"
        await c.open()
        await asyncio.sleep(0.15)
        assert c.state == "open" and "HR" in sent[1]

    asyncio.run(go())


def test_a_command_that_cannot_be_sent_leaves_the_state_unknown():
    async def send(xml):
        raise ConnectionError("not connected to the LDT broker")

    async def go():
        c = LdtCover(send=send)
        await c.startup()
        with pytest.raises(cover.CoverError):
            await c.close()
        assert c.state == "unknown"

    asyncio.run(go())


def test_only_ldt_production_builds_the_ldt_cover(monkeypatch):
    seen = {}
    monkeypatch.setattr(cover, "_ldt_cover", lambda: seen.setdefault("made", SimulatedCover()))
    for mode, made in (("development", False), ("ldt-simulated", False), ("ldt-production", True)):
        cover.reset()
        monkeypatch.setattr(config, "get", lambda m=mode: type("C", (), {"opmode": m})())
        cover.get_cover()
        assert ("made" in seen) is made
        seen.clear()
