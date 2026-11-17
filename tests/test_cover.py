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


# -- the instrument cube's reports ---------------------------------------------------------


def report(cover_state="OPEN", fm="EXTENDED", **kw):
    return cover.CubeReport(
        cover_state=cover_state, cover_position_mm=0.0, fm_c_state=fm, fm_c_position_mm=100.0, **kw
    )


async def _noop(xml):
    return None


def test_a_report_of_arrival_settles_the_move_before_the_timer():
    async def go():
        c = LdtCover(send=_noop)  # 10 s to arrive, if nobody says
        await c.startup()
        c.feed(report("OPEN"))
        assert c.state == "open" and c.sensed
        await c.close()
        assert c.state == "closing"
        c.feed(report("OPEN"))  # still where it was: on its way
        assert c.state == "closing"
        c.feed(report("CLOSED"))
        assert c.state == "closed" and not c.moving
        await asyncio.sleep(0)
        assert c._task is None
        await c.open()  # a new move may start at once
        assert c.state == "opening"

    asyncio.run(go())


def test_the_first_report_tells_where_an_unknown_cover_is():
    async def go():
        c = LdtCover(send=_noop)
        await c.startup()
        assert c.state == "unknown" and not c.sensed and c.status().cube is None
        c.feed(report("CLOSED"))
        assert c.state == "closed" and c.status().sensed

    asyncio.run(go())


def test_a_cover_moved_by_somebody_else_shows_when_it_arrives():
    async def go():
        c = LdtCover(send=_noop)
        await c.startup()
        c.feed(report("OPEN"))
        c.feed(report("CLOSED"))  # the TO closed it
        assert c.state == "closed"
        c.feed(report("OPEN"))
        assert c.state == "open"

    asyncio.run(go())


def test_a_move_the_cube_never_shows_is_judged_by_what_it_says(monkeypatch):
    monkeypatch.setattr(cover.base, "SENSE_GRACE_S", 0.05)

    async def go():
        c = LdtCover(send=_noop, travel_s=0.05)
        await c.startup()
        c.feed(report("OPEN"))
        await c.close()
        await asyncio.sleep(0.3)
        assert c.state == "open"  # it did not move: the cube still says OPEN

    asyncio.run(go())


def test_without_reports_the_commands_are_the_truth(monkeypatch):
    monkeypatch.setattr(cover.base, "STALE_S", 0.05)

    async def go():
        c = LdtCover(send=_noop, travel_s=0.05)
        await c.startup()
        c.feed(report("OPEN"))
        await asyncio.sleep(0.1)  # the cube has gone quiet
        assert not c.sensed and c.state == "open"
        await c.close()
        await asyncio.sleep(0.15)
        assert c.state == "closed"

    asyncio.run(go())


def test_an_unrecognised_state_is_not_believed():
    async def go():
        c = LdtCover(send=_noop)
        await c.startup()
        c.feed(report("OPEN"))
        c.feed(report("MOVING"))
        assert c.state == "open"

    asyncio.run(go())


def test_the_fold_mirror_that_feeds_wilds():
    async def go():
        c = LdtCover(send=_noop)
        await c.startup()
        c.feed(report(fm="EXTENDED"))
        cube = c.status().cube
        assert cube and cube.fold_mirror == "C" and cube.to_wilds is True
        assert cube.fold_mirror_position_mm == 100.0
        c.feed(report(fm="HOME"))
        assert c.status().cube.to_wilds is False
        c.feed(cover.CubeReport(cover_state="OPEN"))
        assert c.status().cube.to_wilds is None

    asyncio.run(go())


def test_the_broker_telemetry_becomes_a_report():
    from wilds.bridge.models.instrument_cube_telemetry import InstrumentCubeTelemetry

    xml = (
        "<InstrumentCubeTelemetry><TimeStamp>1</TimeStamp>"
        "<InstrumentCoverState>CLOSED</InstrumentCoverState>"
        "<InstrumentCoverPosition>12.5</InstrumentCoverPosition>"
        "<FMCState>HOME</FMCState><FMCPosition>0.0</FMCPosition></InstrumentCubeTelemetry>"
    )
    r = cover.cube_report(InstrumentCubeTelemetry.from_xml(xml))
    assert (r.cover_state, r.cover_position_mm) == ("CLOSED", 12.5)
    assert (r.fm_c_state, r.fm_c_position_mm) == ("HOME", 0.0)


def test_the_ldt_cover_listens_at_startup_and_a_broker_that_is_down_does_not_stop_it():
    calls = []

    async def listen():
        calls.append("listen")
        raise ConnectionError("no broker")

    async def go():
        c = LdtCover(send=_noop, listen=listen)
        status = await c.startup()
        assert calls == ["listen"] and status.operational and c.state == "unknown"

    asyncio.run(go())


def test_the_ldt_cover_takes_up_to_10_s_to_move():
    assert LdtCover(send=_noop).travel_s == 10.0


# -- the simulated cube ------------------------------------------------------------------


def test_the_simulated_cube_reports_and_the_cover_follows_it(client):
    assert wait_for(lambda: client.get("/cover").json()["sensed"])
    body = client.get("/status").json()["cover"]
    assert body["state"] == "open" and body["cube"]["to_wilds"] is True
    assert body["cube"]["fold_mirror"] == "C" and body["cube"]["fold_mirror_state"] == "EXTENDED"
    h = hold_lease(client)
    client.post("/cover/close", headers=h)
    assert wait_for(lambda: state(client) == "closed")
    body = client.get("/cover").json()
    assert body["sensed"] and body["cube"]["cover_position_mm"] > 0


def test_the_simulated_cube_says_open_until_the_cover_has_arrived():
    c = SimulatedCover(travel_s=0.3)

    async def go():
        await c._send(True)
        mid = c._hw_report()
        assert mid.cover_state == "OPEN" and 0 <= mid.cover_position_mm < 250
        await asyncio.sleep(0.4)
        done = c._hw_report()
        assert done.cover_state == "CLOSED" and done.cover_position_mm == 250.0

    asyncio.run(go())


def test_the_simulated_cube_stops_with_the_cover(client):
    assert wait_for(lambda: client.get("/cover").json()["sensed"])
    client.post("/system/cover/shutdown", headers=TO)
    assert wait_for(lambda: client.get("/cover").json()["lifecycle"]["opstate"] == "shutdown")
    age = client.get("/cover").json()["cube"]["age_s"]
    time.sleep(0.3)
    assert client.get("/cover").json()["cube"]["age_s"] >= age  # no fresh reports
