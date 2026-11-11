"""The shutter: lifecycle, moving, holding it closed for a procedure, and its HTTP side."""

import asyncio
import time

import pytest
from fastapi.testclient import TestClient

from wilds.control import lease
from wilds.control.activities import registry
from wilds.control.app import create_app
from wilds.control.errors import ApiError
from wilds.control.jobs import StartedBy, manager
from wilds.control.shutter import ShutterBusy, SimulatedShutter

ARIE = StartedBy(name="Arie", code="K7-42")


def make(**kw) -> SimulatedShutter:
    return SimulatedShutter(travel_s=kw.pop("travel_s", 0.01), **kw)


async def started(**kw) -> SimulatedShutter:
    s = make(**kw)
    s.conf.patch({"settle_s": 0}, who=ARIE)
    await s.startup()
    return s


# -- lifecycle ----------------------------------------------------------------------------


async def test_an_unstarted_shutter_is_not_operational_and_will_not_move():
    s = make()
    assert not s.life.operational and s.status().state == "unknown"
    with pytest.raises(ApiError) as e:
        await s.open()
    assert e.value.status == 409 and e.value.body.code == "not_operational"


@pytest.mark.parametrize(("start_open", "state"), [(False, "closed"), (True, "open")])
async def test_startup_finds_the_blade_where_it_is_without_moving_it(start_open, state):
    s = make(start_open=start_open)
    status = await s.startup()
    assert status.operational and s.state == state


async def test_shutdown_closes_the_shutter_and_startup_finds_it_closed():
    s = await started(start_open=True)
    status = await s.shutdown()
    assert not status.operational and s.state == "closed"
    await s.startup()
    assert s.state == "closed" and s.life.operational


# -- moving -------------------------------------------------------------------------------


async def test_open_and_close_change_the_state_and_show_as_activities():
    s = await started()
    seen: list[list[str]] = []

    async def watch():
        for _ in range(20):
            seen.append(list(s.activities.names))
            await asyncio.sleep(0.002)

    watcher = asyncio.create_task(watch())
    assert await s.open() == "open"
    assert await s.close() == "closed"
    await watcher
    assert ["Opening"] in seen and ["Closing"] in seen and s.activities.names == []


async def test_it_reads_moving_while_the_blade_travels():
    s = await started(travel_s=0.1)
    task = asyncio.create_task(s.open())
    await asyncio.sleep(0.03)
    assert s.state == "moving"
    await task
    assert s.state == "open"


async def test_a_second_move_while_moving_is_refused():
    s = await started(travel_s=0.1)
    task = asyncio.create_task(s.open())
    await asyncio.sleep(0.03)
    with pytest.raises(ShutterBusy):
        await s.close()
    await task


async def test_the_settle_time_is_part_of_a_move():
    s = await started(travel_s=0.0)
    s.conf.patch({"settle_s": 0.2}, who=ARIE)
    start = time.monotonic()
    await s.open()
    assert time.monotonic() - start >= 0.2


# -- holding it closed --------------------------------------------------------------------


async def test_closed_puts_it_back_as_found():
    s = await started(start_open=True)
    async with s.closed():
        assert s.state == "closed"
    assert s.state == "open"


async def test_closed_puts_it_back_even_when_the_block_fails():
    s = await started(start_open=True)
    with pytest.raises(RuntimeError):
        async with s.closed():
            raise RuntimeError("boom")
    assert s.state == "open"


async def test_closed_leaves_a_closed_shutter_closed():
    s = await started(start_open=False)
    async with s.closed():
        pass
    assert s.state == "closed"


# -- HTTP ---------------------------------------------------------------------------------


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


def test_the_status_and_the_system_carry_the_shutter(client):
    body = client.get("/status").json()
    assert body["shutter"]["kind"] == "simulated" and body["shutter"]["state"] == "closed"
    assert body["shutter"]["lifecycle"]["operational"] is True  # autostart in development
    assert "shutter" in body["system"]["components"]
    assert client.get("/shutter").json() == body["shutter"] | {"state": "closed"}


def test_moving_it_needs_the_lease(client):
    assert client.post("/shutter/open").status_code == 403
    assert client.post("/shutter/close").status_code == 403


def test_open_and_close_with_the_lease(client):
    h = hold_lease(client)
    assert client.post("/shutter/open", headers=h).json()["state"] == "open"
    assert client.post("/shutter/close", headers=h).json()["state"] == "closed"


def test_a_shutdown_shutter_refuses_with_not_operational(client):
    h = hold_lease(client)
    client.post("/system/shutter/shutdown", headers={"X-TO-Passphrase": "to-pass"})
    for _ in range(100):
        if client.get("/shutter").json()["lifecycle"]["opstate"] == "shutdown":
            break
        time.sleep(0.05)
    r = client.post("/shutter/open", headers=h)
    assert r.status_code == 409 and r.json()["code"] == "not_operational"


def test_a_job_that_has_the_shutter_keeps_it_from_hand_requests(client):
    h = hold_lease(client)
    manager._busy["shutter"] = "j-7"  # as a running dark sequence reserves it
    try:
        r = client.post("/shutter/open", headers=h)
        assert r.status_code == 409 and r.json()["code"] == "busy"
        assert r.json()["details"]["job_id"] == "j-7"
        assert client.post("/shutter/close", headers=h).status_code == 409
    finally:
        del manager._busy["shutter"]
    assert client.post("/shutter/open", headers=h).status_code == 200


def test_its_settings_are_served_by_the_config_endpoints(client):
    body = client.get("/config/shutter/conf").json()
    assert body["values"]["settle_s"] == 0.1
    assert body["json_schema"]["properties"]["settle_s"]["unit"] == "s"


async def test_it_stays_operational_while_the_blade_moves():
    s = await started(travel_s=0.1)
    task = asyncio.create_task(s.open())
    await asyncio.sleep(0.03)
    assert s.life.operational and s.life.why_not_operational() == []
    await task
