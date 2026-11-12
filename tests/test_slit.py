"""The slit stage: lifecycle, moving, presets, widths in arcsec, and its HTTP side."""

import asyncio
import time

import pytest
from fastapi.testclient import TestClient

from wilds.control import lease
from wilds.control.activities import registry
from wilds.control.app import create_app
from wilds.control.errors import ApiError
from wilds.control.jobs import StartedBy, manager
from wilds.control.slit import SimulatedSlit, SlitBusy, SlitRange

ARIE = StartedBy(name="Arie", code="K7-42")


def make(**kw) -> SimulatedSlit:
    return SimulatedSlit(speed_um_s=kw.pop("speed_um_s", 1e6), **kw)


async def started(**kw) -> SimulatedSlit:
    s = make(**kw)
    s.conf.patch({"settle_s": 0}, who=ARIE)
    await s.startup()
    return s


# -- lifecycle ----------------------------------------------------------------------------


async def test_an_unstarted_slit_is_not_operational_and_will_not_move():
    s = make()
    with pytest.raises(ApiError) as e:
        await s.move_to(100)
    assert e.value.status == 409 and e.value.body.code == "not_operational"


async def test_startup_finds_the_stage_where_it_is_without_moving_it():
    s = make(start_um=500.0)
    status = await s.startup()
    assert status.operational and s.position_um == 500.0 and s.at_preset() is None


async def test_shutdown_closes_the_slit():
    s = await started(start_um=500.0)
    await s.shutdown()
    assert s.position_um == 2.0 and s.at_preset() == "closed"


# -- moving -------------------------------------------------------------------------------


async def test_it_moves_to_a_position_and_to_the_presets():
    s = await started()
    assert s.at_preset() == "closed" and s.width_arcsec == pytest.approx(2 / 124.69)
    await s.move_to(124.69)
    assert (
        s.position_um == 124.69 and s.width_arcsec == pytest.approx(1.0) and s.at_preset() is None
    )
    await s.move_to_preset("open")
    assert s.position_um == 21000.0 and s.at_preset() == "open"
    await s.move_to_preset("closed")
    assert s.position_um == 2.0


async def test_a_position_outside_the_travel_is_refused_and_nothing_moves():
    s = await started()
    for bad in (-1.0, 21000.5):
        with pytest.raises(SlitRange):
            await s.move_to(bad)
    assert s.position_um == 2.0


async def test_the_presets_and_the_scale_are_settings():
    s = await started()
    s.conf.patch({"open_um": 1500.0, "um_per_arcsec": 100.0}, who=ARIE)
    await s.move_to_preset("open")
    assert s.position_um == 1500.0 and s.width_arcsec == 15.0
    assert s.um_for_arcsec(1.0) == 100.0


async def test_it_reads_moving_and_shows_an_activity_while_the_stage_travels():
    s = await started(speed_um_s=1000.0)  # 2 s for the full travel
    task = asyncio.create_task(s.move_to(500.0))
    await asyncio.sleep(0.05)
    assert s.status().moving and s.position_um is None and s.activities.names == ["Moving"]
    await task
    assert not s.status().moving and s.activities.names == []


async def test_a_second_move_while_moving_is_refused():
    s = await started(speed_um_s=1000.0)
    task = asyncio.create_task(s.move_to(500.0))
    await asyncio.sleep(0.05)
    with pytest.raises(SlitBusy):
        await s.move_to(100.0)
    await task


async def test_the_settle_time_is_part_of_a_move():
    s = await started()
    s.conf.patch({"settle_s": 0.2}, who=ARIE)
    start = time.monotonic()
    await s.move_to(300.0)
    assert time.monotonic() - start >= 0.2


async def test_holding_puts_the_stage_back_even_when_the_block_fails():
    s = await started(start_um=300.0)
    with pytest.raises(RuntimeError):
        async with s.holding():
            await s.move_to_preset("open")
            raise RuntimeError("boom")
    assert s.position_um == 300.0


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


def test_the_status_and_the_system_carry_the_slit(client):
    body = client.get("/status").json()
    slit = body["slit"]
    assert slit["kind"] == "simulated" and slit["preset"] == "closed"
    assert (slit["min_um"], slit["max_um"], slit["um_per_arcsec"]) == (0.0, 21000.0, 124.69)
    assert slit["lifecycle"]["operational"] is True  # autostart in development
    assert "slit" in body["system"]["components"]
    assert client.get("/slit").json() == slit


def test_moving_it_needs_the_lease(client):
    assert client.post("/slit/move", json={"preset": "open"}).status_code == 403


def test_move_to_a_preset_or_a_position_with_the_lease(client):
    h = hold_lease(client)
    r = client.post("/slit/move", json={"preset": "open"}, headers=h).json()
    assert r["position_um"] == 21000.0 and r["preset"] == "open"
    r = client.post("/slit/move", json={"position_um": 124.69}, headers=h).json()
    assert r["width_arcsec"] == pytest.approx(1.0) and r["preset"] is None


@pytest.mark.parametrize(
    "body", [{}, {"preset": "open", "position_um": 5.0}, {"preset": "ajar"}, {"position_um": -3}]
)
def test_a_malformed_move_is_422(client, body):
    h = hold_lease(client)
    assert client.post("/slit/move", json=body, headers=h).status_code == 422


def test_a_position_beyond_the_travel_is_422_invalid(client):
    h = hold_lease(client)
    r = client.post("/slit/move", json={"position_um": 25000}, headers=h)
    assert r.status_code == 422 and r.json()["code"] == "invalid"


def test_a_job_that_has_the_slit_keeps_it_from_hand_requests(client):
    h = hold_lease(client)
    manager._busy["slit"] = "j-9"  # as a running calibration reserves it
    try:
        r = client.post("/slit/move", json={"preset": "open"}, headers=h)
        assert r.status_code == 409 and r.json()["code"] == "busy"
        assert r.json()["details"]["job_id"] == "j-9"
    finally:
        del manager._busy["slit"]
    assert client.post("/slit/move", json={"preset": "open"}, headers=h).status_code == 200


def test_its_settings_are_served_by_the_config_endpoints(client):
    body = client.get("/config/slit/conf").json()
    assert body["values"]["um_per_arcsec"] == 124.69
    assert body["json_schema"]["properties"]["closed_um"]["unit"] == "µm"


async def test_it_stays_operational_while_the_stage_moves():
    s = await started(speed_um_s=1000.0)
    task = asyncio.create_task(s.move_to(500.0))
    await asyncio.sleep(0.05)
    assert s.life.operational
    await task
