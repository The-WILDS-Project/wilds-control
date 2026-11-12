"""Startup, shutdown, operational and the umbrella activities (plan §3.16)."""

import asyncio
import time
from enum import Enum

import pytest
from fastapi.testclient import TestClient

from wilds.control import guide_camera, lease
from wilds.control.activities import Activities, registry
from wilds.control.app import create_app
from wilds.control.errors import ApiError
from wilds.control.jobs import manager
from wilds.control.lifecycle import Lifecycle, OpState, Umbrella, system


class Acts(Enum):
    Doing = 1
    StartingUp = 2
    ShuttingDown = 3


def part(name: str, **hooks) -> Lifecycle:
    acts = Activities(name, Acts)
    return Lifecycle(name, lambda: acts, **hooks)


# -- one component ----------------------------------------------------------------------


async def test_a_new_component_is_initialized_and_not_operational():
    life = part("adc")
    assert life.opstate is OpState.initialized
    assert not life.operational and life.why_not_operational() == ["not started up"]


async def test_startup_makes_it_startedup_and_operational():
    seen = []
    life = part("adc", startup=lambda: seen.append("home"))
    status = await life.startup()
    assert seen == ["home"] and status.opstate is OpState.startedup
    assert status.operational and status.why_not_operational == []
    assert life.activities.names == []  # StartingUp has ended


async def test_startingup_is_an_activity_while_it_runs():
    names = []
    life = part("adc")
    life._startup = lambda: names.append(list(life.activities.names))
    await life.startup()
    assert names == [["StartingUp"]]


async def test_live_reasons_come_from_the_component_and_can_change_after_startup():
    reasons = ["lamp 1 warming up"]
    life = part("lamps", why_not=lambda: list(reasons))
    await life.startup()
    assert life.opstate is OpState.startedup and not life.operational
    assert life.why_not_operational() == ["lamp 1 warming up"]
    reasons.clear()
    assert life.operational  # computed, not stored


async def test_a_failed_startup_is_startedup_but_not_operational_and_can_be_retried():
    attempts = []

    def hw():
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("home mark not found")

    life = part("adc", startup=hw)
    status = await life.startup()
    assert status.opstate is OpState.startedup and not status.operational
    assert status.why_not_operational == ["startup failed: home mark not found"]
    assert (await life.startup()).operational  # the retry clears it


async def test_a_second_startup_while_one_runs_is_busy():
    life = part("adc")
    life.activities.start(Acts.StartingUp)  # one is running
    with pytest.raises(ApiError) as e:
        await life.startup()
    assert e.value.status == 409 and e.value.body.code == "busy"
    with pytest.raises(ApiError):
        await life.shutdown()
    life.activities.end(Acts.StartingUp)
    assert (await life.startup()).operational


async def test_shutdown_deactivates_and_it_can_start_again():
    life = part("adc")
    await life.startup()
    status = await life.shutdown()
    assert status.opstate is OpState.shutdown
    assert life.why_not_operational() == ["shut down"]
    assert (await life.startup()).operational


async def test_shutdown_refuses_while_busy_unless_forced():
    halted = []
    life = part("cam", halt=lambda: halted.append(1))
    await life.startup()
    life.activities.start(Acts.Doing)
    with pytest.raises(ApiError) as e:
        await life.shutdown()
    assert e.value.body.code == "busy" and life.opstate is OpState.startedup
    life.activities.end(Acts.Doing)  # (the halt of a real component ends its activity)
    life.activities.start(Acts.Doing)
    status = await life.shutdown(force=True)
    assert halted == [1] and status.opstate is OpState.shutdown


async def test_require_operational_says_why():
    life = part("adc")
    with pytest.raises(ApiError) as e:
        life.require_operational()
    assert e.value.status == 409 and e.value.body.code == "not_operational"
    assert e.value.body.details["why_not_operational"] == ["not started up"]
    await life.startup()
    life.require_operational()


# -- umbrellas --------------------------------------------------------------------------


def test_an_umbrella_runs_until_the_last_constituent_is_done():
    parent = Activities("wilds", Acts)
    a, b = Activities("a", Acts), Activities("b", Acts)
    u = Umbrella(lambda: parent, "StartingUp")
    u.add(a)
    u.add(b)
    assert parent.names == []
    a.start(Acts.StartingUp)
    assert parent.names == ["StartingUp"]
    b.start(Acts.StartingUp)
    a.end(Acts.StartingUp)
    assert parent.names == ["StartingUp"]  # b is still going
    b.end(Acts.StartingUp)
    assert parent.names == []


def test_an_umbrella_ignores_other_activities():
    parent, a = Activities("wilds", Acts), Activities("a", Acts)
    u = Umbrella(lambda: parent, "StartingUp")
    u.add(a)
    a.start(Acts.Doing)
    assert parent.names == []


def test_a_hold_spans_the_gap_between_sequential_constituents():
    parent, a = Activities("wilds", Acts), Activities("a", Acts)
    u = Umbrella(lambda: parent, "StartingUp")
    u.add(a)
    with u.hold():
        a.start(Acts.StartingUp)
        a.end(Acts.StartingUp)
        assert parent.names == ["StartingUp"]  # nothing running, but the job is not over
    assert parent.names == []


def test_umbrellas_nest():
    top, mid, leaf = Activities("top", Acts), Activities("mid", Acts), Activities("leaf", Acts)
    Umbrella(lambda: mid, "StartingUp").add(leaf)
    Umbrella(lambda: top, "StartingUp").add(mid)
    leaf.start(Acts.StartingUp)
    assert mid.names == top.names == ["StartingUp"]
    leaf.end(Acts.StartingUp)
    assert mid.names == top.names == []


# -- the system -------------------------------------------------------------------------


async def test_system_startup_runs_every_component_under_one_umbrella():
    during = []
    a = part("a", startup=lambda: time.sleep(0.05))
    b = part("b", startup=lambda: time.sleep(0.05))
    system.register(a)
    system.register(b)
    watcher = asyncio.create_task(_watch(during))
    status = await system.startup()
    await watcher
    assert status.operational and set(status.components) == {"a", "b"}
    assert "StartingUp" in during  # the wilds umbrella ran
    assert system.activities.names == []  # and ended with the last component


async def _watch(into: list[str]) -> None:
    for _ in range(40):
        into += system.activities.names
        await asyncio.sleep(0.005)


async def test_system_status_prefixes_each_reason_with_its_component():
    system.register(part("a"))
    system.register(part("b", why_not=lambda: ["cold"]))
    await system.get("b").startup()
    st = system.status()
    assert not st.operational
    assert st.why_not_operational == ["a: not started up", "b: cold"]


async def test_the_system_opstate_is_startedup_or_shutdown_only_when_every_component_is():
    assert system.status().opstate == "initialized"  # nothing registered yet
    system.register(part("a"))
    system.register(part("b"))
    assert system.status().opstate == "initialized"
    await system.get("a").startup()
    assert system.status().opstate == "initialized"  # one up, one not
    await system.get("b").startup()
    assert system.status().opstate == "startedup"
    await system.get("a").shutdown()
    assert system.status().opstate == "initialized"  # a start-up and a shutdown both change it
    await system.get("b").shutdown()
    assert system.status().opstate == "shutdown"


async def test_one_component_being_busy_does_not_stop_the_others():
    a, b = part("a"), part("b")
    system.register(a)
    system.register(b)
    a.activities.start(Acts.StartingUp)  # a is already starting up by itself
    status = await system.startup()
    assert status.components["b"].operational
    assert any("already running" in r for r in status.components["a"].why_not_operational)
    a.activities.end(Acts.StartingUp)


async def test_individually_started_component_lights_the_system_umbrella():
    a = part("a", startup=lambda: time.sleep(0.05))
    system.register(a)
    task = asyncio.create_task(a.startup())
    await asyncio.sleep(0.02)
    assert system.activities.names == ["StartingUp"]
    await task
    assert system.activities.names == []


# -- the guide camera and the HTTP side -------------------------------------------------

TO = {"X-TO-Passphrase": "to-pass"}


@pytest.fixture
def make_client(config_file, passphrases_file):
    passphrases_file('[passphrases]\nto = "to-pass"\n')

    def make(toml: str = "") -> TestClient:
        config_file(toml)
        lease.state.sessions.clear()
        lease.state.holder_id = lease.state.token = None
        manager.reset()
        registry.clear()
        return TestClient(create_app())

    yield make
    manager.reset()
    registry.clear()


def wait_job(client, job_id, headers=None):
    deadline = time.time() + 5
    while time.time() < deadline:
        job = client.get(f"/jobs/{job_id}").json()
        if job["state"] not in ("queued", "running"):
            return job
        time.sleep(0.02)
    raise AssertionError("job did not finish")


def hold_lease(client) -> dict[str, str]:
    s = client.post("/sessions", json={"name": "Arie"}).json()
    sid = {"X-Session-Id": s["session_id"]}
    client.post("/control/requests", headers=sid)
    client.post(f"/control/requests/{s['code']}/grant", headers=TO)
    token = client.post(f"/sessions/{s['session_id']}/heartbeat").json()["lease_token"]
    return {**sid, "X-Lease-Token": token}


def test_in_development_the_app_starts_everything_up_at_boot(make_client):
    with make_client() as c:
        st = c.get("/system").json()
        assert st["operational"] and st["components"]["guide-camera"]["opstate"] == "startedup"
        assert c.get("/status").json()["system"]["operational"]


def test_in_production_nothing_starts_until_told(make_client):
    with make_client('opmode = "ldt-production"\n[broker]\nhost = "joe"\n') as c:
        cam = c.get("/guide-camera").json()
        assert cam["lifecycle"]["opstate"] == "initialized" and not cam["lifecycle"]["operational"]
        r = c.post("/guide-camera/frames", headers=hold_lease(c))
        assert r.status_code == 409 and r.json()["code"] == "not_operational"
        assert r.json()["details"]["why_not_operational"] == ["not started up"]


def test_autostart_can_be_set_either_way(make_client):
    with make_client(
        'opmode = "ldt-production"\n[broker]\nhost = "joe"\n[startup]\nautostart = true\n'
    ) as c:
        assert c.get("/system").json()["operational"]
    with make_client("[startup]\nautostart = false\n") as c:
        assert not c.get("/system").json()["operational"]


def test_the_telescope_operator_can_start_everything_up_with_no_session(make_client):
    with make_client('opmode = "ldt-production"\n[broker]\nhost = "joe"\n') as c:
        r = c.post("/system/startup", headers=TO)
        assert r.status_code == 202 and r.json()["started_by"] == {
            "name": "Telescope Operator",
            "code": "TO",
        }
        job = wait_job(c, r.json()["id"])
        assert job["state"] == "done" and job["result"]["operational"] is True
        assert c.get("/system").json()["operational"]


def test_the_holder_of_control_can_too_and_nobody_else(make_client):
    with make_client('opmode = "ldt-production"\n[broker]\nhost = "joe"\n') as c:
        assert c.post("/system/startup").status_code == 403
        assert c.post("/system/startup", headers={"X-TO-Passphrase": "nope"}).status_code == 403
        h = hold_lease(c)
        r = c.post("/system/startup", headers=h)
        assert r.status_code == 202 and r.json()["started_by"]["name"] == "Arie"
        wait_job(c, r.json()["id"])


def test_one_component_by_name_and_an_unknown_one(make_client):
    with make_client('opmode = "ldt-production"\n[broker]\nhost = "joe"\n') as c:
        assert c.post("/system/nope/startup", headers=TO).status_code == 404
        r = c.post("/system/guide-camera/startup", headers=TO)
        job = wait_job(c, r.json()["id"])
        assert job["result"]["opstate"] == "startedup" and job["result"]["operational"]


def test_shutdown_by_the_to_then_start_again(make_client):
    with make_client() as c:
        job = wait_job(c, c.post("/system/shutdown", headers=TO).json()["id"])
        assert job["result"]["components"]["guide-camera"]["opstate"] == "shutdown"
        assert c.post("/guide-camera/frames", headers=hold_lease(c)).status_code == 409
        wait_job(c, c.post("/system/startup", headers=TO).json()["id"])
        assert c.get("/system").json()["operational"]


def test_shutdown_refuses_a_streaming_camera_unless_forced(make_client):
    with make_client() as c:
        h = hold_lease(c)
        assert c.post("/guide-camera/start", headers=h).status_code == 200
        job = wait_job(c, c.post("/system/guide-camera/shutdown", headers=TO).json()["id"])
        assert job["state"] == "failed" and job["error"]["code"] == "busy"
        assert c.get("/guide-camera").json()["streaming"] is True
        job = wait_job(
            c, c.post("/system/guide-camera/shutdown?force=true", headers=TO).json()["id"]
        )
        assert job["state"] == "done" and job["result"]["opstate"] == "shutdown"
        assert c.get("/guide-camera").json()["streaming"] is False


def test_the_wilds_umbrella_shows_in_the_activities_during_startup(make_client):
    with make_client('opmode = "ldt-production"\n[broker]\nhost = "joe"\n') as c:
        cam = guide_camera.get_guide_camera()
        cam.life._startup = lambda: time.sleep(0.3)
        job = c.post("/system/startup", headers=TO).json()
        time.sleep(0.1)
        acts = c.get("/status").json()["activities"]
        assert [a["name"] for a in acts["wilds"]] == ["StartingUp"]
        assert [a["name"] for a in acts["guide-camera"]] == ["StartingUp"]
        wait_job(c, job["id"])
        assert "wilds" not in c.get("/status").json()["activities"]


def test_an_unstarted_camera_is_not_touched_by_a_settings_change_but_startup_applies_it(
    make_client,
):
    with make_client('opmode = "ldt-production"\n[broker]\nhost = "joe"\n') as c:
        cam = guide_camera.get_guide_camera()
        h = hold_lease(c)
        r = c.put("/guide-camera/cooler", json={"enabled": True, "setpoint_c": -20}, headers=h)
        assert r.status_code == 200 and r.json()["setpoint_c"] == -20
        before = cam._read_temperature()
        assert before > 0  # still ambient: nothing was written to the hardware
        wait_job(c, c.post("/system/startup", headers=TO).json()["id"])
        assert cam.cooler().enabled and cam.cooler().setpoint_c == -20
