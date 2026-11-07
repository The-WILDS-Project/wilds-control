import asyncio
from enum import Enum

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from wilds.control import lease
from wilds.control.activities import registry
from wilds.control.app import create_app
from wilds.control.errors import ApiError
from wilds.control.jobs import JobCancelled, StartedBy, manager

WHO = StartedBy(name="Arie", code="K7-42")


class Stage(Enum):
    Moving = 1
    Cancelling = 2


class Mount(Enum):
    Slewing = 1  # no Cancelling member


@pytest.fixture(autouse=True)
def clean():
    manager.reset()
    registry.clear()
    registry.register("stage", Stage)
    registry.register("mount", Mount)
    yield
    manager.reset()
    registry.clear()


async def settle():
    for _ in range(5):
        await asyncio.sleep(0)


async def gate_job(ctx):
    await ctx.sleep(5)
    return "finished"


async def test_a_job_runs_to_done_and_ends_its_activities():
    async def run(ctx):
        ctx.start_activity("stage", Stage.Moving, details=["to 3"])
        assert registry.get("stage").names == ["Moving"]
        assert ctx.job.activities == {"stage": ["Moving"]}
        ctx.progress(0.5, "halfway")
        return {"at": 3}

    job = manager.submit("move", ["stage"], run, started_by=WHO)
    assert job.state == "running" and job.started_by == WHO
    await settle()
    assert job.state == "done" and job.result == {"at": 3} and job.error is None
    assert job.progress.fraction == 0.5 and job.duration_s is not None
    assert registry.get("stage").is_idle() and job.activities == {}


async def test_a_failing_job_still_ends_its_activities():
    async def run(ctx):
        ctx.start_activity("stage", Stage.Moving)
        raise ApiError(409, "refused", "TCS said no", {"why": "locked out"})

    job = manager.submit("move", ["stage"], run, started_by=WHO)
    await settle()
    assert job.state == "failed"
    assert job.error.code == "refused" and job.error.details == {"why": "locked out"}
    assert registry.get("stage").is_idle()


async def test_an_unexpected_exception_is_an_internal_error():
    async def run(ctx):
        raise ValueError("boom")

    job = manager.submit("x", [], run, started_by=WHO)
    await settle()
    assert job.state == "failed" and job.error.code == "internal" and "boom" in job.error.message


async def test_a_second_job_on_a_busy_component_is_refused_with_the_blocker():
    first = manager.submit("a", ["stage"], gate_job, started_by=WHO)
    with pytest.raises(ApiError) as e:
        manager.submit("b", ["stage", "mount"], gate_job, started_by=WHO)
    assert e.value.status == 409 and e.value.body.code == "busy"
    assert e.value.body.details == {"job_id": first.id}
    other = manager.submit("c", ["mount"], gate_job, started_by=WHO)  # free component: fine
    assert other.state == "running"
    manager.cancel(first.id)
    manager.cancel(other.id)
    await settle()


async def test_a_queued_job_starts_when_the_component_frees_up():
    first = manager.submit("a", ["stage"], lambda ctx: ctx.sleep(0.05), started_by=WHO)
    second = manager.submit("b", ["stage"], gate_job, started_by=WHO, queue=True)
    assert first.state == "running" and second.state == "queued"
    await asyncio.sleep(0.15)
    assert first.state == "done" and second.state == "running"
    manager.cancel(second.id)
    await settle()


async def test_cancelling_a_queued_job_is_immediate():
    first = manager.submit("a", ["stage"], gate_job, started_by=WHO)
    second = manager.submit("b", ["stage"], gate_job, started_by=WHO, queue=True)
    manager.cancel(second.id)
    assert second.state == "cancelled" and second.error.code == "cancelled"
    manager.cancel(first.id)
    await settle()
    assert first.state == "cancelled" and second.state == "cancelled"


async def test_cancel_is_cooperative_and_raises_cancelling_until_the_job_stops():
    seen = []

    async def run(ctx):
        ctx.start_activity("stage", Stage.Moving)
        try:
            await ctx.sleep(5)
        except JobCancelled:
            seen.append(registry.get("stage").names)
            raise

    job = manager.submit("move", ["stage"], run, started_by=WHO)
    await settle()
    manager.cancel(job.id)
    assert job.cancel_requested and job.state == "running"  # not yet: the runner decides
    await settle()
    assert job.state == "cancelled"
    assert seen == [["Moving", "Cancelling"]]  # Cancelling was visible while it wound down
    assert registry.get("stage").is_idle()


async def test_cancel_skips_components_without_a_cancelling_member():
    job = manager.submit("slew", ["mount"], gate_job, started_by=WHO)
    manager.cancel(job.id)
    await settle()
    assert job.state == "cancelled" and registry.get("mount").is_idle()


async def test_a_job_that_finishes_despite_a_cancel_request_is_done():
    async def run(ctx):
        await asyncio.sleep(0)
        return "already finished"

    job = manager.submit("x", [], run, started_by=WHO)
    manager.cancel(job.id)
    await settle()
    assert job.state == "done"


async def test_cancelling_a_finished_job_changes_nothing():
    async def run(ctx):
        return 1

    job = manager.submit("x", [], run, started_by=WHO)
    await settle()
    assert manager.cancel(job.id).state == "done"


async def test_handover_drops_queued_jobs_and_keeps_running_ones():
    running = manager.submit("a", ["stage"], gate_job, started_by=WHO)
    queued = manager.submit("b", ["stage"], gate_job, started_by=WHO, queue=True)
    manager.handover()
    assert queued.state == "cancelled" and queued.error.message == "holder changed"
    assert running.state == "running"
    manager.cancel(running.id)
    await settle()


async def test_events_reach_subscribers_and_finished_jobs_are_remembered():
    q = manager.subscribe()

    async def run(ctx):
        ctx.progress(0.1)
        return 1

    job = manager.submit("x", [], run, started_by=WHO)
    await settle()
    states = []
    while not q.empty():
        states.append(q.get_nowait().state)
    assert states[0] == "running" and states[-1] == "done"
    assert [j.id for j in manager.snapshot()] == [job.id]
    assert manager.get(job.id) is job


async def test_retention_keeps_only_the_last_finished_jobs(monkeypatch):
    monkeypatch.setattr("wilds.control.jobs.KEEP_FINISHED", 3)

    async def run(ctx):
        return 1

    ids = [manager.submit("x", [], run, started_by=WHO).id for _ in range(5)]
    await settle()
    assert [j.id for j in manager.list()] == ids[:1:-1]  # j-5, j-4, j-3
    with pytest.raises(ApiError):
        manager.get(ids[0])


# -- HTTP -------------------------------------------------------------------------------


@pytest.fixture
def client(config_file, passphrases_file):
    passphrases_file('[passphrases]\nto = "to-pass"\nengineering = "eng-pass"\n')
    lease.state.sessions.clear()
    lease.state.holder_id = lease.state.token = None
    app: FastAPI = create_app()

    @app.post("/_start")
    async def start():
        job = manager.submit("test", ["stage"], gate_job, started_by=WHO)
        return {"job_id": job.id}

    with TestClient(app) as c:
        yield c


def hold_lease(client) -> dict[str, str]:
    s = client.post("/sessions", json={"name": "Arie"}).json()
    sid = {"X-Session-Id": s["session_id"]}
    client.post("/control/requests", headers=sid)
    client.post(f"/control/requests/{s['code']}/grant", headers={"X-TO-Passphrase": "to-pass"})
    token = client.post(f"/sessions/{s['session_id']}/heartbeat").json()["lease_token"]
    return {"X-Lease-Token": token}


def test_unknown_job_is_the_standard_error_shape(client):
    r = client.get("/jobs/j-99")
    assert r.status_code == 404
    assert r.json() == {"code": "not_found", "message": "no job j-99", "details": None}


def test_cancel_needs_the_lease_and_then_cancels(client):
    job_id = client.post("/_start").json()["job_id"]
    listing = client.get("/jobs").json()
    assert [j["id"] for j in listing] == [job_id] and listing[0]["state"] == "running"
    assert client.post(f"/jobs/{job_id}/cancel").status_code == 403

    token = hold_lease(client)
    r = client.post(f"/jobs/{job_id}/cancel", headers=token)
    assert r.status_code == 200 and r.json()["cancel_requested"] is True
    assert [j["id"] for j in client.get("/status").json()["jobs"]] == [job_id]


async def test_the_lease_changing_hands_drops_queued_jobs(config_file, passphrases_file):
    passphrases_file('[passphrases]\nto = "to-pass"\nengineering = "eng-pass"\n')
    create_app()  # registers manager.handover as a lease listener
    lease.state.sessions.clear()
    lease.state.holder_id = lease.state.token = None
    running = manager.submit("a", ["stage"], gate_job, started_by=WHO)
    queued = manager.submit("b", ["stage"], gate_job, started_by=WHO, queue=True)
    s = lease.state.create_session("Arie")
    lease.state.request(s)
    lease.state.grant(s.code)  # nobody -> Arie: the lease changes hands
    assert queued.state == "cancelled" and running.state == "running"
    manager.cancel(running.id)
    await settle()
