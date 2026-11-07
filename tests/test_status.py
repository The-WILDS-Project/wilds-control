from fastapi.testclient import TestClient

from wilds.control.app import create_app


def test_status_snapshot_has_the_shape_the_gui_expects():
    r = TestClient(create_app()).get("/status")
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {
        "opmode",
        "control",
        "tcs",
        "time",
        "to",
        "activities",
        "guide_camera",
        "jobs",
        "warnings",
        "actions",
    }
    assert body["control"]["holder"] is None and body["control"]["engineering"] is False
    assert body["tcs"]["stale"] is True


def test_status_carries_the_servers_clock_and_the_ldt_zone():
    import time

    before = time.time_ns() // 1_000_000
    body = TestClient(create_app()).get("/status").json()
    after = time.time_ns() // 1_000_000
    assert before <= body["time"]["utc_ms"] <= after
    assert body["time"]["zone"] == "America/Phoenix"  # LDT: no daylight saving


def test_status_lists_running_activities_by_component():
    from enum import Enum

    from wilds.control.activities import registry

    class Move(Enum):
        Slewing = 1
        Cancelling = 2

    registry.clear()
    try:
        mount = registry.register("mount", Move)
        client = TestClient(create_app())
        assert client.get("/status").json()["activities"] == {}
        mount.start(Move.Slewing, details=["to M31"])
        got = client.get("/status").json()["activities"]
        assert list(got) == ["mount"]
        assert got["mount"][0]["name"] == "Slewing" and got["mount"][0]["details"] == ["to M31"]
        mount.end(Move.Slewing)
        assert client.get("/status").json()["activities"] == {}
    finally:
        registry.clear()
