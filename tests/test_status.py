from fastapi.testclient import TestClient

from wilds.control.app import create_app


def test_status_snapshot_has_the_shape_the_gui_expects():
    r = TestClient(create_app()).get("/status")
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"control", "tcs", "time", "to", "warnings", "actions"}
    assert body["control"]["holder"] is None and body["control"]["engineering"] is False
    assert body["tcs"]["stale"] is True


def test_status_carries_the_servers_clock_and_the_ldt_zone():
    import time

    before = time.time_ns() // 1_000_000
    body = TestClient(create_app()).get("/status").json()
    after = time.time_ns() // 1_000_000
    assert before <= body["time"]["utc_ms"] <= after
    assert body["time"]["zone"] == "America/Phoenix"  # LDT: no daylight saving
