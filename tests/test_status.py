from fastapi.testclient import TestClient

from wilds.control.app import create_app


def test_status_snapshot_has_the_shape_the_gui_expects():
    r = TestClient(create_app()).get("/status")
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"control", "tcs", "actions"}
    assert body["control"] == {"holder": None, "engineering": False}
    assert body["tcs"]["stale"] is True
