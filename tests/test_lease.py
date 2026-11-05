import pytest
from fastapi.testclient import TestClient

from wilds.control import lease
from wilds.control.app import create_app


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock(monkeypatch):
    c = Clock()
    lease.state.sessions.clear()
    lease.state.holder_id = lease.state.token = None
    lease.state.engineering = False
    monkeypatch.setattr(lease.state, "clock", c)
    monkeypatch.setenv("WILDS_TO_PASSPHRASE", "to-pass")
    monkeypatch.setenv("WILDS_ENGINEERING_PASSPHRASE", "eng-pass")
    return c


@pytest.fixture
def client(clock):
    return TestClient(create_app())


TO = {"X-TO-Passphrase": "to-pass"}


def new_session(client, name="Arie (Rehovot)"):
    r = client.post("/sessions", json={"name": name})
    assert r.status_code == 200
    return r.json()


def sid(s):
    return {"X-Session-Id": s["session_id"]}


def test_session_gets_secret_id_and_short_code(client):
    s = new_session(client)
    assert len(s["session_id"]) == 36
    assert len(s["code"]) == 5 and s["code"][2] == "-"


def test_blank_name_is_refused(client):
    assert client.post("/sessions", json={"name": "   "}).status_code == 422


def test_viewer_cap(client, monkeypatch):
    monkeypatch.setattr(lease, "MAX_SESSIONS", 2)
    new_session(client)
    new_session(client)
    r = client.post("/sessions", json={"name": "third"})
    assert r.status_code == 503 and "limit" in r.json()["detail"]


def test_request_grant_gives_holder_a_token(client):
    s = new_session(client)
    assert client.post("/control/requests", headers=sid(s)).status_code == 202
    listing = client.get("/control/sessions", headers=TO).json()
    assert [r["code"] for r in listing["requests"]] == [s["code"]]
    assert client.post(f"/control/requests/{s['code']}/grant", headers=TO).status_code == 200
    hb = client.post(f"/sessions/{s['session_id']}/heartbeat").json()
    assert hb["holder"] and hb["lease_token"]
    assert client.get("/status").json()["control"]["holder"] == "Arie (Rehovot)"


def test_non_holder_never_sees_the_token(client):
    a, b = new_session(client, "A"), new_session(client, "B")
    client.post(f"/control/requests/{a['code']}/grant", headers=TO)
    hb = client.post(f"/sessions/{b['session_id']}/heartbeat").json()
    assert not hb["holder"] and hb["lease_token"] is None


def test_to_endpoints_need_the_passphrase(client):
    s = new_session(client)
    assert client.get("/control/sessions").status_code == 403
    assert client.post(f"/control/requests/{s['code']}/grant").status_code == 403
    assert client.post("/control/revoke", headers={"X-TO-Passphrase": "nope"}).status_code == 403


def test_to_endpoints_are_closed_when_no_passphrase_is_configured(client, monkeypatch):
    monkeypatch.delenv("WILDS_TO_PASSPHRASE")
    assert client.get("/control/sessions", headers={"X-TO-Passphrase": ""}).status_code == 403


def test_grant_moves_control_to_the_new_holder(client):
    a, b = new_session(client, "A"), new_session(client, "B")
    client.post(f"/control/requests/{a['code']}/grant", headers=TO)
    token_a = client.post(f"/sessions/{a['session_id']}/heartbeat").json()["lease_token"]
    client.post(f"/control/requests/{b['code']}/grant", headers=TO)
    assert not client.post(f"/sessions/{a['session_id']}/heartbeat").json()["holder"]
    with pytest.raises(lease.LeaseError):
        lease.state.check_token(token_a)


def test_release_and_revoke(client):
    a = new_session(client)
    client.post(f"/control/requests/{a['code']}/grant", headers=TO)
    assert client.delete("/control", headers=sid(new_session(client, "B"))).status_code == 403
    assert client.delete("/control", headers=sid(a)).status_code == 200
    client.post(f"/control/requests/{a['code']}/grant", headers=TO)
    assert client.post("/control/revoke", headers=TO).status_code == 200
    assert client.get("/status").json()["control"]["holder"] is None


def test_engineering_needs_the_passphrase_and_marks_status(client):
    s = new_session(client)
    r = client.post("/control/engineering", json={"passphrase": "bad"}, headers=sid(s))
    assert r.status_code == 403
    r = client.post("/control/engineering", json={"passphrase": "eng-pass"}, headers=sid(s))
    assert r.status_code == 200
    ctl = client.get("/status").json()["control"]
    assert ctl["holder"] == "Arie (Rehovot)" and ctl["engineering"] is True


def test_dead_holder_lapses_the_lease(client, clock):
    a = new_session(client)
    client.post(f"/control/requests/{a['code']}/grant", headers=TO)
    clock.now += lease.SESSION_TIMEOUT_S + 1
    assert client.get("/status").json()["control"]["holder"] is None
    assert client.post(f"/sessions/{a['session_id']}/heartbeat").status_code == 404


def test_heartbeat_keeps_a_session_alive(client, clock):
    a = new_session(client)
    for _ in range(5):
        clock.now += lease.SESSION_TIMEOUT_S - 1
        assert client.post(f"/sessions/{a['session_id']}/heartbeat").status_code == 200


def test_refuse_clears_the_request(client):
    s = new_session(client)
    client.post("/control/requests", headers=sid(s))
    client.post(f"/control/requests/{s['code']}/refuse", json={"reason": "busy"}, headers=TO)
    assert client.get("/control/sessions", headers=TO).json()["requests"] == []


def test_unknown_session_header_is_404(client):
    assert client.post("/control/requests", headers={"X-Session-Id": "nope"}).status_code == 404
    assert client.post("/control/requests").status_code == 404
