import pytest
from fastapi.testclient import TestClient

from wilds.control import lease, provenance
from wilds.control.app import create_app


@pytest.fixture
def client():
    lease.state.sessions.clear()
    lease.state.holder_id = lease.state.token = None
    return TestClient(create_app())


def test_origin_label_from_the_configured_table(monkeypatch):
    monkeypatch.setenv("WILDS_CONTROL_ORIGINS", "10.0.0.0/8=Weizmann VPN;192.168.7.0/24=LDT")
    assert provenance.classify("10.2.3.4") == ("Weizmann VPN", False)
    assert provenance.classify("192.168.7.9") == ("LDT", False)
    assert provenance.classify("8.8.8.8") == (None, False)


def test_loopback_is_this_computer():
    assert provenance.classify("127.0.0.1") == ("this computer", True)


def test_garbage_in_the_origin_table_is_ignored(monkeypatch):
    monkeypatch.setenv("WILDS_CONTROL_ORIGINS", "nonsense;10.0.0.0/8=VPN;bad/net=x")
    assert provenance.classify("10.1.1.1") == ("VPN", False)


def test_browser_strings_are_cleaned_and_bounded():
    assert provenance.clean("a\nb\x00c") == "abc"
    assert len(provenance.clean("x" * 500)) == 120
    assert provenance.clean("") is None


def test_session_carries_provenance_and_it_reaches_the_tooltip_fields(client):
    hints = {"timezone": "Asia/Jerusalem", "page_host": "wilds.lowell.edu"}
    r = client.post(
        "/sessions",
        json={"name": "Arie", "hints": hints},
        headers={"User-Agent": "TestBrowser/1.0"},
    )
    body = r.json()
    assert "Address: testclient" in body["provenance"]
    assert "Browser: TestBrowser/1.0" in body["provenance"]
    assert "Time zone: Asia/Jerusalem" in body["provenance"]
    assert "Opened as: wilds.lowell.edu" in body["provenance"]

    to = {"X-TO-Passphrase": "p"}
    import os

    os.environ["WILDS_TO_PASSPHRASE"] = "p"
    try:
        sess = client.get("/control/sessions", headers=to).json()
        info = sess["viewers"][0]
        assert info["provenance"] == body["provenance"]
        assert info["details"]["hints"]["timezone"] == "Asia/Jerusalem"
        client.post(f"/control/requests/{body['code']}/grant", headers=to)
        ctl = client.get("/status").json()["control"]
        assert ctl["holder_code"] == body["code"] and "Address:" in ctl["holder_info"]
    finally:
        del os.environ["WILDS_TO_PASSPHRASE"]


def test_summary_lines_for_a_vnc_desktop():
    p = provenance.Provenance(
        ip="127.0.0.1",
        first_ip="127.0.0.1",
        origin="this computer",
        local=True,
        display=":1",
        vnc_viewers=["10.0.3.7"],
    )
    assert "Desktop: VNC :1 (best effort: 10.0.3.7)" in p.summary()
    p.display = ":0"
    assert "Desktop: console (:0)" in p.summary()


def test_vnc_detection_degrades_quietly():
    assert provenance.find_vnc(None, None) == (None, [])
    assert provenance.find_vnc(1, 2) == (None, [])  # no such connection


def test_address_change_keeps_the_first_one(client):
    r = client.post("/sessions", json={"name": "Arie"}).json()
    s = lease.state.sessions[r["session_id"]]
    provenance.moved(s.prov, "10.9.9.9")
    assert s.prov.ip == "10.9.9.9" and s.prov.first_ip == "testclient"
    assert "First seen from: testclient" in s.prov.summary()
