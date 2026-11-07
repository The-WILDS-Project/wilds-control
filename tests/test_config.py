import logging

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from wilds.control import config, lease
from wilds.control.app import create_app
from wilds.control.routers.control import require_engineering


def test_without_a_file_the_defaults_apply_and_warn():
    assert config.check_engineering("physics")
    assert not config.check_engineering("nope")
    assert not config.check_to("anything")  # TO endpoints stay closed until configured
    assert config.get().warnings()


def test_a_file_sets_the_passphrases_and_clears_the_warning(config_file, passphrases_file):
    passphrases_file('[passphrases]\nengineering = "quarks"\nto = "scope"\n')
    assert config.check_engineering("quarks") and not config.check_engineering("physics")
    assert config.check_to("scope")
    assert config.get().warnings() == []


def test_a_changed_file_applies_without_a_restart(config_file, passphrases_file):
    passphrases_file('[passphrases]\nengineering = "one"\n')
    assert config.check_engineering("one")
    passphrases_file('[passphrases]\nengineering = "two"\n')
    assert config.check_engineering("two") and not config.check_engineering("one")


def test_an_invalid_file_is_rejected_whole_and_the_running_config_kept(
    config_file, caplog, passphrases_file
):
    passphrases_file('[passphrases]\nengineering = "one"\n')
    assert config.check_engineering("one")
    with caplog.at_level(logging.ERROR):
        passphrases_file("[passphrases\nbroken")
        assert config.check_engineering("one")
        passphrases_file("[passphrases]\nengineering = 12\n")  # wrong type
        assert config.check_engineering("one")
    assert "rejected" in caplog.text


def test_status_carries_the_default_passphrase_warning(config_file, passphrases_file):
    client = TestClient(create_app())
    assert any("default" in w for w in client.get("/status").json()["warnings"])
    passphrases_file('[passphrases]\nengineering = "quarks"\n')
    assert client.get("/status").json()["warnings"] == []


@pytest.fixture
def protected():
    app = FastAPI()

    @app.post("/calibrate", dependencies=[Depends(require_engineering)])
    def calibrate() -> dict[str, bool]:
        return {"ok": True}

    app.include_router(create_app().router)
    lease.state.sessions.clear()
    lease.state.holder_id = lease.state.token = None
    lease.state.engineering = False
    return TestClient(app)


def test_calibration_needs_the_lease_in_engineering_mode(protected, config_file, passphrases_file):
    passphrases_file('[passphrases]\nto = "t"\n')
    assert protected.post("/calibrate").status_code == 403  # no token

    s = protected.post("/sessions", json={"name": "A"}).json()
    sid = {"X-Session-Id": s["session_id"]}
    protected.post(f"/control/requests/{s['code']}/grant", headers={"X-TO-Passphrase": "t"})
    token = protected.post(f"/sessions/{s['session_id']}/heartbeat").json()["lease_token"]
    r = protected.post("/calibrate", headers={"X-Lease-Token": token})
    assert r.status_code == 403 and "Engineering" in r.json()["detail"]  # TO grant is not enough

    assert (
        protected.post(
            "/control/engineering", json={"passphrase": "physics"}, headers=sid
        ).status_code
        == 200
    )
    token = protected.post(f"/sessions/{s['session_id']}/heartbeat").json()["lease_token"]
    assert protected.post("/calibrate", headers={"X-Lease-Token": token}).status_code == 200


class Clock:
    now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def presence(monkeypatch, config_file, passphrases_file):
    clock = Clock()
    clock.now = 1000.0
    monkeypatch.setattr(lease.state, "clock", clock)
    lease.state.to_seen = None
    lease.state.to_logged_online = False
    passphrases_file('[passphrases]\nto = "t"\n')
    config_file("[presence]\nto_timeout_s = 60\n")
    return clock


def test_the_to_is_online_while_a_to_page_polls(presence, caplog):
    client = TestClient(create_app())
    assert client.get("/status").json()["to"]["online"] is False
    with caplog.at_level(logging.INFO, logger="wilds.control.audit"):
        client.get("/control/sessions", headers={"X-TO-Passphrase": "t"})
        assert client.get("/status").json()["to"]["online"] is True
        presence.now += 59
        assert client.get("/status").json()["to"]["online"] is True
        presence.now += 2
        assert client.get("/status").json()["to"]["online"] is False
        client.get("/control/sessions", headers={"X-TO-Passphrase": "t"})
        assert client.get("/status").json()["to"]["online"] is True
    log = [r.message for r in caplog.records]
    assert log.count("Telescope Operator online") == 2
    assert log.count("Telescope Operator offline (no contact)") == 1


def test_opening_the_page_without_the_passphrase_does_not_make_the_to_online(presence):
    client = TestClient(create_app())
    assert client.get("/control/sessions").status_code == 403
    assert client.get("/control/sessions", headers={"X-TO-Passphrase": "wrong"}).status_code == 403
    assert client.get("/status").json()["to"]["online"] is False


def test_the_ldt_zone_is_configurable_and_a_bad_zone_is_rejected(config_file, caplog):
    config_file('[site]\ntimezone = "America/Denver"\n')
    assert config.get().site.timezone == "America/Denver"
    with caplog.at_level(logging.ERROR):
        config_file('[site]\ntimezone = "Mars/Olympus"\n')
        assert config.get().site.timezone == "America/Denver"  # rejected whole, kept
    assert "unknown time zone" in caplog.text


def test_a_passphrases_section_in_config_toml_is_ignored_with_a_warning(
    config_file, passphrases_file
):
    config_file('[passphrases]\nengineering = "sneaky"\nto = "sneaky"\n')
    assert config.check_engineering("physics") and not config.check_engineering("sneaky")
    assert not config.check_to("sneaky")
    assert any("passphrases.toml" in w for w in config.get().warnings())
    passphrases_file('[passphrases]\nengineering = "real"\nto = "scope"\n')
    assert config.check_engineering("real") and config.check_to("scope")


def test_each_file_applies_on_its_own(config_file, passphrases_file):
    passphrases_file('[passphrases]\nto = "t"\n')
    config_file('opmode = "production"\n')
    assert config.get().opmode == "production" and config.check_to("t")
    config_file('opmode = "development"\n')  # the passphrases file is untouched
    assert config.get().opmode == "development" and config.check_to("t")
    passphrases_file('[passphrases]\nto = "u"\n')  # and the other way round
    assert config.get().opmode == "development" and config.check_to("u")


def test_the_passphrases_file_holds_nothing_but_the_section(passphrases_file, caplog):
    passphrases_file('[passphrases]\nengineering = "one"\n')
    assert config.check_engineering("one")
    with caplog.at_level(logging.ERROR):
        passphrases_file('opmode = "production"\n[passphrases]\nengineering = "two"\n')
        assert config.check_engineering("one")  # rejected whole, the running one kept
    assert "rejected" in caplog.text


def test_a_broken_config_toml_keeps_the_passphrases_too(config_file, passphrases_file, caplog):
    passphrases_file('[passphrases]\nengineering = "one"\n')
    assert config.check_engineering("one")
    with caplog.at_level(logging.ERROR):
        config_file("opmode = ")
        assert config.check_engineering("one")
    assert "rejected" in caplog.text
