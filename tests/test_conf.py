import logging
import threading

import pytest
from pydantic import ValidationError

from wilds.control import conf as confmod
from wilds.control.conf import ConfHandle, ConfModel, ConfRegistry, by_opmode, setting
from wilds.control.errors import ApiError
from wilds.control.jobs import StartedBy
from wilds.control.settings_store import SettingsConflict, SettingsStore, StoreError

ARIE = StartedBy(name="Arie", code="K7-42")
NOA = StartedBy(name="Noa", code="B2-07")


class CamConf(ConfModel):
    exposure_s: float = setting(
        0.5, unit="s", label="Exposure", help="Exposure time", ge=0.001, le=60
    )
    gain: int = setting(100, ge=0, le=500)
    presets: dict[str, float] = setting({"short": 0.1})
    cooler_enabled: bool = by_opmode(development=True, production=False)


class CamCal(ConfModel):
    plate_scale_arcsec_px: float | None = setting(None, unit="arcsec/px", label="Plate scale")
    slit_pixel_x: float | None = None


@pytest.fixture
def store(tmp_path):
    return SettingsStore(tmp_path / "conf.db")


@pytest.fixture
def conf(store):
    return ConfHandle("guide-camera", "conf", CamConf, store)


@pytest.fixture
def cal(store):
    return ConfHandle("guide-camera", "cal", CamCal, store)


# -- defaults and the model -------------------------------------------------------------


def test_nothing_saved_gives_the_defaults_at_version_zero(conf):
    assert conf.current == CamConf() and conf.version == 0 and conf.history() == []


def test_models_are_frozen_and_ignore_keys_they_do_not_know():
    with pytest.raises(ValidationError):
        CamConf().gain = 5
    assert CamConf.model_validate({"gain": 7, "retired_key": 1}).gain == 7


def test_fields_describe_themselves_for_the_gui_form():
    props = CamConf.model_json_schema()["properties"]
    assert props["exposure_s"]["title"] == "Exposure" and props["exposure_s"]["unit"] == "s"
    assert props["exposure_s"]["minimum"] == 0.001 and props["exposure_s"]["maximum"] == 60
    assert props["exposure_s"]["description"] == "Exposure time"
    assert "unit" not in props["gain"]


def test_a_default_can_depend_on_the_opmode(config_file):
    config_file('opmode = "development"\n')
    assert CamConf().cooler_enabled is True
    config_file('opmode = "ldt-production"\n')
    assert CamConf().cooler_enabled is False


# -- saving ------------------------------------------------------------------------------


def test_a_save_is_recorded_and_made_live(conf):
    entry = conf.save(CamConf(gain=250), who=ARIE, reason="faint field")
    assert (
        entry.version == 1
        and entry.who == "Arie"
        and entry.changed == ["cooler_enabled", "exposure_s", "gain", "presets"]
    )
    assert conf.current.gain == 250 and conf.version == 1
    assert conf.history()[0].reason == "faint field"


def test_saving_what_is_already_there_changes_nothing(conf):
    conf.save(CamConf(gain=250), who=ARIE)
    seen = []
    conf.on_change(lambda *a: seen.append(a))
    assert conf.save(CamConf(gain=250), who=NOA) is None
    assert conf.version == 1 and seen == []


def test_only_the_handles_own_model_is_accepted(conf):
    with pytest.raises(TypeError):
        conf.save(CamCal(), who=ARIE)


def test_a_patch_changes_some_fields_and_validates_the_whole(conf):
    conf.save(CamConf(gain=250), who=ARIE)
    conf.patch({"exposure_s": 2.0}, who=NOA)
    assert conf.current.exposure_s == 2.0 and conf.current.gain == 250  # the rest stayed
    with pytest.raises(ValidationError):
        conf.patch({"exposure_s": 1000}, who=NOA)
    with pytest.raises(ValidationError):
        conf.patch({"gain": "lots"}, who=NOA)
    assert conf.current.exposure_s == 2.0 and conf.version == 2  # nothing half-applied


def test_a_stale_form_is_refused_and_nothing_changes(conf):
    conf.save(CamConf(gain=1), who=ARIE)
    conf.save(CamConf(gain=2), who=NOA)
    seen = []
    conf.on_change(lambda *a: seen.append(a))
    with pytest.raises(SettingsConflict):
        conf.save(CamConf(gain=3), who=ARIE, expected_version=1)
    assert conf.current.gain == 2 and conf.version == 2 and seen == []
    assert conf.save(CamConf(gain=3), who=ARIE, expected_version=2).version == 3


# -- live, not copied ----------------------------------------------------------------------


def test_everyone_holding_the_handle_sees_a_save_but_a_snapshot_stays_put(conf):
    held = conf
    snap = conf.snapshot()
    conf.save(CamConf(gain=300), who=ARIE)
    assert held.current.gain == 300
    assert snap.gain == 100  # a job's snapshot does not change under it


def test_a_new_handle_on_the_same_store_loads_what_was_saved(store, conf):
    conf.save(CamConf(gain=300), who=ARIE)
    again = ConfHandle("guide-camera", "conf", CamConf, store)
    assert again.current.gain == 300 and again.version == 1


def test_a_stored_section_that_no_longer_validates_stops_the_component_loudly(store):
    ConfHandle("guide-camera", "conf", CamConf, store).save(CamConf(gain=5), who=ARIE)
    import sqlite3

    with sqlite3.connect(store.path) as db:
        db.execute("UPDATE section SET json = '{\"gain\": \"x\"}' WHERE component = 'guide-camera'")
    with pytest.raises(StoreError):
        ConfHandle("guide-camera", "conf", CamConf, store)


# -- listeners --------------------------------------------------------------------------------


def test_listeners_hear_the_old_and_new_models_and_the_changed_keys_after_the_swap(conf):
    heard = []
    conf.on_change(
        lambda old, new, changed: heard.append((old.gain, new.gain, changed, conf.current.gain))
    )
    conf.save(CamConf(gain=250), who=ARIE)
    conf.patch({"gain": 260}, who=ARIE)
    assert (
        heard[0][:2] == (100, 250) and "gain" in heard[0][2] and heard[0][3] == 250
    )  # live already
    assert heard[1] == (250, 260, ["gain"], 260)


def test_a_failing_listener_is_logged_and_does_not_stop_the_others_or_the_save(conf, caplog):
    heard = []

    def broken(*_):
        raise RuntimeError("camera unplugged")

    conf.on_change(broken)
    conf.on_change(lambda *a: heard.append(a))
    with caplog.at_level(logging.ERROR, logger="wilds.control.conf"):
        assert conf.save(CamConf(gain=9), who=ARIE).version == 1
    assert conf.current.gain == 9 and len(heard) == 1
    assert any("change listener failed" in r.message for r in caplog.records)


def test_subscribing_twice_listens_once(conf):
    heard = []

    def listener(*a):
        heard.append(a)

    conf.on_change(listener)
    conf.on_change(listener)
    conf.save(CamConf(gain=9), who=ARIE)
    assert len(heard) == 1


# -- revert -----------------------------------------------------------------------------------


def test_revert_makes_an_old_version_live_as_a_new_one_and_tells_the_listeners(conf):
    conf.save(CamConf(gain=1), who=ARIE)
    conf.save(CamConf(gain=2), who=ARIE)
    heard = []
    conf.on_change(lambda old, new, changed: heard.append((old.gain, new.gain)))
    entry = conf.revert(1, who=NOA)
    assert (
        entry.version == 3
        and entry.reverted_from == 1
        and conf.current.gain == 1
        and conf.version == 3
    )
    assert heard == [(2, 1)]
    assert conf.revert(1, who=NOA) is None  # already those values


# -- calibration ------------------------------------------------------------------------------


def test_a_calibration_value_not_yet_measured_is_refused_with_a_clear_error(cal):
    with pytest.raises(ApiError) as e:
        cal.require("plate_scale_arcsec_px")
    assert e.value.status == 409 and e.value.body.code == "not_calibrated"
    assert e.value.body.details == {"component": "guide-camera", "field": "plate_scale_arcsec_px"}
    assert "has not been calibrated" in e.value.body.message
    cal.save(CamCal(plate_scale_arcsec_px=0.304), who=ARIE, reason="astrometry on M13")
    assert cal.require("plate_scale_arcsec_px") == 0.304


def test_conf_and_cal_are_separate_sections_with_their_own_histories(conf, cal):
    conf.save(CamConf(gain=1), who=ARIE)
    conf.save(CamConf(gain=2), who=ARIE)
    cal.save(CamCal(slit_pixel_x=511.0), who=NOA)
    assert (conf.section, cal.section) == ("guide-camera", "guide-camera.cal")
    assert (conf.version, cal.version) == (2, 1)
    assert [e.who for e in cal.history()] == ["Noa"]


# -- threads ------------------------------------------------------------------------------------


def test_concurrent_saves_end_with_the_live_model_equal_to_the_latest_version(conf):
    errors = []

    def go(n):
        try:
            conf.save(CamConf(gain=n), who=StartedBy(name=f"w{n}", code="X0-00"))
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=go, args=(n,)) for n in range(1, 31)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert errors == [] and conf.version == 30
    assert conf.current.gain == conf.history(1)[0].values["gain"]  # live == the newest stored


# -- the registry -------------------------------------------------------------------------------


def test_the_registry_hands_out_one_handle_per_component_and_slice(store):
    reg = ConfRegistry()
    conf = reg.register("guide-camera", "conf", CamConf, store)
    assert (
        reg.register("guide-camera", "conf", CamConf, store) is conf
    )  # rebuilt component, same handle
    cal = reg.register("guide-camera", "cal", CamCal, store)
    reg.register("adc", "conf", CamConf, store)
    assert reg.get("guide-camera") is conf and reg.get("guide-camera", "cal") is cal
    assert reg.components() == ["adc", "guide-camera"] and reg.slices("guide-camera") == [
        "cal",
        "conf",
    ]
    with pytest.raises(ValueError):
        reg.register("guide-camera", "conf", CamCal, store)
    with pytest.raises(KeyError, match="no cal registered"):
        reg.get("adc", "cal")
    reg.clear()
    assert reg.components() == []


def test_the_module_registry_exists_and_uses_the_one_store(tmp_path):
    assert isinstance(confmod.confs, ConfRegistry)


# -- the component's say (before_change) ---------------------------------------------------------


def test_a_component_can_refuse_a_change_and_then_nothing_is_recorded_or_announced(conf):
    heard = []
    conf.on_change(lambda *a: heard.append(a))

    def busy(old, new, changed):
        if "gain" in changed:
            raise ApiError(409, "busy", "the camera is exposing")

    conf.before_change(busy)
    conf.save(CamConf(exposure_s=2.0), who=ARIE)  # not refused
    with pytest.raises(ApiError) as e:
        conf.save(CamConf(exposure_s=2.0, gain=400), who=ARIE)
    assert e.value.status == 409
    with pytest.raises(ApiError):
        conf.patch({"gain": 400}, who=ARIE)
    assert conf.current.gain == 100 and conf.version == 1 and len(conf.history()) == 1
    assert len(heard) == 1  # only the first save was announced


def test_the_hook_hears_the_old_and_new_models_and_the_changed_keys_before_the_swap(conf):
    seen = []
    conf.before_change(
        lambda old, new, changed: seen.append((old.gain, new.gain, changed, conf.current.gain))
    )
    conf.patch({"gain": 250}, who=ARIE)
    assert seen == [(100, 250, ["gain"], 100)]  # still the old model while it decides


def test_a_change_that_changes_nothing_does_not_ask(conf):
    asked = []
    conf.before_change(lambda *a: asked.append(a))
    assert conf.save(CamConf(), who=ARIE) is None
    assert asked == []


def test_a_hook_that_applies_to_hardware_and_fails_leaves_the_setting_unrecorded(conf):
    device = {"gain": 100}

    def apply(old, new, changed):
        if "gain" in changed:
            if new.gain > 300:
                raise OSError("the camera does not answer")
            device["gain"] = new.gain

    conf.before_change(apply)
    conf.patch({"gain": 200}, who=ARIE)
    with pytest.raises(OSError):
        conf.patch({"gain": 400}, who=ARIE)
    assert (
        device["gain"] == 200 == conf.current.gain and conf.version == 1
    )  # store, live and device agree


def test_a_refused_revert_is_not_recorded_either(conf):
    conf.save(CamConf(gain=1), who=ARIE)
    conf.save(CamConf(gain=2), who=ARIE)

    def never(old, new, changed):
        raise ApiError(409, "busy", "not now")

    conf.before_change(never)
    with pytest.raises(ApiError):
        conf.revert(1, who=NOA)
    assert conf.current.gain == 2 and conf.version == 2


def test_reverting_to_a_version_that_never_existed_is_refused(conf):
    from wilds.control.settings_store import UnknownVersion

    conf.save(CamConf(gain=1), who=ARIE)
    with pytest.raises(UnknownVersion):
        conf.revert(9, who=ARIE)


def test_the_registry_adopts_a_handle_a_component_built_and_replaces_an_older_one(store):
    reg = ConfRegistry()
    first = ConfHandle("adc", "conf", CamConf, store)
    second = ConfHandle("adc", "conf", CamConf, store)
    reg.add(first)
    reg.add(second)
    assert reg.get("adc") is second


def test_patch_keeps_the_rest_of_a_nested_group_that_is_sent_in_part(store):
    class Inner(ConfModel):
        a: int = 1
        b: int = 2

    class Outer(ConfModel):
        first: Inner = Inner()
        second: Inner = Inner()

    conf = ConfHandle("deep", "conf", Outer, store)
    conf.patch({"first": {"a": 10}}, who=ARIE)
    assert conf.current.first == Inner(a=10, b=2) and conf.current.second == Inner()
