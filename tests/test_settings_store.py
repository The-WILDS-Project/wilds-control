import sqlite3
import threading

import pytest
from pydantic import BaseModel, Field

from wilds.control import config, paths
from wilds.control.jobs import StartedBy
from wilds.control.settings_store import (
    SettingsConflict,
    SettingsStore,
    StoreError,
    UnknownVersion,
    get_store,
)

ARIE = StartedBy(name="Arie", code="K7-42")
NOA = StartedBy(name="Noa", code="B2-07")


class Slit(BaseModel):
    pixel_x: float = 512.0
    pixel_y: float = 300.0
    presets: dict[str, float] = Field(default_factory=lambda: {"narrow": 0.5})
    note: str | None = None


class Clock:
    def __init__(self) -> None:
        self.now = 1_000_000.0

    def __call__(self) -> float:
        self.now += 1
        return self.now


@pytest.fixture
def store(tmp_path):
    return SettingsStore(tmp_path / "db" / "settings.db", clock=Clock())


def test_nothing_saved_gives_the_models_defaults_at_version_zero(store):
    assert store.get("slit", Slit) == Slit()
    assert store.version("slit") == 0 and store.components() == [] and store.history("slit") == []


def test_a_saved_section_comes_back_validated_at_the_next_version(store):
    entry = store.put(
        "slit", Slit(pixel_x=500.5, presets={"wide": 2.0}), who=ARIE, reason="after the lamp run"
    )
    assert entry.version == 1 and entry.who == "Arie" and entry.code == "K7-42"
    got = store.get("slit", Slit)
    assert got.pixel_x == 500.5 and got.presets == {"wide": 2.0} and got.note is None
    assert store.version("slit") == 1 and store.components() == ["slit"]


def test_a_put_that_changes_nothing_records_nothing(store):
    store.put("slit", Slit(), who=ARIE)
    assert store.put("slit", Slit(), who=NOA) is None
    assert store.version("slit") == 1 and len(store.history("slit")) == 1


def test_history_is_newest_first_with_who_when_why_and_the_keys_changed(store):
    store.put("slit", Slit(), who=ARIE, reason="first")
    store.put("slit", Slit(pixel_x=510.0), who=NOA, reason="refit")
    store.put("slit", Slit(pixel_x=510.0, pixel_y=301.0, note="cloudy"), who=ARIE)
    h = store.history("slit")
    assert [e.version for e in h] == [3, 2, 1]
    assert [e.who for e in h] == ["Arie", "Noa", "Arie"]
    assert h[1].reason == "refit" and h[2].reason == "first" and h[0].reason is None
    assert h[0].changed == ["note", "pixel_y"] and h[1].changed == ["pixel_x"]
    assert h[2].changed == [
        "note",
        "pixel_x",
        "pixel_y",
        "presets",
    ]  # the first version: everything is new
    assert h[0].at_ms > h[1].at_ms > h[2].at_ms
    assert h[0].values["note"] == "cloudy"
    assert [e.version for e in store.history("slit", limit=2)] == [3, 2]


def test_revert_restores_an_old_version_as_a_new_one(store):
    store.put("slit", Slit(pixel_x=1.0), who=ARIE)
    store.put("slit", Slit(pixel_x=2.0), who=ARIE)
    entry = store.revert("slit", 1, Slit, who=NOA)
    assert entry.version == 3 and entry.reverted_from == 1 and entry.reason == "revert to version 1"
    assert store.get("slit", Slit).pixel_x == 1.0
    assert [e.version for e in store.history("slit")] == [3, 2, 1]  # it only grows
    assert store.history("slit")[0].who == "Noa"


def test_reverting_to_what_is_current_records_nothing(store):
    store.put("slit", Slit(pixel_x=1.0), who=ARIE)
    assert store.revert("slit", 1, Slit, who=NOA) is None and store.version("slit") == 1


def test_reverting_to_a_version_that_never_existed_is_refused(store):
    store.put("slit", Slit(), who=ARIE)
    with pytest.raises(UnknownVersion):
        store.revert("slit", 9, Slit, who=ARIE)
    with pytest.raises(UnknownVersion):
        store.revert("adc", 1, Slit, who=ARIE)


def test_a_form_that_saw_an_old_version_is_refused_not_overwritten(store):
    store.put("slit", Slit(), who=ARIE)
    store.put("slit", Slit(pixel_x=1.0), who=NOA)
    with pytest.raises(SettingsConflict) as e:
        store.put("slit", Slit(pixel_x=2.0), who=ARIE, expected_version=1)
    assert e.value.expected == 1 and e.value.actual == 2
    assert store.get("slit", Slit).pixel_x == 1.0  # untouched
    assert store.put("slit", Slit(pixel_x=2.0), who=ARIE, expected_version=2).version == 3
    with pytest.raises(SettingsConflict):  # nothing saved yet counts as version 0
        store.put("adc", Slit(), who=ARIE, expected_version=1)
    assert store.put("adc", Slit(), who=ARIE, expected_version=0).version == 1


def test_components_are_independent(store):
    store.put("slit", Slit(pixel_x=1.0), who=ARIE)
    store.put("adc", Slit(pixel_x=9.0), who=ARIE)
    store.put("slit", Slit(pixel_x=2.0), who=ARIE)
    assert (store.version("slit"), store.version("adc")) == (2, 1)
    assert store.components() == ["adc", "slit"]
    assert [e.version for e in store.history("adc")] == [1]


def test_it_survives_a_restart_and_a_second_opener(tmp_path):
    path = tmp_path / "s.db"
    SettingsStore(path).put("slit", Slit(pixel_x=7.0), who=ARIE)
    again = SettingsStore(path)
    assert again.get("slit", Slit).pixel_x == 7.0 and again.history("slit")[0].who == "Arie"
    again.put("slit", Slit(pixel_x=8.0), who=NOA)
    assert SettingsStore(path).version("slit") == 2


def test_a_key_added_to_the_model_since_is_defaulted(store):
    store.put("slit", Slit(), who=ARIE)

    class Newer(Slit):
        extra: int = 5

    assert store.get("slit", Newer).extra == 5


def test_a_stored_section_that_no_longer_validates_is_reported_not_guessed(store):
    store.put("slit", Slit(), who=ARIE)
    with sqlite3.connect(store.path) as db:
        db.execute(
            "UPDATE section SET json = ? WHERE component = 'slit'", ('{"pixel_x": "not a number"}',)
        )
    with pytest.raises(StoreError, match="no longer validates"):
        store.get("slit", Slit)


def test_a_database_from_a_newer_version_is_refused(tmp_path):
    path = tmp_path / "s.db"
    SettingsStore(path)
    with sqlite3.connect(path) as db:
        db.execute("PRAGMA user_version = 99")
    with pytest.raises(StoreError, match="schema 99"):
        SettingsStore(path)


def test_concurrent_writers_each_get_their_own_version_and_nothing_is_lost(store):
    errors = []

    def save(n):
        try:
            store.put("slit", Slit(pixel_x=float(n)), who=StartedBy(name=f"w{n}", code="X0-00"))
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=save, args=(n,)) for n in range(1, 21)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert errors == []
    assert [e.version for e in store.history("slit", limit=100)] == list(range(20, 0, -1))
    assert {e.values["pixel_x"] for e in store.history("slit", limit=100)} == {
        float(n) for n in range(1, 21)
    }


def test_two_processes_worth_of_writers_do_not_collide(tmp_path):
    a, b = (
        SettingsStore(tmp_path / "s.db"),
        SettingsStore(tmp_path / "s.db"),
    )  # separate locks, as two processes
    threads = [
        threading.Thread(target=lambda s=s, n=n: s.put("slit", Slit(pixel_x=float(n)), who=ARIE))
        for n, s in enumerate([a, b] * 10, start=1)
    ]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert a.version("slit") == 20 and len(b.history("slit", limit=100)) == 20


def test_a_failed_put_leaves_no_half_written_version(store):
    store.put("slit", Slit(), who=ARIE)

    class Bad(BaseModel):
        def model_dump(self, **kw):
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        store.put("slit", Bad(), who=ARIE)
    assert store.version("slit") == 1 and len(store.history("slit")) == 1


def test_the_location_comes_from_config_and_is_made(tmp_path, config_file):
    config_file(f'[store]\npath = "{tmp_path}/made/here/s.db"\n')
    SettingsStore(config.get().store.path).put("slit", Slit(), who=ARIE)
    assert (tmp_path / "made/here/s.db").is_file()


def test_the_one_store_is_made_once(tmp_path):
    assert get_store() is get_store()


def test_the_config_defaults_and_validation(config_file):
    default = config.get().store.path
    assert default.name == "settings.db" and default.is_relative_to(paths.var_lib())
    config_file('[store]\npath = "relative.db"\n')
    assert config.get().store.path == default  # rejected whole
