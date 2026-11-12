"""Where the files are: one root, the usual Linux hierarchy under it, overridable."""

import pytest

from wilds.control import config, paths


def test_everything_derives_from_the_root(wilds_root):
    assert paths.root() == wilds_root
    assert paths.config_file() == wilds_root / "etc" / "config.toml"
    assert paths.passphrases_file() == wilds_root / "etc" / "passphrases.toml"
    assert paths.settings_db() == wilds_root / "var" / "lib" / "settings.db"
    assert paths.staging_root() == wilds_root / "var" / "lib" / "staging"
    assert paths.guide_camera_calibration() == wilds_root / "var/lib/guide-camera/calibration"
    assert paths.var_log() == wilds_root / "var" / "log"
    assert paths.var_run() == wilds_root / "var" / "run"


def test_the_root_defaults_to_opt_wilds(monkeypatch):
    monkeypatch.delenv("WILDS_ROOT")
    assert paths.root().as_posix() == "/opt/wilds"
    assert paths.config_file().as_posix() == "/opt/wilds/etc/config.toml"


def test_the_root_must_be_absolute(monkeypatch):
    monkeypatch.setenv("WILDS_ROOT", "relative/dir")
    with pytest.raises(ValueError, match="WILDS_ROOT"):
        paths.root()


def test_the_data_mount_has_its_own_variable(monkeypatch, tmp_path):
    assert paths.lois_data_root().as_posix() == "/lois/data/dct"
    monkeypatch.setenv("WILDS_LOIS_DATA_ROOT", str(tmp_path / "mnt"))
    assert paths.lois_data_root() == tmp_path / "mnt"
    assert config.get().filing.share_root == tmp_path / "mnt"


def test_config_defaults_follow_the_root(wilds_root):
    cfg = config.get()
    assert cfg.filing.staging_root == wilds_root / "var/lib/staging"
    assert cfg.store.path == wilds_root / "var/lib/settings.db"


def test_a_path_in_config_toml_wins_over_the_derived_default(config_file, tmp_path):
    config_file(
        f'[filing]\nstaging_root = "{tmp_path}/elsewhere"\n[store]\npath = "{tmp_path}/s.db"\n'
    )
    cfg = config.get()
    assert cfg.filing.staging_root == tmp_path / "elsewhere" and cfg.store.path == tmp_path / "s.db"


def test_the_two_files_default_under_etc_unless_named(monkeypatch, wilds_root, tmp_path):
    # the fixtures name both; unset them to see the defaults being looked for
    monkeypatch.delenv("WILDS_CONTROL_CONFIG")
    monkeypatch.delenv("WILDS_CONTROL_PASSPHRASES")
    (wilds_root / "etc").mkdir(parents=True)
    (wilds_root / "etc" / "config.toml").write_text('opmode = "ldt-production"\n')
    (wilds_root / "etc" / "passphrases.toml").write_text('[passphrases]\nto = "from-etc"\n')
    cfg = config.get()
    assert cfg.opmode == "ldt-production" and cfg.passphrases.to == "from-etc"


def test_a_missing_root_stops_the_start_and_says_what_to_set(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    from wilds.control.app import create_app

    monkeypatch.setenv("WILDS_ROOT", str(tmp_path / "nowhere"))
    with pytest.raises(RuntimeError, match="WILDS_ROOT=~/wilds-dev"), TestClient(create_app()):
        pass

    monkeypatch.delenv("WILDS_ROOT")  # unset: the default root is named in the message
    monkeypatch.setattr(paths, "DEFAULT_ROOT", str(tmp_path / "default"))
    with pytest.raises(RuntimeError, match="the default root .*default"):
        paths.require_root()
