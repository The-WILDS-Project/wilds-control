import os

import pytest


@pytest.fixture(autouse=True)
def config_file(tmp_path, monkeypatch):
    """Every test gets its own (initially absent) config.toml; call the result with TOML text."""
    return _writer(tmp_path / "config.toml", monkeypatch, "WILDS_CONTROL_CONFIG")


@pytest.fixture(autouse=True)
def passphrases_file(tmp_path, monkeypatch):
    """Likewise for passphrases.toml, the file holding the [passphrases] section."""
    return _writer(tmp_path / "passphrases.toml", monkeypatch, "WILDS_CONTROL_PASSPHRASES")


@pytest.fixture(autouse=True)
def fresh_filing(tmp_path, monkeypatch):
    """Each test stages under its own folders and never touches a real share."""
    from wilds.control import config, filing

    filing.reset()
    config_filing = config.Filing(
        staging_root=tmp_path / "staging", share_root=tmp_path / "share", sweep_interval_s=0.05
    )
    monkeypatch.setattr(filing, "_default_config", lambda: config_filing)
    yield
    filing.reset()


@pytest.fixture(autouse=True)
def fresh_settings_store(tmp_path, monkeypatch):
    """Each test has its own settings database."""
    from wilds.control import config, settings_store
    from wilds.control.conf import confs

    settings_store.reset()
    confs.clear()
    cfg = config.Store(path=tmp_path / "state" / "settings.db")
    monkeypatch.setattr(settings_store, "_default_path", lambda: cfg.path)
    yield
    settings_store.reset()
    confs.clear()


@pytest.fixture(autouse=True)
def fresh_guide_camera():
    """Each test builds its own guide camera, from its own config."""
    from wilds.control import guide_camera

    guide_camera.reset()
    yield
    guide_camera.reset()


@pytest.fixture(autouse=True)
def fresh_system():
    """Each test starts with no components registered for startup and no umbrella running."""
    from wilds.control.lifecycle import system

    system.reset()
    yield
    system.reset()


def _writer(path, monkeypatch, env):
    monkeypatch.setenv(env, str(path))
    version = [0]

    def write(text: str) -> None:
        path.write_text(text)
        version[0] += 1  # distinct mtime even when rewritten within the same clock tick
        os.utime(path, ns=(version[0] * 10**9, version[0] * 10**9))

    return write
