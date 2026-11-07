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


def _writer(path, monkeypatch, env):
    monkeypatch.setenv(env, str(path))
    version = [0]

    def write(text: str) -> None:
        path.write_text(text)
        version[0] += 1  # distinct mtime even when rewritten within the same clock tick
        os.utime(path, ns=(version[0] * 10**9, version[0] * 10**9))

    return write
