import os

import pytest


@pytest.fixture(autouse=True)
def config_file(tmp_path, monkeypatch):
    """Every test gets its own (initially absent) wilds.toml; call the result with TOML text."""
    path = tmp_path / "wilds.toml"
    monkeypatch.setenv("WILDS_CONTROL_CONFIG", str(path))
    version = [0]

    def write(text: str) -> None:
        path.write_text(text)
        version[0] += 1  # distinct mtime even when rewritten within the same clock tick
        os.utime(path, ns=(version[0] * 10**9, version[0] * 10**9))

    return write
