"""Where WILDS keeps its files (plan §3.10).

Everything lives under one root, ``$WILDS_ROOT`` (``/opt/wilds`` when unset; the .deb package
installs there), laid out as Linux expects:

``bin/ lib/ share/``
    the program: owned by the package, read-only, replaced on upgrade.
``etc/``
    ``config.toml`` and ``passphrases.toml`` (the passphrases in a file of their own, mode 0600).
    The package never ships these, so an upgrade cannot overwrite local configuration.
``var/lib/``
    what the software writes and must keep: ``settings.db``, ``staging/``, and
    ``guide-camera/calibration/``. An upgrade, or a purge, must not delete it.
``var/log/``, ``var/run/``
    logs and runtime files.

The facility's data folder (``/lois/data/dct``) is not under the root: it is a mount point of
ours to place, ``$WILDS_LOIS_DATA_ROOT``.

These are only the *defaults*. A path set in ``config.toml`` wins over the one derived here, and
``$WILDS_CONTROL_CONFIG`` / ``$WILDS_CONTROL_PASSPHRASES`` still name the two files directly
(development keeps its passphrases outside every repo this way). Development sets ``WILDS_ROOT``
to ``~/wilds-dev``, which then has the same layout as a deployment.
"""

import os
from pathlib import Path

ROOT_ENV = "WILDS_ROOT"
LOIS_DATA_ROOT_ENV = "WILDS_LOIS_DATA_ROOT"
DEFAULT_ROOT = "/opt/wilds"
DEFAULT_LOIS_DATA_ROOT = "/lois/data/dct"


def _absolute(name: str, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(f"${name} must be an absolute path, not {value!r}")
    return path


def root() -> Path:
    return _absolute(ROOT_ENV, os.environ.get(ROOT_ENV) or DEFAULT_ROOT)


def etc() -> Path:
    return root() / "etc"


def var_lib() -> Path:
    return root() / "var" / "lib"


def var_log() -> Path:
    return root() / "var" / "log"


def var_run() -> Path:
    return root() / "var" / "run"


def config_file() -> Path:
    return etc() / "config.toml"


def passphrases_file() -> Path:
    return etc() / "passphrases.toml"


def settings_db() -> Path:
    return var_lib() / "settings.db"


def staging_root() -> Path:
    return var_lib() / "staging"


def guide_camera_calibration() -> Path:
    return var_lib() / "guide-camera" / "calibration"


def lois_data_root() -> Path:
    """Where the facility's data folder is mounted (``$WILDS_LOIS_DATA_ROOT``)."""
    return _absolute(
        LOIS_DATA_ROOT_ENV, os.environ.get(LOIS_DATA_ROOT_ENV) or DEFAULT_LOIS_DATA_ROOT
    )
