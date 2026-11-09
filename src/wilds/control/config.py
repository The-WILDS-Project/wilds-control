"""The TOML configuration (plan §3.10), minimal for now: opmode and presence, and the
passphrases, which live in a file of their own.

``$WILDS_CONTROL_CONFIG``, else ``/etc/wilds/config.toml``: everything but the passphrases.
``$WILDS_CONTROL_PASSPHRASES``, else ``/etc/wilds/passphrases.toml``: just a ``[passphrases]``
section, so the secrets can have their own owner and mode, and stay out of anything copied
around or committed (in development it is ``WILDS/.vault/passphrases.toml``).

Both are checked on every access and re-read when they change, so a new passphrase applies
without a restart. An invalid file is rejected whole and the running configuration kept. A
missing file means the defaults. Passphrases are deployment settings: files only, never
editable from the GUI.
"""

import logging
import os
import secrets
import threading
import tomllib
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, Field, ValidationError, field_validator

log = logging.getLogger(__name__)

DEFAULT_PATH = "/etc/wilds/config.toml"
PASSPHRASES_DEFAULT_PATH = "/etc/wilds/passphrases.toml"
DEFAULT_ENGINEERING_PASSPHRASE = "physics"


class Passphrases(BaseModel):
    engineering: str = DEFAULT_ENGINEERING_PASSPHRASE
    to: str | None = None  # the TO endpoints stay closed until this is set


class Presence(BaseModel):
    # Seconds without a valid TO request before the TO counts as offline. Not 15: browsers
    # throttle timers in background tabs to about one a minute.
    to_timeout_s: float = 75.0


class Site(BaseModel):
    # "LDT local time" in the GUI header. Arizona has no daylight saving, so this is UTC-7
    # all year; the GUI formats times in this zone from the server's UTC clock.
    timezone: str = "America/Phoenix"

    @field_validator("timezone")
    @classmethod
    def _known_zone(cls, v: str) -> str:
        try:
            ZoneInfo(v)
        except (ZoneInfoNotFoundError, ValueError) as e:
            raise ValueError(f"unknown time zone {v!r}") from e
        return v


class GuideCameraConfig(BaseModel):
    # "simulated" is the default so a bare checkout runs; "asi" is the real camera (deployment
    # only: changing it needs a restart).
    kind: Literal["simulated", "asi"] = "simulated"


class Filing(BaseModel):
    """Where data files are staged and delivered (plan §3.15). Deployment settings: read when
    wilds-control starts, so a change needs a restart."""

    staging_root: Path = Path("/var/lib/wilds/staging")  # local disk, large enough for a night
    share_root: Path = Path("/lois/data/dct")  # the facility's data folder; never created here
    # True in production: deliver only while share_root is a mount point. An unmounted mount
    # point is an empty local folder, and delivering into it would fill the local disk.
    share_is_mount: bool = False
    sweep_interval_s: float = Field(default=30.0, gt=0)
    probe_timeout_s: float = Field(default=2.0, gt=0)
    disk_ceiling: float = Field(default=0.90, gt=0, le=1)  # no new files above this

    @field_validator("staging_root", "share_root")
    @classmethod
    def _absolute(cls, v: Path) -> Path:
        v = v.expanduser()  # "~/wilds-dev/staging" in a development config.toml
        if not v.is_absolute():
            raise ValueError(f"{v} must be an absolute path")
        return v


class Store(BaseModel):
    """The SQLite file for operational and calibration settings (plan §3.10). Deployment
    setting: read when the store is first used."""

    path: Path = Path("/var/lib/wilds/settings.db")

    @field_validator("path")
    @classmethod
    def _absolute(cls, v: Path) -> Path:
        v = v.expanduser()
        if not v.is_absolute():
            raise ValueError(f"{v} must be an absolute path")
        return v


OpMode = Literal["development", "production"]


class Startup(BaseModel):
    """Whether wilds-control starts the components up by itself when it boots. Unset: yes in
    development (simulators), no in production, where a startup moves stages and heats lamps and
    someone should choose the moment."""

    autostart: bool | None = None


class Config(BaseModel):
    # What this machine accepts: development-only helpers (the demo job) are refused in
    # production. Development is the default so a bare checkout works; a deployed
    # config.toml says "production".
    opmode: OpMode = "development"
    passphrases: Passphrases = Passphrases()  # from passphrases.toml, never from config.toml
    passphrases_in_config_toml: bool = False  # a [passphrases] there was ignored
    presence: Presence = Presence()
    site: Site = Site()
    guide_camera: GuideCameraConfig = GuideCameraConfig()
    filing: Filing = Filing()
    store: Store = Store()
    startup: Startup = Startup()

    @property
    def is_development(self) -> bool:
        return self.opmode == "development"

    @property
    def autostart(self) -> bool:
        a = self.startup.autostart
        return self.is_development if a is None else a

    def warnings(self) -> list[str]:
        out = []
        if self.passphrases_in_config_toml:
            out.append(
                "config.toml has a [passphrases] section, which is ignored; "
                "passphrases are read from passphrases.toml"
            )
        return out


def _stat(path: Path) -> int | None:
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return None


class _Store:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._key: tuple[tuple[Path, int | None], tuple[Path, int | None]] | None = None
        self._config = Config()

    def get(self) -> Config:
        path = Path(os.environ.get("WILDS_CONTROL_CONFIG", DEFAULT_PATH))
        secrets_path = Path(os.environ.get("WILDS_CONTROL_PASSPHRASES", PASSPHRASES_DEFAULT_PATH))
        key = ((path, _stat(path)), (secrets_path, _stat(secrets_path)))
        with self._lock:
            if self._key != key:
                self._key = key
                self._load(*key)
            return self._config

    def _load(self, main: tuple[Path, int | None], secret: tuple[Path, int | None]) -> None:
        try:
            data = self._read(*main)
            ignored = "passphrases" in data
            data.pop("passphrases", None)
            vault = self._read(*secret)
            if extra := set(vault) - {"passphrases"}:
                raise ValueError(f"only a [passphrases] section belongs here, not {sorted(extra)}")
            data["passphrases"] = vault.get("passphrases", {})
            self._config = Config.model_validate({**data, "passphrases_in_config_toml": ignored})
        except (OSError, tomllib.TOMLDecodeError, ValidationError, ValueError) as e:
            log.error("rejected the configuration, keeping the running one: %s", e)
        for w in self._config.warnings():
            log.warning(w)

    @staticmethod
    def _read(path: Path, mtime: int | None) -> dict:
        if mtime is None:
            log.info("no file at %s; using defaults", path)
            return {}
        try:
            data = tomllib.loads(path.read_text())
        except (OSError, tomllib.TOMLDecodeError) as e:
            raise ValueError(f"{path}: {e}") from e
        log.info("read %s", path)
        return data


_store = _Store()


def get() -> Config:
    return _store.get()


def _matches(given: str | None, expected: str | None) -> bool:
    return bool(expected) and given is not None and secrets.compare_digest(given, expected)


def check_engineering(given: str | None) -> bool:
    return _matches(given, get().passphrases.engineering)


def check_to(given: str | None) -> bool:
    return _matches(given, get().passphrases.to)
