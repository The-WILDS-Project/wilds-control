"""The TOML configuration file (plan §3.10), minimal for now: passphrases and presence.

Read from ``$WILDS_CONTROL_CONFIG``, else ``/etc/wilds/config.toml``. The file is checked on
every access and re-read when it changes, so a new passphrase applies without a restart. An
invalid file is rejected whole and the running configuration kept. A missing file means the
defaults. Passphrases are deployment settings: file only, never editable from the GUI.
"""

import logging
import os
import secrets
import threading
import tomllib
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ValidationError, field_validator

log = logging.getLogger(__name__)

DEFAULT_PATH = "/etc/wilds/config.toml"
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


class Config(BaseModel):
    passphrases: Passphrases = Passphrases()
    presence: Presence = Presence()
    site: Site = Site()

    @property
    def engineering_passphrase_is_default(self) -> bool:
        return self.passphrases.engineering == DEFAULT_ENGINEERING_PASSPHRASE

    def warnings(self) -> list[str]:
        out = []
        if self.engineering_passphrase_is_default:
            out.append("The engineering passphrase is still the default; set it in config.toml")
        return out


class _Store:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._key: tuple[Path, int | None] | None = None
        self._config = Config()

    def get(self) -> Config:
        path = Path(os.environ.get("WILDS_CONTROL_CONFIG", DEFAULT_PATH))
        try:
            mtime: int | None = path.stat().st_mtime_ns
        except OSError:
            mtime = None
        with self._lock:
            if self._key != (path, mtime):
                self._key = (path, mtime)
                self._load(path, mtime)
            return self._config

    def _load(self, path: Path, mtime: int | None) -> None:
        if mtime is None:
            self._config = Config()
            log.info("no configuration file at %s; using defaults", path)
        else:
            try:
                self._config = Config.model_validate(tomllib.loads(path.read_text()))
                log.info("read configuration from %s", path)
            except (OSError, tomllib.TOMLDecodeError, ValidationError) as e:
                log.error("rejected %s, keeping the running configuration: %s", path, e)
        for w in self._config.warnings():
            log.warning(w)


_store = _Store()


def get() -> Config:
    return _store.get()


def _matches(given: str | None, expected: str | None) -> bool:
    return bool(expected) and given is not None and secrets.compare_digest(given, expected)


def check_engineering(given: str | None) -> bool:
    return _matches(given, get().passphrases.engineering)


def check_to(given: str | None) -> bool:
    return _matches(given, get().passphrases.to)
