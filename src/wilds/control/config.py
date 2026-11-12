"""The TOML configuration (plan §3.10), minimal for now: opmode and presence, and the
passphrases, which live in a file of their own.

``$WILDS_CONTROL_CONFIG``, else ``$WILDS_ROOT/etc/config.toml``: everything but the passphrases.
``$WILDS_CONTROL_PASSPHRASES``, else ``$WILDS_ROOT/etc/passphrases.toml``: just a ``[passphrases]``
section, so the secrets can have their own owner and mode, and stay out of anything copied
around or committed (in development it is ``WILDS/.vault/passphrases.toml``). Where everything
else lives, and its defaults, is in ``paths.py``.

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

from . import paths

log = logging.getLogger(__name__)

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
    # Fill in the calibration (plate scale, slit pixel, orientation...) the first time the
    # simulated camera starts, so the live view's slit overlay can be tried; never touches
    # values that are already set.
    seed_simulated_calibration: bool = False


class SkyConfig(BaseModel):
    """Where the simulated guide camera gets real stars (plan §3.3); without a catalogue it draws an
    invented field that no plate solver can match. Deployment settings: read at start-up."""

    # The folder of a catsHTM catalogue, e.g. Gaia DR3 (MAST: \\\\last0\\catalogs\\GAIA\\DR3, here
    # wherever the share is mounted). Empty: no catalogue.
    gaia_root: Path | None = None
    # A star field saved on local disk by ``python -m wilds.control.fetch_field`` (M13 by
    # default). When the file exists it is the data set: no share and no h5py needed.
    field_file: Path | None = None
    catalog_name: str = "GAIADR3"
    mag_column: str = "phot_g_mean_mag"
    mag_limit: float = Field(18.0, gt=0, le=22)  # the faintest star drawn


class SolverConfig(BaseModel):
    """The plate solver the optics calibration uses (plan §3.3). Deployment settings."""

    # "astrometry" is wilds-solving over astrometry.net index files (series 5200); "simulated"
    # answers from the simulated camera's own hidden optics, for a machine with no index files.
    kind: Literal["astrometry", "simulated"] = "astrometry"
    # Where the index files are ($WILDS_ASTROMETRY_CACHE, else ~/.astrometry); never downloaded
    # here: the files are large, fetch them on purpose.
    index_dir: Path | None = None
    scales: list[int] = Field(default_factory=lambda: [0, 1, 2])  # series 5200 scale numbers


class Filing(BaseModel):
    """Where data files are staged and delivered (plan §3.15). Deployment settings: read when
    wilds-control starts, so a change needs a restart."""

    # Local disk, large enough for a night; $WILDS_ROOT/var/lib/staging unless set.
    staging_root: Path = Field(default_factory=paths.staging_root)
    # The facility's data folder, never created here; $WILDS_LOIS_DATA_ROOT unless set.
    share_root: Path = Field(default_factory=paths.lois_data_root)
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

    path: Path = Field(default_factory=paths.settings_db)  # $WILDS_ROOT/var/lib/settings.db

    @field_validator("path")
    @classmethod
    def _absolute(cls, v: Path) -> Path:
        v = v.expanduser()
        if not v.is_absolute():
            raise ValueError(f"{v} must be an absolute path")
        return v


# What WILDS runs against (plan §3.10):
#   development    this machine, with in-process simulators of everything: the /dev helpers exist
#   ldt-simulated  LDT's own simulators (the TCS and MCU on nysa) through the test broker (tanagra)
#   ldt-production the real telescope, through its own broker (joe)
OpMode = Literal["development", "ldt-simulated", "ldt-production"]

# The broker hosts we know. A mode and its broker must agree, or wilds-control does not start: a
# session meant for the simulators must never reach the real telescope, and the other way round.
TEST_BROKER_HOSTS = frozenset({"tanagra", "tanagra.lowell.local"})


class Broker(BaseModel):
    """The LDT broker the telescope's messages go through (``[broker]``): needed, and checked
    against the mode, when the mode is not ``development``. Deployment setting: read at start-up."""

    host: str | None = None

    @field_validator("host")
    @classmethod
    def _tidy(cls, v: str | None) -> str | None:
        return v.strip().lower() or None if v else None


class Startup(BaseModel):
    """Whether wilds-control starts the components up by itself when it boots. Unset: yes in
    development (simulators), no in production, where a startup moves stages and heats lamps and
    someone should choose the moment."""

    autostart: bool | None = None


class Config(BaseModel):
    # What this machine runs against, see OpMode. Development is the default so a bare checkout
    # works; a deployed config.toml says "ldt-simulated" or "ldt-production".
    opmode: OpMode = "development"
    broker: Broker = Broker()
    passphrases: Passphrases = Passphrases()  # from passphrases.toml, never from config.toml
    passphrases_in_config_toml: bool = False  # a [passphrases] there was ignored
    presence: Presence = Presence()
    site: Site = Site()
    guide_camera: GuideCameraConfig = GuideCameraConfig()
    sky: SkyConfig = SkyConfig()
    solver: SolverConfig = SolverConfig()
    filing: Filing = Field(default_factory=Filing)  # built per load: its paths follow the root
    store: Store = Field(default_factory=Store)
    startup: Startup = Startup()

    @property
    def is_development(self) -> bool:
        return self.opmode == "development"

    @property
    def uses_ldt(self) -> bool:
        """Talks to an LDT broker (the simulators' or the telescope's), not in-process simulators."""
        return self.opmode != "development"

    def broker_problem(self) -> str:
        """Why the mode and the broker do not fit, or ``""``."""
        if not self.uses_ldt:
            return ""
        host = self.broker.host
        if not host:
            return f"opmode {self.opmode} needs [broker] host in config.toml"
        test = host in TEST_BROKER_HOSTS
        if self.opmode == "ldt-simulated" and not test:
            return (
                f"opmode ldt-simulated is for LDT's simulators, whose broker is "
                f"{', '.join(sorted(TEST_BROKER_HOSTS))}, not '{host}'"
            )
        if self.opmode == "ldt-production" and test:
            return f"opmode ldt-production would use the test broker '{host}': that is for ldt-simulated"
        return ""

    @property
    def autostart(self) -> bool:
        a = self.startup.autostart
        return self.is_development if a is None else a

    def warnings(self) -> list[str]:
        out = []
        if self.uses_ldt:  # until LdtTelescope exists the telescope is still the in-process one
            out.append(
                f"opmode {self.opmode}, but the telescope is still the in-process simulator: "
                "the LDT connection (wilds-bridge) is not wired in yet"
            )
        if self.passphrases_in_config_toml:
            out.append(
                "config.toml has a [passphrases] section, which is ignored; "
                "passphrases are read from passphrases.toml"
            )
        return out


def require_consistent_broker() -> None:
    """Stop at start-up, with a clear message, if the mode and the broker do not agree (like
    ``paths.require_root``): the one mistake that would put a test session on the telescope."""
    if problem := get().broker_problem():
        raise RuntimeError(problem)


def _stat(path: Path) -> int | None:
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return None


class _Store:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._key: tuple | None = None
        self._config = Config()

    def get(self) -> Config:
        path = Path(os.environ.get("WILDS_CONTROL_CONFIG") or paths.config_file())
        secrets_path = Path(os.environ.get("WILDS_CONTROL_PASSPHRASES") or paths.passphrases_file())
        key = (
            (path, _stat(path)),
            (secrets_path, _stat(secrets_path)),
            # the defaults of the paths derive from these, so a change must rebuild the model
            tuple(os.environ.get(n) for n in (paths.ROOT_ENV, paths.LOIS_DATA_ROOT_ENV)),
        )
        with self._lock:
            if self._key != key:
                self._key = key
                self._load(*key[:2])
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
