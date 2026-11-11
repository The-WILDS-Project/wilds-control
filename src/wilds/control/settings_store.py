"""The store for operational and calibration settings (plan §3.10): one SQLite file.

Deployment settings (broker host, device addresses, paths, opmode) are hand-edited in
``$WILDS_ROOT/etc/config.toml`` and read at start (``config.py``). Everything a *procedure or the GUI*
writes while the instrument runs lives here instead: stage presets, camera exposure and gain,
plate scale, slit pixel, focus offset, ADC zero points. A hand-edited, root-owned file is the
wrong home for values the application changes; and a database server (MAST uses MongoDB, for
a fleet of units) is the wrong weight for one machine. SQLite is in the standard library, one
file in ``$WILDS_ROOT/var/lib``, transactional, and needs no server to be running.

**A section is one component's settings**: the JSON of its Pydantic model, versioned as a
whole. ``put`` validates nothing itself (the model has already), records who changed it, when
and why, and keeps every version: ``history`` lists them with the keys each changed, and
``revert`` restores an earlier version *as a new version*, so the history only ever grows.
A ``put`` that changes nothing records nothing. ``expected_version`` is optimistic
concurrency: a GUI form loaded at version 4 that finds version 5 is refused (``409``), not
silently overwritten.

A database written by a newer wilds-control than the one reading it is refused, not guessed at.
"""

import json
import logging
import sqlite3
import threading
import time
from collections.abc import Callable
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from . import config
from .jobs import StartedBy

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE section (
    component TEXT PRIMARY KEY,
    version   INTEGER NOT NULL,
    json      TEXT NOT NULL
);
CREATE TABLE history (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    component     TEXT NOT NULL,
    version       INTEGER NOT NULL,
    json          TEXT NOT NULL,
    who           TEXT NOT NULL,
    code          TEXT NOT NULL,
    reason        TEXT,
    at_ms         INTEGER NOT NULL,
    reverted_from INTEGER,
    UNIQUE (component, version)
);
"""


class SettingsConflict(Exception):
    """The section is no longer at the version the caller saw (HTTP 409)."""

    def __init__(self, component: str, expected: int | None, actual: int):
        super().__init__(f"{component} is at version {actual}, not {expected}")
        self.component, self.expected, self.actual = component, expected, actual


class UnknownVersion(KeyError):
    """No such version of the section (HTTP 404)."""


class StoreError(RuntimeError):
    """The database cannot be used (too new, or a stored section no longer validates)."""


class Entry(BaseModel):
    """One version of one section."""

    component: str
    version: int
    values: dict[str, Any]
    who: str
    code: str
    reason: str | None = None
    at_ms: int
    reverted_from: int | None = None  # set when this version restored an older one
    changed: list[str] = []  # the keys that differ from the version before


def _dump(model: BaseModel) -> dict[str, Any]:
    return model.model_dump(mode="json")


def _diff(before: dict[str, Any], after: dict[str, Any]) -> list[str]:
    return sorted(
        k
        for k in before.keys() | after.keys()
        if k not in before or k not in after or before[k] != after[k]
    )


class SettingsStore:
    def __init__(self, path: Path, clock: Callable[[], float] = time.time):
        self.path = Path(path)
        self._clock = clock
        self._lock = threading.Lock()  # one writer at a time in this process; SQLite does the rest
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            version = db.execute("PRAGMA user_version").fetchone()[0]
            if version > SCHEMA_VERSION:
                raise StoreError(
                    f"{self.path} is schema {version}; this wilds-control knows {SCHEMA_VERSION}"
                )
            if version == 0:
                db.executescript(_SCHEMA)
                db.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(self.path, timeout=10, isolation_level=None)  # we BEGIN ourselves
        try:
            db.execute("PRAGMA journal_mode = WAL")  # readers never block the writer
            db.execute("PRAGMA synchronous = FULL")  # a calibration is worth the fsync
            yield db
        finally:
            db.close()

    # -- reading ------------------------------------------------------------------------

    def get[M: BaseModel](self, component: str, model: type[M]) -> M:
        """The current settings of ``component``: the stored ones, or the model's defaults
        while nothing has been saved. Keys added to the model since are defaulted."""
        stored = self.version(component)
        if stored == 0:
            return model()
        with closing(sqlite3.connect(self.path, timeout=10)) as db:
            (text,) = db.execute(
                "SELECT json FROM section WHERE component = ?", (component,)
            ).fetchone()
        try:
            return model.model_validate(json.loads(text))
        except ValueError as e:
            raise StoreError(
                f"stored {component} (version {stored}) no longer validates: {e}"
            ) from e

    def version(self, component: str) -> int:
        """The current version (0 while nothing has been saved)."""
        with closing(sqlite3.connect(self.path, timeout=10)) as db:
            row = db.execute(
                "SELECT version FROM section WHERE component = ?", (component,)
            ).fetchone()
        return row[0] if row else 0

    def entry(self, component: str, version: int) -> Entry:
        """One stored version (``KeyError`` if there is no such version)."""
        for e in self.history(component, limit=1_000_000):
            if e.version == version:
                return e
        raise UnknownVersion(f"{component} has no version {version}")

    def components(self) -> list[str]:
        with closing(sqlite3.connect(self.path, timeout=10)) as db:
            return [r[0] for r in db.execute("SELECT component FROM section ORDER BY component")]

    def history(self, component: str, limit: int = 50) -> list[Entry]:
        """Versions of the section, newest first, each with the keys it changed."""
        with closing(sqlite3.connect(self.path, timeout=10)) as db:
            rows = db.execute(
                "SELECT version, json, who, code, reason, at_ms, reverted_from FROM history "
                "WHERE component = ? ORDER BY version DESC LIMIT ?",
                (component, limit + 1),
            ).fetchall()
        out = []
        for i, (version, text, who, code, reason, at_ms, reverted_from) in enumerate(rows[:limit]):
            values = json.loads(text)
            older = json.loads(rows[i + 1][1]) if i + 1 < len(rows) else {}
            out.append(
                Entry(
                    component=component,
                    version=version,
                    values=values,
                    who=who,
                    code=code,
                    reason=reason,
                    at_ms=at_ms,
                    reverted_from=reverted_from,
                    changed=_diff(older, values),
                )
            )
        return out

    # -- writing ------------------------------------------------------------------------

    def put(
        self,
        component: str,
        settings: BaseModel,
        *,
        who: StartedBy,
        reason: str | None = None,
        expected_version: int | None = None,
        reverted_from: int | None = None,
    ) -> Entry | None:
        """Save ``settings`` as the next version. Returns the new entry, or ``None`` when the
        values are what is already stored. Raises :class:`SettingsConflict` if
        ``expected_version`` is given and the section has moved on."""
        values = _dump(settings)
        with self._lock, self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                row = db.execute(
                    "SELECT version, json FROM section WHERE component = ?", (component,)
                ).fetchone()
                current, current_json = (row[0], row[1]) if row else (0, None)
                if expected_version is not None and expected_version != current:
                    raise SettingsConflict(component, expected_version, current)
                if current_json is not None and json.loads(current_json) == values:
                    db.execute("ROLLBACK")
                    return None
                version, at_ms = current + 1, int(self._clock() * 1000)
                text = json.dumps(values, sort_keys=True)
                db.execute(
                    "INSERT INTO section (component, version, json) VALUES (?, ?, ?) "
                    "ON CONFLICT (component) DO UPDATE SET version = excluded.version, json = excluded.json",
                    (component, version, text),
                )
                db.execute(
                    "INSERT INTO history (component, version, json, who, code, reason, at_ms, reverted_from) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (component, version, text, who.name, who.code, reason, at_ms, reverted_from),
                )
                db.execute("COMMIT")
            except BaseException:
                if db.in_transaction:
                    db.execute("ROLLBACK")
                raise
        log.info(
            "settings: %s v%d saved by %s [%s]%s",
            component,
            version,
            who.name,
            who.code,
            f": {reason}" if reason else "",
        )
        return Entry(
            component=component,
            version=version,
            values=values,
            who=who.name,
            code=who.code,
            reason=reason,
            at_ms=at_ms,
            reverted_from=reverted_from,
            changed=_diff(json.loads(current_json) if current_json else {}, values),
        )

    def revert[M: BaseModel](
        self,
        component: str,
        version: int,
        model: type[M],
        *,
        who: StartedBy,
        reason: str | None = None,
    ) -> Entry | None:
        """Restore an earlier version as a new one (the history only grows). ``None`` if the
        section already has those values."""
        with closing(sqlite3.connect(self.path, timeout=10)) as db:
            row = db.execute(
                "SELECT json FROM history WHERE component = ? AND version = ?", (component, version)
            ).fetchone()
        if row is None:
            raise UnknownVersion(f"{component} has no version {version}")
        try:
            restored = model.model_validate(json.loads(row[0]))
        except ValueError as e:
            raise StoreError(f"{component} version {version} no longer validates: {e}") from e
        return self.put(
            component,
            restored,
            who=who,
            reason=reason or f"revert to version {version}",
            reverted_from=version,
        )


_store: SettingsStore | None = None


def _default_path() -> Path:
    return config.get().store.path


def get_store() -> SettingsStore:
    """The one store, opened on first use. Use it as a FastAPI ``Depends``; tests override it
    through ``app.dependency_overrides``."""
    global _store
    if _store is None:
        _store = SettingsStore(_default_path())
    return _store


def reset() -> None:
    """Forget the instance (for tests)."""
    global _store
    _store = None
