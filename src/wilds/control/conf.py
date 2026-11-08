"""Live, typed access to a component's settings (plan §3.10).

Every component has up to two *slices* of configuration, each a frozen Pydantic model kept in
the settings store (``settings_store.py``) and reached through a :class:`ConfHandle`:

* ``component.conf``: **operational** settings (exposure, gain, presets, a cooler setpoint);
  the control holder may change them.
* ``component.cal``: **calibration** values (plate scale, slit pixel, focus offset, ADC zero
  points); the holder in engineering mode may change them, each change is a recorded version,
  and they default to ``None`` ("not calibrated"): procedures that need one call
  :meth:`ConfHandle.require`, which refuses with a clear error instead of using a made-up
  number. Separate slices keep calibrations from drowning in the churn of operational tweaks.

Deployment settings (addresses, paths, ``opmode``) are not here: they are in ``config.toml``.

**Defaults come from the model, the current values from the store**; nothing saved yet means
the defaults, and "reset to defaults" is saving ``Model()`` as a new version.

**The handle is the live reference**, the model is not: models are frozen and a save swaps in a
new one, so anyone holding ``component.conf`` sees the change, and nobody sees half of one. Use
``conf.current.gain`` for a single value; for several that must agree, or a whole job, take
``snap = conf.snapshot()`` once and use that (a procedure must not see its plate scale change
half-way through). Treat containers inside a model (a dict of presets) as read-only too.

**One way to write:** ``save``, ``patch`` or ``revert`` validate, ask the component (the
``before_change`` hooks, which may refuse: busy, or a value this hardware cannot take), record
who/when/why in the store, swap the live model and then tell the listeners (``on_change``), in
that order, so the live model and the store cannot disagree: a refused change is never recorded.
Components subscribe to apply a change to hardware. A procedure never saves: it works on its
snapshot, with its own temporary overrides.
"""

import logging
import threading
from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from . import config
from .errors import ApiError
from .jobs import StartedBy
from .settings_store import Entry, SettingsStore, UnknownVersion, get_store

log = logging.getLogger(__name__)

Slice = Literal["conf", "cal"]
Listener = Callable[[Any, Any, list[str]], None]  # (old model, new model, keys changed)
Veto = Callable[[Any, Any, list[str]], None]  # same arguments; raises to refuse the change


class ConfModel(BaseModel):
    """Base of every settings model: frozen (a save replaces it, nothing edits it) and tolerant
    of keys a newer or older version of the model does not know."""

    model_config = ConfigDict(frozen=True, extra="ignore")


def setting(
    default=None,
    *,
    unit: str | None = None,
    label: str | None = None,
    help: str | None = None,
    **constraints,
):
    """A model field that describes itself to the GUI's generic form (served as the model's JSON
    Schema): a label, a unit and help text, with Pydantic's limits (``ge``, ``le``, ...)."""
    extra = {"unit": unit} if unit else None
    return Field(
        default=default, title=label, description=help, json_schema_extra=extra, **constraints
    )


def by_opmode(*, development, production, label: str | None = None, help: str | None = None):
    """A default that depends on ``opmode`` in config.toml, read when the model is built."""
    return Field(
        default_factory=lambda: development if config.get().is_development else production,
        title=label,
        description=help,
    )


class ConfHandle[M: ConfModel]:
    def __init__(
        self,
        component: str,
        slice: Slice,
        model: type[M],
        store: SettingsStore | None = None,
    ):
        self.component = component
        self.slice = slice
        self.model = model
        self.section = component if slice == "conf" else f"{component}.cal"
        self._store = store or get_store()
        self._lock = threading.Lock()
        self._listeners: list[Listener] = []
        self._vetoes: list[Veto] = []
        self._current: M = self._store.get(self.section, model)
        self._version = self._store.version(self.section)

    # -- reading ------------------------------------------------------------------------

    @property
    def current(self) -> M:
        """The live model. Read one value from it; take a :meth:`snapshot` for several."""
        return self._current

    def snapshot(self) -> M:
        """The model as it is now; it never changes, however many saves follow."""
        return self._current

    @property
    def version(self) -> int:
        """The stored version (0 while nothing has been saved: the defaults)."""
        return self._version

    def history(self, limit: int = 50) -> list[Entry]:
        return self._store.history(self.section, limit)

    def schema(self) -> dict[str, Any]:
        return self.model.model_json_schema()

    def require(self, name: str):
        """A calibration value, or ``409 not_calibrated`` if it has not been measured yet."""
        value = getattr(self._current, name)
        if value is None:
            raise ApiError(
                409,
                "not_calibrated",
                f"{self.component}: {name} has not been calibrated",
                {"component": self.component, "field": name},
            )
        return value

    # -- changing -----------------------------------------------------------------------

    def save(
        self,
        new: M,
        *,
        who: StartedBy,
        reason: str | None = None,
        expected_version: int | None = None,
    ) -> Entry | None:
        """Record ``new`` as the next version and make it live. ``None`` when nothing changed.
        Raises ``SettingsConflict`` if ``expected_version`` is stale (nothing changes then)."""
        if not isinstance(new, self.model):
            raise TypeError(
                f"{self.section} takes a {self.model.__name__}, not {type(new).__name__}"
            )
        with self._lock:
            old = self._current
            changed = _changed(old, new)
            if not changed:
                return None
            self._allow(old, new, changed)  # the component may refuse; then nothing is recorded
            entry = self._store.put(
                self.section, new, who=who, reason=reason, expected_version=expected_version
            )
            if entry is None:
                return None
            self._current, self._version = new.model_copy(deep=True), entry.version
            current = self._current
        self._notify(old, current, entry.changed)
        return entry

    def patch(
        self,
        changes: dict[str, Any],
        *,
        who: StartedBy,
        reason: str | None = None,
        expected_version: int | None = None,
    ) -> Entry | None:
        """Change some fields, validated as a whole (a partial PUT). Raises pydantic's
        ``ValidationError`` for a bad value, with nothing changed."""
        merged = self.model.model_validate({**self._current.model_dump(), **changes})
        return self.save(merged, who=who, reason=reason, expected_version=expected_version)

    def revert(self, version: int, *, who: StartedBy, reason: str | None = None) -> Entry | None:
        """Restore an earlier version as a new one, and make it live."""
        with self._lock:
            old = self._current
            try:
                target = self.model.model_validate(self._store.entry(self.section, version).values)
            except KeyError as e:
                raise UnknownVersion(str(e)) from e
            changed = _changed(old, target)
            if changed:
                self._allow(old, target, changed)
            entry = self._store.revert(self.section, version, self.model, who=who, reason=reason)
            if entry is None:
                return None
            self._current, self._version = self.model.model_validate(entry.values), entry.version
            current = self._current
        self._notify(old, current, entry.changed)
        return entry

    # -- the component's say ------------------------------------------------------------

    def before_change(self, veto: Veto) -> None:
        """Ask ``veto(old, new, changed_keys)`` before a change is recorded. This is where a
        component checks that it can take the change (a window inside the sensor, not busy)
        and applies it to its hardware: if it raises (an ``ApiError`` such as ``409 busy``, or
        any error the caller should see) nothing is saved, swapped or announced, so a device
        that refused or failed is never recorded as having the setting. It runs under the
        handle's lock, so it must be quick and must not save."""
        if veto not in self._vetoes:
            self._vetoes.append(veto)

    def _allow(self, old: M, new: M, changed: list[str]) -> None:
        for veto in list(self._vetoes):
            veto(old, new, changed)

    # -- listeners ----------------------------------------------------------------------

    def on_change(self, listener: Listener) -> None:
        """Call ``listener(old, new, changed_keys)`` after every change that lands (not at
        subscription). A listener that raises is logged and does not stop the others."""
        if listener not in self._listeners:
            self._listeners.append(listener)

    def _notify(self, old: M, new: M, changed: list[str]) -> None:
        for listener in list(self._listeners):
            try:
                listener(old, new, changed)
            except Exception:
                log.exception("%s: a change listener failed", self.section)


def _changed(old: BaseModel, new: BaseModel) -> list[str]:
    before, after = old.model_dump(mode="json"), new.model_dump(mode="json")
    return sorted(k for k in before.keys() | after.keys() if before.get(k) != after.get(k))


class ConfRegistry:
    """Every component's handles by name, for the ``/config/{component}`` endpoints."""

    def __init__(self) -> None:
        self._handles: dict[tuple[str, Slice], ConfHandle[Any]] = {}

    def register[M: ConfModel](
        self, component: str, slice: Slice, model: type[M], store: SettingsStore | None = None
    ) -> ConfHandle[M]:
        """The handle for ``component``'s ``slice``, made on the first call; later calls with the
        same model return it (components are rebuilt in tests), a different model is a bug."""
        key = (component, slice)
        have = self._handles.get(key)
        if have is not None:
            if have.model is not model:
                raise ValueError(
                    f"{component}.{slice} is already registered with {have.model.__name__}"
                )
            return have
        self._handles[key] = handle = ConfHandle(component, slice, model, store)
        return handle

    def add(self, handle: ConfHandle[Any]) -> None:
        """Adopt a handle a component built itself, replacing an earlier one (a component that
        is made again, as in tests, takes over its name)."""
        self._handles[(handle.component, handle.slice)] = handle

    def get(self, component: str, slice: Slice = "conf") -> ConfHandle[Any]:
        try:
            return self._handles[(component, slice)]
        except KeyError:
            raise KeyError(f"{component} has no {slice} registered") from None

    def components(self) -> list[str]:
        return sorted({c for c, _ in self._handles})

    def slices(self, component: str) -> list[Slice]:
        return [s for c, s in sorted(self._handles) if c == component]

    def clear(self) -> None:
        """Forget every handle (for tests)."""
        self._handles.clear()


confs = ConfRegistry()
