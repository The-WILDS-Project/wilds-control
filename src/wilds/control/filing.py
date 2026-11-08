"""Local staging and delivery of data files (plan §3.15).

Everything the instrument produces (science frames, solver products) is written on the
local *staging* disk and delivered from there to the facility's data folder on the *share*
(``/lois/data/dct``). The share may be down, slow or not yet mounted; the night must not
care. Adapted from MAST's ``Filer`` (RAM disk to share, deferred moves, a sweeper), without
its ``MoveGuardian``: here a file only becomes visible to the sweeper when it is whole, so
nothing needs guarding.

**The staging folder is the queue.** A producer writes through :meth:`Filing.writing`, to a
hidden ``.<name>.partial`` beside the final name, and the file is renamed into place when the
block ends cleanly. Whatever is in staging under a normal name is complete and waiting, and
survives a restart; there is no in-memory queue to lose.

**Delivery** (a background thread, woken by every new file and otherwise every
``sweep_interval_s``): probe the share; if it answers, copy each waiting file to
``.<name>.partial`` on the share, rename it into place, and only then remove it from staging.
So the final name on the share is always a whole file (facility software watches that folder),
an interrupted copy leaves nothing but a partial that the next attempt overwrites, and a
file already there with the same bytes counts as delivered. A *different* file with the same
name is never overwritten: the source stays and the clash is reported once.

**The probe** creates and removes a folder in the share's root, in a thread joined with a
timeout, so a hung mount costs a probe timeout, not a stuck sweeper. While an earlier probe
is still hung, no second one is started. The share's root itself is never created, and with
``share_is_mount`` it must be a mount point: an unmounted mount point is an empty local
folder, and delivering into it would fill the local disk and look like success.

**Disk:** new files are refused once the staging disk is ``disk_ceiling`` full.

Numbering of the files (``<abbr>.NNNN.fits``) belongs to whoever names them; it must not
restart when staging empties, which is the producer's job when science exposures arrive.
"""

import contextlib
import errno
import filecmp
import logging
import os
import shutil
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from . import config

log = logging.getLogger(__name__)

PARTIAL_SUFFIX = ".partial"
DISK_CACHE_S = 5.0
PROBE_CACHE_S = 2.0

ShareState = Literal["unknown", "online", "offline"]


class StagingFull(OSError):
    """The staging disk is at its ceiling: no new files until delivery frees space."""

    def __init__(self, used: float, ceiling: float):
        super().__init__(errno.ENOSPC, f"the staging disk is {used:.0%} full (limit {ceiling:.0%})")


class FilingStatus(BaseModel):
    """What ``/status`` carries for the filing."""

    share: ShareState = "unknown"
    share_root: str
    staging_root: str
    pending: int = 0  # whole files waiting in staging
    pending_bytes: int = 0
    oldest_age_s: float | None = None  # age of the oldest waiting file
    delivered: int = 0  # since this process started
    last_delivery_ms: int | None = None
    last_error: str | None = None
    staging_used: float | None = None  # fraction of the staging disk in use
    staging_full: bool = False

    def warnings(self) -> list[str]:
        out = []
        if self.pending and self.share == "offline":
            age = f", the oldest {_age(self.oldest_age_s)} old" if self.oldest_age_s else ""
            out.append(
                f"The data share is not reachable: {self.pending} file(s) "
                f"({self.pending_bytes / 1e6:.0f} MB) wait on the staging disk{age}"
            )
        if self.staging_full:
            out.append(f"The staging disk is {self.staging_used or 0:.0%} full: no new files")
        if self.last_error and self.share == "online":
            out.append(f"Delivery problem: {self.last_error}")
        return out


def _age(seconds: float | None) -> str:
    s = int(seconds or 0)
    return f"{s} s" if s < 120 else f"{s // 60} min" if s < 7200 else f"{s // 3600} h"


def is_bookkeeping(name: str) -> bool:
    """Hidden names are the mechanism (partial files), never a product."""
    return name.startswith(".")


def partial_of(path: Path) -> Path:
    return path.with_name(f".{path.name}{PARTIAL_SUFFIX}")


class Probe:
    """Runs a check that may hang (a dead mount) without ever blocking its caller for more
    than ``timeout``, and without piling up threads: while one check is hung, the answer is
    simply "no"."""

    def __init__(self, check: Callable[[], bool], clock: Callable[[], float] = time.monotonic):
        self._check = check
        self._clock = clock
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._cached: tuple[float, bool] | None = None

    def ok(self, timeout: float, *, fresh: bool = False) -> bool:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return False  # an earlier probe is hung: do not start another
            if not fresh and self._cached and self._clock() - self._cached[0] < PROBE_CACHE_S:
                return self._cached[1]
            result: dict[str, bool] = {}

            def run() -> None:
                try:
                    result["ok"] = bool(self._check())
                except OSError:
                    result["ok"] = False

            self._thread = thread = threading.Thread(target=run, name="filing-probe", daemon=True)
            thread.start()
        thread.join(timeout)
        answer = result.get("ok", False)  # no answer in time counts as no
        with self._lock:
            self._cached = (self._clock(), answer)
        return answer


def _default_config() -> config.Filing:
    return config.get().filing


class Filing:
    def __init__(self, cfg: config.Filing | None = None, clock: Callable[[], float] = time.time):
        self.cfg = cfg or _default_config()
        self.staging = self.cfg.staging_root
        self.share = self.cfg.share_root
        self._clock = clock
        self._lock = threading.Lock()  # the counters below
        self._pending = 0
        self._pending_bytes = 0
        self._oldest: float | None = None  # mtime of the oldest waiting file
        self._delivered = 0
        self._last_delivery: float | None = None
        self._last_error: str | None = None
        self._share_state: ShareState = "unknown"
        self._reported: set[tuple[str, str]] = set()  # problems already logged, once each
        self._disk: tuple[float, float] | None = None  # (when, fraction used)
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._probe = Probe(self._share_writable)

    # -- producers ----------------------------------------------------------------------

    @contextlib.contextmanager
    def writing(self, relative: str | Path) -> Iterator[Path]:
        """Write one file: yields the path to write to; on a clean exit the file appears
        under ``relative`` in staging, whole, and delivery is woken. On an error nothing
        appears. Raises :class:`StagingFull` before yielding when the disk is at its ceiling.

        >>> with filing.writing("dct/20261107/vis.0001.fits") as path:
        ...     hdul.writeto(path)
        """
        final = self._inside(relative)
        used = self._disk_used()
        if used >= self.cfg.disk_ceiling:
            raise StagingFull(used, self.cfg.disk_ceiling)
        final.parent.mkdir(parents=True, exist_ok=True)
        partial = partial_of(final)
        try:
            yield partial
            os.replace(partial, final)
        except BaseException:
            with contextlib.suppress(OSError):
                partial.unlink()
            raise
        with self._lock:
            size = final.stat().st_size
            self._pending += 1
            self._pending_bytes += size
            self._oldest = self._oldest if self._oldest is not None else final.stat().st_mtime
        self._wake.set()

    def _inside(self, relative: str | Path) -> Path:
        path = (self.staging / relative).resolve()
        if not path.is_relative_to(self.staging.resolve()) or path == self.staging.resolve():
            raise ValueError(f"{relative!r} is not a file path inside the staging folder")
        return path

    # -- the share ----------------------------------------------------------------------

    def _share_writable(self) -> bool:
        """A folder can be made and removed in the share's root. Never creates the root."""
        if not self.share.is_dir():
            return False
        if self.cfg.share_is_mount and not os.path.ismount(self.share):
            return False
        probe = self.share / f".wilds-probe-{os.getpid()}-{threading.get_ident()}"
        probe.mkdir()
        probe.rmdir()
        return True

    def probe_share(self, *, fresh: bool = False) -> bool:
        online = self._probe.ok(self.cfg.probe_timeout_s, fresh=fresh)
        state: ShareState = "online" if online else "offline"
        with self._lock:
            before, self._share_state = self._share_state, state
        if state != before:
            if online:
                log.info("data share %s is reachable%s", self.share, self._backlog_text())
            elif before != "unknown" or self._pending:
                log.warning("data share %s is not reachable%s", self.share, self._backlog_text())
            else:
                log.info("data share %s is not reachable yet", self.share)
        return online

    def _backlog_text(self) -> str:
        return f"; {self._pending} file(s) waiting" if self._pending else ""

    # -- delivery -----------------------------------------------------------------------

    def waiting(self) -> list[Path]:
        """Whole files in staging, oldest first."""
        found: list[tuple[float, Path]] = []
        for folder, dirs, files in os.walk(self.staging):
            dirs[:] = [d for d in dirs if not is_bookkeeping(d)]
            for name in files:
                if not is_bookkeeping(name):
                    path = Path(folder) / name
                    with contextlib.suppress(OSError):
                        found.append((path.stat().st_mtime, path))
        return [p for _, p in sorted(found)]

    def rescan(self) -> None:
        """Recount what is waiting (the staging folder is the truth)."""
        files, total, oldest = 0, 0, None
        for path in self.waiting():
            with contextlib.suppress(OSError):
                st = path.stat()
                files, total = files + 1, total + st.st_size
                oldest = st.st_mtime if oldest is None else min(oldest, st.st_mtime)
        with self._lock:
            self._pending, self._pending_bytes, self._oldest = files, total, oldest

    def sweep(self) -> int:
        """One pass: probe, then deliver what is waiting. Returns how many were delivered."""
        self.rescan()
        if not self.probe_share(fresh=True):
            return 0
        with self._lock:
            self._last_error = None  # a problem still there is found again below
        delivered = 0
        for path in self.waiting():
            if self._stop.is_set():
                break
            try:
                if self._deliver(path):
                    delivered += 1
            except OSError as e:
                self._problem(path, f"{e.strerror or e}")
                if not self.probe_share(fresh=True):
                    break  # the share went away mid-sweep; the next sweep resumes
        self.rescan()
        return delivered

    def _deliver(self, source: Path) -> bool:
        """Copy to the share under a partial name, rename into place, then drop the source."""
        relative = source.relative_to(self.staging)
        target = self.share / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if filecmp.cmp(source, target, shallow=False):
                self._drop(source)  # already there, byte for byte
                return True
            self._problem(source, f"{target} exists and differs; not overwriting")
            return False
        partial = partial_of(target)
        try:
            shutil.copyfile(source, partial)
            if partial.stat().st_size != source.stat().st_size:
                raise OSError(errno.EIO, "the copy is a different size from the source")
            os.replace(partial, target)
        except BaseException:
            with contextlib.suppress(OSError):
                partial.unlink()
            raise
        self._drop(source)
        with self._lock:
            self._delivered += 1
            self._last_delivery = self._clock()
        log.info("delivered %s", relative)
        return True

    def _drop(self, source: Path) -> None:
        source.unlink()
        folder = source.parent
        while folder != self.staging and not any(folder.iterdir()):
            folder.rmdir()
            folder = folder.parent

    def _problem(self, source: Path, why: str) -> None:
        key = (str(source), why)
        with self._lock:
            self._last_error = f"{source.name}: {why}"
            first = key not in self._reported
            self._reported.add(key)
        if first:  # a clash is reported once, not every sweep
            log.error("cannot deliver %s: %s", source, why)

    # -- the thread ---------------------------------------------------------------------

    def start(self) -> None:
        """Make the staging folder, drop partials a crash left, and start sweeping."""
        if self._thread is not None:
            return
        self.staging.mkdir(parents=True, exist_ok=True)
        for folder, _, files in os.walk(self.staging):
            for name in files:
                if name.startswith(".") and name.endswith(PARTIAL_SUFFIX):
                    with contextlib.suppress(OSError):
                        (Path(folder) / name).unlink()  # never a product: whoever wrote it died
        self.rescan()
        if self._pending:
            log.info(
                "%d file(s) wait in staging from before; delivering when the share allows",
                self._pending,
            )
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="filing-sweeper", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.sweep()
            except Exception:
                log.exception("filing sweep failed")
            self._wake.wait(self.cfg.sweep_interval_s)
            self._wake.clear()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self._wake.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout)

    def flush(self, timeout: float = 30.0) -> bool:
        """Wait until nothing is waiting (for shutdown and tests). False on timeout."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.rescan()
            if self._pending == 0:
                return True
            self._wake.set()
            time.sleep(0.05)
        return False

    # -- disk and status ----------------------------------------------------------------

    def _disk_used(self) -> float:
        now = time.monotonic()
        if self._disk is None or now - self._disk[0] >= DISK_CACHE_S:
            self.staging.mkdir(parents=True, exist_ok=True)
            usage = shutil.disk_usage(self.staging)
            self._disk = (now, usage.used / usage.total)
        return self._disk[1]

    def status(self) -> FilingStatus:
        try:
            used = self._disk_used()
        except OSError:
            used = None
        with self._lock:
            oldest = None if self._oldest is None else max(0.0, self._clock() - self._oldest)
            return FilingStatus(
                share=self._share_state,
                share_root=str(self.share),
                staging_root=str(self.staging),
                pending=self._pending,
                pending_bytes=self._pending_bytes,
                oldest_age_s=oldest,
                delivered=self._delivered,
                last_delivery_ms=None
                if self._last_delivery is None
                else int(self._last_delivery * 1000),
                last_error=self._last_error,
                staging_used=used,
                staging_full=used is not None and used >= self.cfg.disk_ceiling,
            )


_filing: Filing | None = None


def get_filing() -> Filing:
    """The one instance, made on first use (the app starts its thread at startup). Use it as
    a FastAPI ``Depends``; tests override it through ``app.dependency_overrides``."""
    global _filing
    if _filing is None:
        _filing = Filing()
    return _filing


def close_filing() -> None:
    global _filing
    filing, _filing = _filing, None
    if filing is not None:
        filing.stop()


def reset() -> None:
    """Forget the instance, stopping its thread (for tests)."""
    close_filing()
