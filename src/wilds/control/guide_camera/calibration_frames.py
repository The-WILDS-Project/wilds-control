"""The guide camera's dark and bias frames, kept on the local disk (plan §3.17).

Autofocus, guiding correlations and the like want a dark or bias frame taken with the camera's own
settings. They are kept in *calibration sessions*, folders named by the UTC time they were
started, so a set can be put aside and brought back::

    $WILDS_ROOT/var/lib/guide-camera/calibration/
        current                       the id of the active session (a text file)
        20261111T093012Z/
            index.json                what the session holds
            dark-0001/frame-01.fits … master.fits
            bias-0002/…

An *entry* is one acquisition: N raw frames and their master, the median of the N (a float32
FITS file, with ``NCOMBINE``). Every file is written whole under a temporary name and renamed
into place, so a reader never sees half a file. :meth:`CalibrationStore.find_master` is what a
consumer asks: the master of the active session that matches its settings (and, for a dark,
its exposure), nearest in sensor temperature.

This is engineering data kept for reuse, not facility data: it stays here and is never delivered
to the share (``filing.py``).
"""

import os
import shutil
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from ..errors import ApiError
from .base import CalibrationKind, CaptureSettings, Frame, FrameMeta, Roi

MAX_DARK_TEMPERATURE_GAP_C = 3.0  # a master this much off the sensor's temperature is not offered


class Entry(BaseModel):
    id: str  # "dark-0001"
    kind: CalibrationKind
    taken_ms: int  # when the last frame ended
    count: int
    exposure_s: float
    gain: int
    offset: int
    binning: int
    roi: Roi | None
    temperature_c: float | None  # the mean over the frames
    temperatures_c: list[float] = []
    match: str  # the settings group these settings followed ("single_frame", "video")
    frames: list[str]  # file names inside the entry's folder
    master: str


class SessionInfo(BaseModel):
    id: str
    created_ms: int
    label: str | None = None
    active: bool = False
    entries: list[Entry] = []


class CalibrationStore:
    def __init__(self, root: Path):
        self.root = root
        self._lock = threading.Lock()

    # -- sessions -----------------------------------------------------------------------

    def sessions(self) -> list[SessionInfo]:
        """Every session, newest first, with the active one marked."""
        active = self.active_id()
        out = []
        for folder in sorted((p for p in self._folders()), reverse=True):
            info = self._read(folder.name)
            if info is not None:
                info.active = info.id == active
                out.append(info)
        return out

    def active_id(self) -> str | None:
        try:
            sid = (self.root / "current").read_text().strip()
        except OSError:
            return None
        return sid if sid and (self.root / sid / "index.json").is_file() else None

    def new_session(self, label: str | None = None) -> SessionInfo:
        """Start a session and make it the active one."""
        with self._lock:
            now = datetime.now(UTC)
            sid = now.strftime("%Y%m%dT%H%M%SZ")
            n = 1
            while (self.root / sid).exists():  # two within a second
                n += 1
                sid = f"{now.strftime('%Y%m%dT%H%M%SZ')}-{n}"
            info = SessionInfo(id=sid, created_ms=int(now.timestamp() * 1000), label=label)
            (self.root / sid).mkdir(parents=True)
            self._write_index(info)
            self._set_active(sid)
        info.active = True
        return info

    def activate(self, session_id: str) -> SessionInfo:
        with self._lock:
            info = self._read(session_id)
            if info is None:
                raise ApiError(404, "not_found", f"no calibration session {session_id}")
            self._set_active(session_id)
        info.active = True
        return info

    def ensure_session(self) -> str:
        """The active session's id; one is started if there is none."""
        return self.active_id() or self.new_session().id

    # -- entries ------------------------------------------------------------------------

    def add(
        self, session_id: str, kind: CalibrationKind, frames: list[Frame], *, match: str
    ) -> Entry:
        """File ``frames`` (all taken with the same settings) and their median as a new entry."""
        if not frames:
            raise ValueError("no frames to file")
        with self._lock:
            info = self._read(session_id)
            if info is None:
                raise ApiError(404, "not_found", f"no calibration session {session_id}")
            # one more than the highest number used (a discarded entry leaves a gap, never a clash)
            number = 1 + max((int(e.id.rsplit("-", 1)[1]) for e in info.entries), default=0)
            entry_id = f"{kind}-{number:04d}"
            folder = self.root / session_id / entry_id
            folder.mkdir()
            names = []
            for i, frame in enumerate(frames, 1):
                name = f"frame-{i:02d}.fits"
                _write(folder / name, frame.fits())
                names.append(name)
            master = _median(frames)
            _write(folder / "master.fits", master.fits({"NCOMBINE": (len(frames), "frames")}))
            meta = frames[-1].meta
            temps = [f.meta.temperature_c for f in frames if f.meta.temperature_c is not None]
            entry = Entry(
                id=entry_id,
                kind=kind,
                taken_ms=meta.taken_ms,
                count=len(frames),
                exposure_s=meta.exposure_s,
                gain=meta.gain,
                offset=meta.offset,
                binning=meta.binning,
                roi=meta.roi,
                temperature_c=round(sum(temps) / len(temps), 2) if temps else None,
                temperatures_c=[round(t, 2) for t in temps],
                match=match,
                frames=names,
                master="master.fits",
            )
            info.entries.append(entry)
            self._write_index(info)
        return entry

    def delete_entry(self, session_id: str, entry_id: str) -> SessionInfo:
        """Discard an entry: its frames and its master are deleted from the disk. The session
        stays, empty if it was the last. The index is changed first, so a failure part-way leaves
        an unlisted folder behind, never a listed entry with missing files."""
        with self._lock:
            info = self._read(session_id)
            entry = next((e for e in info.entries if e.id == entry_id), None) if info else None
            if info is None or entry is None:
                raise ApiError(404, "not_found", f"no entry {session_id}/{entry_id}")
            info.entries = [e for e in info.entries if e.id != entry_id]
            self._write_index(info)
            shutil.rmtree(self.root / session_id / entry_id, ignore_errors=True)
        info.active = info.id == self.active_id()
        return info

    def path(self, session_id: str, entry_id: str, name: str) -> Path:
        """A filed FITS file; only names the index lists are served (no path can escape)."""
        info = self._read(session_id)
        entry = next((e for e in info.entries if e.id == entry_id), None) if info else None
        if entry is None or name not in (*entry.frames, entry.master):
            raise ApiError(404, "not_found", f"no file {session_id}/{entry_id}/{name}")
        return self.root / session_id / entry_id / name

    def find_master(
        self,
        kind: CalibrationKind,
        settings: CaptureSettings,
        *,
        temperature_c: float | None = None,
        max_gap_c: float = MAX_DARK_TEMPERATURE_GAP_C,
    ) -> tuple[Entry, Path] | None:
        """The active session's master for ``kind`` taken with the same gain, offset, binning and
        window as ``settings`` (and, for a dark, the same exposure); the nearest in temperature
        to ``temperature_c`` (the newest if none is given). ``None`` if nothing fits."""
        sid = self.active_id()
        info = self._read(sid) if sid else None
        if info is None:
            return None
        fits = [e for e in info.entries if e.kind == kind and _same(e, kind, settings)]
        if temperature_c is not None:
            fits = [
                e
                for e in fits
                if e.temperature_c is not None and abs(e.temperature_c - temperature_c) <= max_gap_c
            ]
            fits.sort(key=lambda e: abs(e.temperature_c - temperature_c))
        else:
            fits.sort(key=lambda e: -e.taken_ms)
        if not fits:
            return None
        return fits[0], self.root / sid / fits[0].id / fits[0].master

    # -- on disk ------------------------------------------------------------------------

    def _folders(self):
        try:
            return [p for p in self.root.iterdir() if p.is_dir()]
        except OSError:
            return []

    def _read(self, session_id: str) -> SessionInfo | None:
        if not session_id or "/" in session_id or session_id.startswith("."):
            return None
        try:
            return SessionInfo.model_validate_json(
                (self.root / session_id / "index.json").read_text()
            )
        except OSError, ValueError:
            return None

    def _write_index(self, info: SessionInfo) -> None:
        data = info.model_copy(update={"active": False}).model_dump_json(indent=1)
        _write(self.root / info.id / "index.json", data.encode())

    def _set_active(self, session_id: str) -> None:
        _write(self.root / "current", session_id.encode())


def _same(e: Entry, kind: CalibrationKind, s: CaptureSettings) -> bool:
    same = e.gain == s.gain and e.offset == s.offset and e.binning == s.binning
    same = same and (e.roi == s.roi)
    if kind == "dark":
        same = same and abs(e.exposure_s - s.exposure_s) <= 1e-9 * max(1.0, s.exposure_s)
    return same


def _median(frames: list[Frame]) -> Frame:
    """The per-pixel median of frames of one size, as a float32 frame."""
    stack = np.stack([f.pixels for f in frames])
    pixels = np.median(stack, axis=0).astype(np.float32)
    meta: FrameMeta = frames[-1].meta
    temps = [f.meta.temperature_c for f in frames if f.meta.temperature_c is not None]
    meta = meta.model_copy(
        update={"temperature_c": sum(temps) / len(temps) if temps else None, "id": 0}
    )
    return Frame(meta=meta, jpeg=b"", pixels=pixels)


def _write(path: Path, data: bytes) -> None:
    """Write the file whole: a temporary name, then renamed into place."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    with tmp.open("wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def now_ms() -> int:
    return time.time_ns() // 1_000_000


def store() -> CalibrationStore:
    """The store under ``$WILDS_ROOT``; made on use, so nothing is created until a frame is."""
    from .. import paths

    root = paths.guide_camera_calibration()
    root.mkdir(parents=True, exist_ok=True)
    return CalibrationStore(root)


__all__ = ["CalibrationStore", "Entry", "SessionInfo", "now_ms", "store"]
