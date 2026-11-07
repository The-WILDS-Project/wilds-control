"""Sessions and the control lease (plan §3.11).

One browser = one session. Any number may watch; at most one holds the *control lease*.
The holder is issued a secret lease token, which every command and configuration change
must present (see :func:`routers.control.require_lease`). Sessions that stop sending
heartbeats are dropped, and if the holder is dropped the lease lapses.

State is in memory: a restart of wilds-control drops every session and the lease.
"""

import logging
import os
import secrets
import string
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field

from . import config
from .provenance import Provenance

audit = logging.getLogger("wilds.control.audit")

# Defaults for the open decisions in plan §6; each can be overridden from the environment
# until the TOML configuration singleton exists.
MAX_SESSIONS = int(os.environ.get("WILDS_CONTROL_MAX_SESSIONS", "20"))
SESSION_TIMEOUT_S = float(os.environ.get("WILDS_CONTROL_SESSION_TIMEOUT_S", "30"))
NAME_MAX = 60


class LeaseError(Exception):
    """A refused session or lease operation; ``status`` is the HTTP status to answer with."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


@dataclass
class Session:
    id: str  # secret: only this session's browser knows it
    code: str  # public short code (``K7-42``), read out to the TO
    name: str
    last_seen: float
    requested_at: float | None = None
    refused: bool = False  # the TO refused this session's last request (until it asks again)
    refusal_reason: str | None = None
    prov: Provenance | None = None


@dataclass
class ControlState:
    sessions: dict[str, Session] = field(default_factory=dict)
    holder_id: str | None = None
    engineering: bool = False
    token: str | None = None
    to_seen: float | None = None  # last valid request from a Telescope Operator page
    to_logged_online: bool = False
    clock: Callable[[], float] = time.monotonic

    # -- helpers ------------------------------------------------------------------------

    @property
    def holder(self) -> Session | None:
        return self.sessions.get(self.holder_id) if self.holder_id else None

    def _label(self, s: Session) -> str:
        where = f" from {s.prov.origin or s.prov.ip}" if s.prov else ""
        return f"{s.name} [{s.code}]{where}"

    def _new_code(self) -> str:
        used = {s.code for s in self.sessions.values()}
        while True:
            code = (
                secrets.choice(string.ascii_uppercase)
                + secrets.choice(string.digits)
                + "-"
                + "".join(secrets.choice(string.digits) for _ in range(2))
            )
            if code not in used:
                return code

    def _clear_lease(self) -> None:
        self.holder_id = None
        self.engineering = False
        self.token = None

    def _give_lease(self, s: Session, *, engineering: bool) -> None:
        self.holder_id = s.id
        self.engineering = engineering
        self.token = secrets.token_urlsafe(24)
        s.requested_at = None
        s.refused, s.refusal_reason = False, None

    @property
    def to_online(self) -> bool:
        """A Telescope Operator page is open and unlocked (it polls with the passphrase)."""
        return self.to_seen is not None and (
            self.clock() - self.to_seen <= config.get().presence.to_timeout_s
        )

    def touch_to(self) -> None:
        self.to_seen = self.clock()
        if not self.to_logged_online:
            self.to_logged_online = True
            audit.info("Telescope Operator online")

    def expire(self) -> None:
        """Drop sessions that missed their heartbeats; lapse the lease if the holder went."""
        if self.to_logged_online and not self.to_online:
            self.to_logged_online = False
            audit.info("Telescope Operator offline (no contact)")
        now = self.clock()
        for s in [s for s in self.sessions.values() if now - s.last_seen > SESSION_TIMEOUT_S]:
            del self.sessions[s.id]
            audit.info("session dropped (no heartbeat): %s", self._label(s))
            if s.id == self.holder_id:
                audit.info("lease lapsed: holder %s is gone", self._label(s))
                self._clear_lease()

    def get(self, session_id: str | None) -> Session:
        self.expire()
        s = self.sessions.get(session_id or "")
        if s is None:
            raise LeaseError(404, "Unknown or expired session; create a new one")
        s.last_seen = self.clock()
        return s

    def by_code(self, code: str) -> Session:
        self.expire()
        for s in self.sessions.values():
            if s.code == code.upper():
                return s
        raise LeaseError(404, f"No session with code {code}")

    # -- sessions -----------------------------------------------------------------------

    def create_session(self, name: str, prov: Provenance | None = None) -> Session:
        self.expire()
        name = " ".join(name.split())[:NAME_MAX]
        if not name:
            raise LeaseError(422, "A name is required")
        if len(self.sessions) >= MAX_SESSIONS:
            raise LeaseError(503, f"WILDS is at its limit of {MAX_SESSIONS} sessions; try later")
        s = Session(
            id=str(uuid.uuid4()),
            code=self._new_code(),
            name=name,
            last_seen=self.clock(),
            prov=prov,
        )
        self.sessions[s.id] = s
        audit.info("session created: %s from %s", self._label(s), prov.ip if prov else "?")
        return s

    # -- requesting, granting, releasing -----------------------------------------------

    def request(self, s: Session) -> None:
        s.requested_at = self.clock()  # re-applying moves the request to the top
        s.refused, s.refusal_reason = False, None
        audit.info("control requested by %s", self._label(s))

    def grant(self, code: str) -> Session:
        s = self.by_code(code)
        prev = self.holder
        if prev is not None and prev.id != s.id:
            audit.info("handover: %s -> %s", self._label(prev), self._label(s))
        self._give_lease(s, engineering=False)
        audit.info("control granted to %s", self._label(s))
        return s

    def refuse(self, code: str, reason: str) -> None:
        s = self.by_code(code)
        s.requested_at = None
        s.refused, s.refusal_reason = True, (reason or None)
        audit.info("control refused for %s: %s", self._label(s), reason or "no reason given")

    def revoke(self) -> None:
        self.expire()
        h = self.holder
        if h is not None:
            audit.info("control revoked from %s", self._label(h))
        self._clear_lease()

    def release(self, s: Session) -> None:
        if s.id != self.holder_id:
            raise LeaseError(403, "You do not hold control")
        audit.info("control released by %s", self._label(s))
        self._clear_lease()

    def take_engineering(self, s: Session) -> None:
        """Take control in engineering mode. The caller has checked the passphrase."""
        prev = self.holder
        if prev is not None and prev.id != s.id:
            audit.info("handover: %s -> %s (engineering)", self._label(prev), self._label(s))
        self._give_lease(s, engineering=True)
        audit.info("engineering mode entered by %s", self._label(s))

    # -- checking -----------------------------------------------------------------------

    def check_token(self, token: str | None) -> None:
        self.expire()
        if self.token is None or token is None or not secrets.compare_digest(token, self.token):
            raise LeaseError(403, "You do not hold control (missing or stale lease token)")

    def requests(self) -> list[Session]:
        """Pending requests, most recent first."""
        pending = [s for s in self.sessions.values() if s.requested_at is not None]
        return sorted(pending, key=lambda s: s.requested_at or 0.0, reverse=True)


state = ControlState()
