"""Where a session comes from, as far as we can tell (best effort).

Trustworthy: the TCP peer address, the origin label derived from it, and the hostname from
reverse DNS. Informational only: what the browser reports about itself (user agent, time
zone, page host, screen), which anyone can fake. Best effort: for a browser on this computer,
which desktop it runs in and, if that is a VNC desktop, who is viewing it (Linux, and only
if wilds-control may inspect the browser's process).
"""

import ipaddress
import logging
import os
import re
import socket
import threading

from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

# First TCP port of the VNC server; display :N listens on base + N (TigerVNC default).
VNC_PORT_BASE = int(os.environ.get("WILDS_CONTROL_VNC_PORT_BASE", "5900"))


def _origin_table() -> list[tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, str]]:
    """``WILDS_CONTROL_ORIGINS="10.0.0.0/8=Weizmann VPN;192.168.7.0/24=LDT"``."""
    table = []
    for item in os.environ.get("WILDS_CONTROL_ORIGINS", "").split(";"):
        if "=" not in item:
            continue
        net, label = item.split("=", 1)
        try:
            table.append((ipaddress.ip_network(net.strip(), strict=False), label.strip()))
        except ValueError:
            log.warning("ignoring bad WILDS_CONTROL_ORIGINS entry %r", item)
    return table


def clean(value: str | None, limit: int = 120) -> str | None:
    """Plain, bounded text: strings from browsers are shown to other people."""
    if not value:
        return None
    return "".join(c for c in value if c.isprintable())[:limit] or None


class Hints(BaseModel):
    """What the browser says about itself. Untrusted."""

    timezone: str | None = None
    page_host: str | None = None
    screen: str | None = None
    language: str | None = None


class Provenance(BaseModel):
    ip: str
    first_ip: str
    hostname: str | None = None
    origin: str | None = None
    local: bool = False
    user_agent: str | None = None
    hints: Hints = Field(default_factory=Hints)
    display: str | None = None
    vnc_viewers: list[str] = Field(default_factory=list)

    def summary(self) -> str:
        """Plain-text lines for a tooltip."""
        lines = []
        where = self.origin or ("this computer" if self.local else "unknown network")
        lines.append(f"From: {where}")
        addr = self.ip + (f" ({self.hostname})" if self.hostname else "")
        lines.append(f"Address: {addr}")
        if self.first_ip != self.ip:
            lines.append(f"First seen from: {self.first_ip}")
        if self.display == ":0":
            lines.append("Desktop: console (:0)")
        elif self.display:
            who = ", ".join(self.vnc_viewers) if self.vnc_viewers else "viewer unknown"
            lines.append(f"Desktop: VNC {self.display} (best effort: {who})")
        if self.user_agent:
            lines.append(f"Browser: {self.user_agent}")
        h = self.hints
        if h.timezone:
            lines.append(f"Time zone: {h.timezone}")
        if h.page_host:
            lines.append(f"Opened as: {h.page_host}")
        return "\n".join(lines)


def _local_addresses() -> set[str]:
    addrs = {"127.0.0.1", "::1"}
    try:
        import psutil

        for nics in psutil.net_if_addrs().values():
            addrs.update(a.address.split("%")[0] for a in nics)
    except Exception:  # noqa: BLE001 - best effort
        pass
    return addrs


def classify(ip: str) -> tuple[str | None, bool]:
    """(origin label, runs on this computer)."""
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return None, False
    local = addr.is_loopback or ip in _local_addresses()
    for net, label in _origin_table():
        if addr in net:
            return label, local
    return ("this computer" if local else None), local


def _fill_hostname(p: Provenance) -> None:
    ip = p.ip
    try:
        host = socket.gethostbyaddr(ip)[0]
    except (OSError, UnicodeError):
        return
    if p.ip == ip:
        p.hostname = clean(host)


def lookup_hostname_later(p: Provenance) -> None:
    """Reverse DNS can be slow, so it fills in ``p.hostname`` from a background thread."""
    threading.Thread(target=_fill_hostname, args=(p,), daemon=True).start()


def find_vnc(client_port: int | None, server_port: int | None) -> tuple[str | None, list[str]]:
    """For a connection from this computer: (X display of the browser, attached VNC viewers).

    Finds the process that owns the browser's end of the connection, reads its DISPLAY, and
    for a VNC display (:N, N >= 1) lists who is connected to the VNC port. Returns
    ``(None, [])`` whenever any step is unavailable (no psutil, not Linux, not permitted).
    """
    if client_port is None or server_port is None:
        return None, []
    try:
        import psutil

        conns = psutil.net_connections(kind="tcp")
        pid = next(
            (
                c.pid
                for c in conns
                if c.pid
                and c.laddr
                and c.raddr
                and c.laddr.port == client_port
                and c.raddr.port == server_port
            ),
            None,
        )
        if pid is None:
            return None, []
        m = re.fullmatch(r":(\d+)(?:\.\d+)?", psutil.Process(pid).environ().get("DISPLAY", ""))
        if m is None:
            return None, []
        n = int(m.group(1))
        if n == 0:
            return ":0", []
        port = VNC_PORT_BASE + n
        viewers = sorted(
            {
                c.raddr.ip
                for c in conns
                if c.status == "ESTABLISHED" and c.laddr and c.raddr and c.laddr.port == port
            }
        )
        return f":{n}", viewers
    except Exception as e:  # noqa: BLE001 - best effort, never break session creation
        log.debug("VNC detection unavailable: %s", e)
        return None, []


def collect(
    ip: str,
    port: int | None,
    server_port: int | None,
    user_agent: str | None,
    hints: Hints,
) -> Provenance:
    origin, local = classify(ip)
    display, viewers = find_vnc(port, server_port) if local else (None, [])
    p = Provenance(
        ip=ip,
        first_ip=ip,
        origin=origin,
        local=local,
        user_agent=clean(user_agent),
        hints=Hints(
            timezone=clean(hints.timezone, 60),
            page_host=clean(hints.page_host, 80),
            screen=clean(hints.screen, 20),
            language=clean(hints.language, 20),
        ),
        display=display,
        vnc_viewers=viewers,
    )
    lookup_hostname_later(p)
    return p


def moved(p: Provenance, ip: str) -> None:
    """The session's address changed (laptop moved networks): keep the first, show the latest."""
    if ip != p.ip:
        p.ip = ip
        p.origin, p.local = classify(ip)
        p.hostname = None
        lookup_hostname_later(p)
