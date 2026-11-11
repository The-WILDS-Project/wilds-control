"""Resolve an object name to J2000 coordinates.

Copied from MAST's ``common/object_resolver.py`` (MAST_common, 2026-10) and kept close to it:
the changes are the logger (the standard library's), the proxy lookup (the environment's
``http_proxy`` / ``https_proxy``; MAST's ``common.internet`` is not carried), and the TNS
credentials, which come from the environment (``WILDS_TNS_API_KEY``, ``WILDS_TNS_ID``,
``WILDS_TNS_BOT_NAME``) instead of MAST's vault; without them TNS is skipped and Sesame
answers, as the design intends. Used by ``GET /resolve`` and so by the target entry on the
Instrument Operator's Acquisition tab (plan §3.3).

Standalone by design: it imports nothing else from WILDS.

Two serial gates, then every service at once.

The **gates** decide whether to touch the network at all, and both are free:

1. **Moving targets are refused without a network call.** Comets and minor planets have
   no fixed J2000 -- only an ephemeris at an instant -- so a plausible answer would be
   worse than none. Refusing by pattern also stops such a name reaching a catalogue that
   might match it for unrelated reasons.
2. **Cache.**

Past them, **TNS, Sesame/CDS and Sesame/CfA are asked simultaneously**. Nothing is asked
speculatively -- the gates already decided -- but once the decision is made there is no
reason for one service to queue behind another. Two rules govern who wins:

* The **Sesame mirrors race**; first positive answer wins. They are interchangeable, so
  that is not a shortcut but the correct reading of "whichever answers first". A miss from
  one is not the verdict -- only when every mirror says no is the miss definitive, and a
  definitive miss is the one failure this module may remember.
* **TNS is preferred, not raced.** For a TNS-shaped name (`AT`/`SN` + year) its verdict is
  awaited before Sesame's is looked at. Taking whoever answers soonest would be WRONG:
  Sesame is the faster of the two, so first-wins would systematically prefer the staler
  source for exactly the names TNS exists to answer. A TNS miss or outage falls through --
  a name that merely looks like a transient may be an older object Sesame knows -- and
  because Sesame was started at the same instant, falling through costs nothing.

This was sequential until 2026-08-26, and the arrangement hid a fallback that could not
work. With the mirrors tried in turn behind TNS, the per-mirror cap had to be 4s to fit the
budget -- but measured over 20 minutes from mast00, the CfA mirror's FASTEST response was
9.27s (median 9.59s, and a third of responses near 20.5s). CDS answers in 1.97s. So the
fallback existed in the code and never once in reality, and the cost of finding that out
was borne by whoever was on shift when CDS refused a query.

A note on why TNS is preferred, since the reasoning was wrong at first and the correction
matters: it is NOT that Sesame returns a transient's host galaxy instead of the transient.
Measured on 2026-08-12, SIMBAD gives SN2023ixf its own position, 4.39 arcmin from M101's
nucleus. The real gap is recency -- a transient reported last night is in TNS and not yet
in SIMBAD -- so Sesame's failure mode for a fresh transient is "not found", which is loud
and safe. That is a weaker argument for TNS-first than the one originally recorded in
MAST_common#60, and it is why falling through to Sesame is correct rather than dangerous.

Every result carries where it came from. A misresolution is indistinguishable from a
correct one at every layer below this -- the mount slews normally, guiding locks, and a
spectrum is taken of the wrong object -- so the provenance is the only way to work out
afterwards why a night pointed where it did.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
import urllib.request
from dataclasses import asdict, dataclass
from urllib.parse import quote, urlsplit

logger = logging.getLogger(__name__)


def internet_proxy(url: str) -> str | None:
    """The proxy for `url`'s scheme from the environment, or None to go direct. For httpx:
    pass it as ``proxy=`` with ``trust_env=False``."""
    return urllib.request.getproxies_environment().get(urlsplit(url).scheme)


#: Whole-enquiry budget. The callers are operator-facing or sweep-driven; past this a
#: human assumes it is broken, and a sweeper has other targets to get to.
#:
#: A DEADLINE over the whole enquiry, not a sum of its parts. Every service is asked at
#: once, so the budget bounds the slowest arm rather than their total.
TOTAL_TIMEOUT_SECONDS = 15.0

#: One POST to one host.
TNS_TIMEOUT_SECONDS = 5.0

#: Per MIRROR, and the two are asked simultaneously, so this is the loser's ceiling rather
#: than a cost paid on every enquiry. Generous on purpose: measured from mast00 over 20
#: minutes (70 samples, 2026-08-26), CDS answers in 1.97s median and never took 2.61s,
#: while the CfA mirror is bimodal -- 9.59s median, but a third of responses land near
#: 20.5s and nothing at all arrives between 10s and 20s. So a cap below ~10s makes CfA
#: unreachable by construction, and one between 12s and 20s buys nothing over 12s.
#:
#: It was 4.0 when the mirrors were asked in turn, which meant the fallback could NEVER
#: answer: CfA's fastest observed response was 9.27s. The fallback existed in the code and
#: not in reality.
SESAME_TIMEOUT_SECONDS = 25.0

#: Sesame mirrors, asked CONCURRENTLY -- first positive answer wins.
#:
#: They are interchangeable: same protocol, same underlying databases, neither more
#: authoritative, so racing them is not a shortcut but the correct reading of "whichever
#: answers first". Asking them in turn made the whole enquiry as slow as the slower host
#: whenever the first failed; asking together costs the happy path ~0.4s of thread and
#: parse overhead and makes the fallback real.
#:
#: Note CDS's load is unchanged by this -- one query per enquiry either way. It is CfA that
#: goes from near-zero traffic to one query per enquiry.
SESAME_MIRRORS = (
    ("cds", "https://cds.unistra.fr/cgi-bin/nph-sesame/-oI/A?"),
    ("cfa", "http://vizier.cfa.harvard.edu/viz-bin/nph-sesame/-oI/A?"),
)

#: A fixed object's position does not change; the cost of caching it is a stale name, not
#: a stale sky. Long, but not forever, so a corrected catalogue entry is eventually seen.
CATALOGUE_TTL_SECONDS = 30 * 24 * 3600.0

#: TNS positions are refined after discovery as astrometry improves, so an early answer
#: must not be pinned.
TNS_TTL_SECONDS = 3600.0

#: Short, deliberately. A transient reported ten minutes from now must not stay
#: unresolvable because a miss was cached hard -- for follow-up that is the wrong thing
#: to remember. Long enough only to stop a sweeper re-asking on every pass.
#:
#: Applied ONLY to a definitive miss -- the service answered and said no such object. A
#: timeout, a throttle or an outage is never remembered: see `_is_definitive_miss`.
NEGATIVE_TTL_SECONDS = 300.0

TNS_OBJECT_URL = "https://www.wis-tns.org/api/get/object"


class ObjectNameError(Exception):
    """A name could not be turned into coordinates."""


class MovingTargetError(ObjectNameError):
    """The name designates something with no fixed J2000 position."""


class SesameMissError(ObjectNameError):
    """Sesame answered, and there is no such object.

    Distinct from any transport failure ON PURPOSE, and the distinction is the whole point
    of `NEGATIVE_TTL_SECONDS`: only a service that actually answered may have its "no"
    remembered. A timeout or an outage says nothing about whether the object exists.

    This used to be inferred by matching astropy's error text ("unable to find coordinates
    for name") against its other one ("All Sesame queries failed"). That worked but was
    one upstream rewording away from silently caching outages as missing objects -- the
    2026-08-12 `NGC 224` incident, in reverse. Reading it off the response structure
    instead cannot drift: a miss is a 200 with no `%J` line in it.
    """


@dataclass(frozen=True)
class ResolvedObject:
    """Where a name resolved to, and who said so."""

    name: str
    ra_j2000_hours: float
    dec_j2000_degs: float
    resolver: str
    """Which service answered -- "tns", or "sesame" (SIMBAD/NED/VizieR internally)."""
    canonical_name: str | None = None
    """The identifier the service matched, when it reports one. Worth recording: a name
    that resolved via an alias is the case most likely to have resolved to the wrong
    thing."""
    database: str | None = None
    """For Sesame, which database answered and on which mirror -- `simbad@cds`.

    `resolver` says "sesame", which is true but coarse: Sesame is three catalogues in a
    trench coat, and SIMBAD and NED disagreeing about a name is precisely the shape of a
    misresolution. Optional, so a result built without it is still valid."""

    def as_dict(self) -> dict:
        return asdict(self)


class _Arm:
    """One service being asked, on its own daemon thread.

    Daemon threads rather than a `ThreadPoolExecutor` deliberately. The executor's threads
    are joined at interpreter exit, so a losing arm still waiting on a 25s mirror would
    hold up the shutdown of a service that already has its answer. These are abandoned the
    moment they stop being interesting.
    """

    __slots__ = ("_done", "_error", "_value")

    def __init__(self) -> None:
        self._done = threading.Event()
        self._value: ResolvedObject | None = None
        self._error: BaseException | None = None

    def _run(self, fn, args) -> None:
        try:
            self._value = fn(*args)
        except BaseException as e:  # noqa: BLE001 -- carried to the waiter, never swallowed
            # Deliberately BaseException, not Exception. Nothing raised on this thread has
            # anywhere else to go: an escape would print to stderr and vanish, leaving the
            # waiter to time out against an arm that died instantly.
            self._error = e
        finally:
            self._done.set()

    def wait(self, timeout: float) -> bool:
        return self._done.wait(max(0.0, timeout))

    @property
    def finished(self) -> bool:
        return self._done.is_set()

    def result(self) -> ResolvedObject | None:
        if self._error is not None:
            raise self._error
        return self._value


def _spawn(fn, *args) -> _Arm:
    arm = _Arm()
    # getattr, because `fn` is not always a function: the tests substitute callable
    # objects, which have no __name__, and a thread's label is not worth an AttributeError.
    label = getattr(fn, "__name__", type(fn).__name__)
    threading.Thread(target=arm._run, args=(fn, args), daemon=True, name=f"resolve-{label}").start()
    return arm


def _as_they_finish(arms, deadline: float):
    """Yield (label, arm) as each finishes, giving up at `deadline`.

    A short poll rather than a condition variable shared between arms: with at most three
    of them the simplicity is worth more than the microseconds, and it keeps each arm
    independent of the others' bookkeeping.
    """
    pending = list(arms)
    while pending:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        for entry in list(pending):
            if entry[1].finished:
                pending.remove(entry)
                yield entry
                break
        else:
            time.sleep(min(0.02, max(0.0, remaining)))


# Comets: `C/2023 A3`, `1P/Halley`, `73P-C`. Minor planets: `(433)`, `433 Eros`,
# provisional `2024 AB`, `2024 AB1`. Deliberately narrow -- a false positive refuses a
# real object, which is worse than a false negative falling through to a catalogue that
# will simply not find it.
_COMET = re.compile(r"^\s*\d*[PDCXAI][/-]\s*", re.IGNORECASE)
_MINOR_PLANET_PROVISIONAL = re.compile(r"^\s*\(?\s*\d{4}\s*\)?\s+[A-Z]{2}\d*\s*$")
_MINOR_PLANET_NUMBERED = re.compile(r"^\s*\(\s*\d+\s*\)")
_SOLAR_SYSTEM_BODIES = frozenset(
    [
        "mercury",
        "venus",
        "mars",
        "jupiter",
        "saturn",
        "uranus",
        "neptune",
        "pluto",
        "sun",
        "moon",
        "luna",
        "sol",
        "ceres",
        "vesta",
        "pallas",
        "juno",
        "io",
        "europa",
        "ganymede",
        "callisto",
        "titan",
        "enceladus",
        "phobos",
        "deimos",
    ]
)

# `AT2024xyz`, `SN2023ixf`, `SN 1987A`.
_TNS_NAME = re.compile(r"^\s*(AT|SN)\s*(\d{4})\s*([A-Za-z]{1,4})\s*$", re.IGNORECASE)


def _is_moving_target(name: str) -> bool:
    stripped = name.strip()
    return bool(
        stripped.lower() in _SOLAR_SYSTEM_BODIES
        or _COMET.match(stripped)
        or _MINOR_PLANET_PROVISIONAL.match(stripped)
        or _MINOR_PLANET_NUMBERED.match(stripped)
    )


def is_tns_name(name: str) -> bool:
    """True for `AT`/`SN` + year + letters -- the shape TNS assigns."""
    return bool(_TNS_NAME.match(name))


class _Cache:
    """Name -> (result or None) with a per-entry expiry.

    In memory and process-local on purpose. The durable record of a resolution belongs in
    the plan that needed it -- explicit, auditable, and never re-resolved -- rather than in
    a shared cache where a wrong answer would be served to the whole fleet with no trail.
    """

    def __init__(self) -> None:
        self._entries: dict[str, tuple[float, ResolvedObject | None]] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _key(name: str) -> str:
        return " ".join(name.split()).casefold()

    def get(self, name: str) -> tuple[bool, ResolvedObject | None]:
        """(hit, value). A hit whose value is None is a remembered failure."""
        with self._lock:
            entry = self._entries.get(self._key(name))
            if entry is None:
                return False, None
            expires_at, value = entry
            if time.monotonic() >= expires_at:
                del self._entries[self._key(name)]
                return False, None
            return True, value

    def put(self, name: str, value: ResolvedObject | None, ttl: float) -> None:
        with self._lock:
            self._entries[self._key(name)] = (time.monotonic() + ttl, value)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


_cache = _Cache()


def _tns_credentials() -> tuple[str, str, str] | None:
    """(api_key, tns_id, bot_name) from the environment, or None if all three are not set.

    Requests need every one of them: the key authenticates, and the id and name go into
    the `tns_marker` User-Agent, without which TNS rejects the request.
    """
    api_key = os.environ.get("WILDS_TNS_API_KEY")
    tns_id = os.environ.get("WILDS_TNS_ID")
    bot_name = os.environ.get("WILDS_TNS_BOT_NAME")
    if not (api_key and tns_id and bot_name):
        return None
    return api_key, tns_id, bot_name


def _is_definitive_miss(error: BaseException) -> bool:
    """True when Sesame answered and said there is no such object.

    Now a type check rather than a string match. It used to read astropy's error text,
    distinguishing "Unable to find coordinates for name ..." from "All Sesame queries
    failed ...", which worked but was one upstream rewording away from remembering an
    outage as a missing object.

    Why that matters, from 2026-08-12: after several queries in quick succession CDS
    refused one, `NGC 224` -- which is M31 -- came back unresolved, and the miss was
    cached. It resolved in 0.83s on the next attempt. Only a service that ANSWERED may
    have its "no" remembered.
    """
    return isinstance(error, SesameMissError)


# Sesame's `-oI/A` reply is a per-database transcript. Each section opens with `#=<code>`
# when that database answered or `#!<code>` when it did not, and a section that answered
# carries `%J <ra_degrees> <dec_degrees>` and `%I.0 <primary identifier>`. Verified against
# the live service 2026-08-26 for a hit, a miss, a name with spaces and an NED-only object.
_SESAME_COORDINATES = re.compile(r"^%J\s+([-+\d.]+)\s+([-+\d.]+)", re.MULTILINE)
_SESAME_SECTION = re.compile(r"^#[=!](\w+)=(\w+)", re.MULTILINE)
_SESAME_IDENTIFIER = re.compile(r"^%I\.0\s+(.+?)\s*$", re.MULTILINE)


def _parse_sesame(name: str, text: str, mirror: str) -> ResolvedObject | None:
    """A result, or None when the transcript shows no database found anything.

    None means a DEFINITIVE miss -- the service answered and said no. A malformed or
    truncated reply raises instead, because "we could not read the answer" must never be
    remembered as "the object does not exist".
    """
    coordinates = _SESAME_COORDINATES.search(text)
    if coordinates is None:
        return None

    # Which database actually answered: the last section header before the coordinates.
    # Worth recording -- SIMBAD and NED disagreeing about a name is exactly the shape of a
    # misresolution, and the flat "sesame" it used to report could not show that.
    database = None
    for section in _SESAME_SECTION.finditer(text):
        if section.start() > coordinates.start():
            break
        database = section.group(2)

    identifier = _SESAME_IDENTIFIER.search(
        text[: coordinates.start()] + text[coordinates.start() :]
    )
    return ResolvedObject(
        name=name,
        ra_j2000_hours=float(coordinates.group(1)) / 15.0,
        dec_j2000_degs=float(coordinates.group(2)),
        resolver="sesame",
        canonical_name=" ".join(identifier.group(1).split()) if identifier else None,
        database=f"{database.lower()}@{mirror}" if database else mirror,
    )


def _query_sesame_mirror(name: str, mirror: str, url: str, timeout: float) -> ResolvedObject:
    """One mirror. Returns a result, or raises -- `SesameMissError` for a definitive no."""
    import httpx

    # The proxy is looked up here (trust_env=False), not by httpx.
    response = httpx.get(
        url + quote(name),
        timeout=timeout,
        follow_redirects=True,
        proxy=internet_proxy(url),
        trust_env=False,
    )
    response.raise_for_status()
    resolved = _parse_sesame(name, response.text, mirror)
    if resolved is None:
        raise SesameMissError(f"sesame/{mirror}: no database has '{name}'")
    return resolved


def _resolve_via_sesame(name: str, timeout: float) -> ResolvedObject:
    """Both mirrors at once; the first positive answer wins.

    A miss from one mirror does NOT end the enquiry -- the other is still asked out, and
    only when every mirror has said no is the miss definitive. The two are not quite
    identical (CDS runs VizieR locally), and a `SesameMissError` is the one failure this module
    is allowed to remember, so it must not be concluded from a single host.

    Raises `SesameMissError` only if every mirror answered and none found anything; otherwise
    the last transport failure, which is never remembered.
    """
    arms = [
        (mirror, _spawn(_query_sesame_mirror, name, mirror, url, timeout))
        for mirror, url in SESAME_MIRRORS
    ]
    deadline = time.monotonic() + timeout
    misses, failures = [], []

    for mirror, arm in _as_they_finish(arms, deadline):
        try:
            return arm.result()
        except SesameMissError as miss:
            misses.append(str(miss))
        except Exception as e:  # noqa: BLE001 -- one mirror's failure is not the verdict
            failures.append(f"{mirror}: {type(e).__name__}: {e}")

    if failures:
        raise ObjectNameError(f"sesame: no mirror answered ({'; '.join(failures)})")
    if misses:
        raise SesameMissError(f"sesame: no database has '{name}' ({len(misses)} mirror(s) agreed)")
    raise ObjectNameError(f"sesame: no mirror answered within {timeout:g}s")


def _describe(result: ResolvedObject) -> str:
    canonical = f" as '{result.canonical_name}'" if result.canonical_name else ""
    return f"ra={result.ra_j2000_hours:.6f}h dec={result.dec_j2000_degs:+.6f}d via {result.resolver}{canonical}"


def _try_tns_arm(name: str, deadline: float) -> ResolvedObject | None:
    """TNS's answer, or None to fall through to Sesame. Runs on its own thread.

    Raises nothing the caller has to special-case: a TNS outage, a missing credential or a
    miss all mean the same thing -- use Sesame -- and a name that merely looks like a
    transient may be an older object Sesame knows. The failure reason is carried on the
    exception for `_collect_tns` to record.
    """
    credentials = _tns_credentials()
    if credentials is None:
        # Not fatal and not a reason to refuse: Sesame carries transients once they are
        # catalogued, and a fresh one is simply not found, which is safe.
        raise ObjectNameError("no TNS credentials configured")
    return _resolve_via_tns(
        name, credentials, min(TNS_TIMEOUT_SECONDS, deadline - time.monotonic())
    )


def _collect_tns(arm: _Arm, attempts: list[str]) -> ResolvedObject | None:
    """TNS's result, or None -- for any reason at all -- meaning "use Sesame"."""
    try:
        resolved = arm.result()
    except Exception as e:  # noqa: BLE001 -- every TNS failure falls through to Sesame
        attempts.append(f"tns: {e}")
        logger.info(f"TNS did not answer ({e}); using Sesame")
        return None

    if resolved is None:
        attempts.append("tns: no match")
    return resolved


def _collect_sesame(
    name: str, arm: _Arm, attempts: list[str]
) -> tuple[ResolvedObject | None, bool]:
    """(result, definitive). `definitive` says whether a failure may be remembered.

    Never raises: an unexpected exception from a catalogue client must not surprise a
    caller that is already handling ObjectNameError.
    """
    try:
        return arm.result(), True
    except SesameMissError as e:
        attempts.append(f"sesame: {e}")
        return None, True
    except ObjectNameError as e:
        attempts.append(f"sesame: {e}")
        return None, False
    except Exception as e:
        attempts.append(f"sesame: {type(e).__name__}: {e}")
        logger.exception(f"unexpected failure resolving '{name}' through Sesame")
        return None, False


def _remembered(asked: str) -> ResolvedObject | None:
    """The remembered answer, or None if nothing is remembered.

    Raises for a remembered MISS -- that is the point of `NEGATIVE_TTL_SECONDS`, and only
    a definitive miss is ever put there, never a timeout or an outage.
    """
    hit, cached = _cache.get(asked)
    if not hit:
        return None
    if cached is None:
        logger.info(f"'{asked}' unresolved (remembered from a recent attempt)")
        raise ObjectNameError(f"'{asked}' could not be resolved (remembered from a recent attempt)")
    logger.debug(f"resolved '{asked}' from cache: {_describe(cached)}")
    return cached


def _await_tns(
    arm: _Arm | None, deadline: float, total_timeout: float, attempts: list[str]
) -> ResolvedObject | None:
    """TNS's answer if it has one within the budget, else None meaning "use Sesame"."""
    if arm is None:
        return None
    if not arm.wait(max(0.0, deadline - time.monotonic())):
        attempts.append(f"tns: still running at the {total_timeout:g}s deadline")
        return None
    return _collect_tns(arm, attempts)


def resolve_object_name(name: str, total_timeout: float = TOTAL_TIMEOUT_SECONDS) -> ResolvedObject:
    """Coordinates for `name`, in J2000, with a record of which service supplied them.

    Raises `MovingTargetError` for a designation with no fixed position, and
    `ObjectNameError` when nothing could resolve it within the budget. Both are the
    caller's cue to report and move on -- a plan sweeper leaves the target for its next
    pass, an endpoint answers with an error.
    """
    if not name or not name.strip():
        raise ObjectNameError("no object name given")

    asked = name.strip()
    started = time.monotonic()
    deadline = started + total_timeout

    if _is_moving_target(asked):
        logger.info(f"refused '{asked}': moving target, no fixed J2000 position")
        raise MovingTargetError(
            f"'{asked}' is a moving target; it has no fixed J2000 position, only an "
            "ephemeris at an instant, so it cannot be resolved to coordinates here"
        )

    remembered = _remembered(asked)
    if remembered is not None:
        return remembered

    attempts: list[str] = []

    # Everything that needs the network starts NOW, together. The gates above are what
    # decide whether to ask at all, so nothing is asked speculatively -- but once the
    # decision is made there is no reason for one service to queue behind another.
    sesame_arm = _spawn(_resolve_via_sesame, asked, min(SESAME_TIMEOUT_SECONDS, total_timeout))
    tns_arm = _spawn(_try_tns_arm, asked, deadline) if is_tns_name(asked) else None

    # TNS is preferred, not merely first past the post, so its verdict is awaited before
    # Sesame's is looked at. Racing them and taking whoever answers soonest would be
    # WRONG: Sesame is the faster of the two, so first-wins would systematically prefer
    # the staler source for exactly the names TNS exists to answer -- a transient reported
    # last night is in TNS and not yet in SIMBAD.
    #
    # Running them together still pays. On a TNS miss or outage Sesame's answer is already
    # in hand instead of being started from cold, and the worst case becomes the slower of
    # the two rather than their sum.
    resolved = _await_tns(tns_arm, deadline, total_timeout, attempts)
    if resolved is not None:
        _cache.put(asked, resolved, TNS_TTL_SECONDS)
        logger.info(
            f"resolved '{asked}': {_describe(resolved)} in {time.monotonic() - started:.2f}s"
        )
        return resolved

    if not sesame_arm.wait(max(0.0, deadline - time.monotonic())):
        # Not cached: running out of budget says nothing about whether the object exists.
        attempts.append(f"sesame: still running at the {total_timeout:g}s deadline")
        logger.warning(
            f"'{asked}' not resolved within {total_timeout:g}s -- tried {'; '.join(attempts)}"
        )
        raise ObjectNameError(
            f"'{asked}' not resolved within {total_timeout:g}s ({'; '.join(attempts)})"
        )

    resolved, definitive = _collect_sesame(asked, sesame_arm, attempts)
    if resolved is not None:
        _cache.put(asked, resolved, CATALOGUE_TTL_SECONDS)
        logger.info(
            f"resolved '{asked}': {_describe(resolved)} in {time.monotonic() - started:.2f}s"
        )
        return resolved

    if definitive:
        _cache.put(asked, None, NEGATIVE_TTL_SECONDS)
    logger.warning(
        f"'{asked}' could not be resolved in {time.monotonic() - started:.2f}s "
        f"({'definitive' if definitive else 'not conclusive, will retry'}) -- tried {'; '.join(attempts)}"
    )
    raise ObjectNameError(f"'{asked}' could not be resolved ({'; '.join(attempts)})")


def _resolve_via_tns(
    name: str, credentials: tuple[str, str, str], timeout: float
) -> ResolvedObject | None:
    """Ask TNS for a transient's own position. None if TNS has no such object.

    Verified against the live service on 2026-08-23, once bot credentials existed:
    `SN 2023ixf` came back at 14.06071h +54.31165, which is the published position to
    0.6" in RA and 0.2" in Dec, and `AT 2024gy` came back as `SN2024gy` -- TNS returning
    the promoted prefix for a transient since classified, which is the reason TNS is
    asked before Sesame for these names.

    The marker is the part to get right, and it fails in a way that points at the wrong
    thing. TNS enforces it only when it parses as a bot identity: omit it, or malform it,
    and the request succeeds. A well-formed marker naming a bot the key does not belong
    to gives `401 Unauthorized` with no detail -- the same response as a garbage key. So
    a 401 here means the `tns_id`/`bot_name` in the vault disagree with the key at least
    as often as it means the key is bad; check those before suspecting the key. Do not
    "fix" a 401 by dropping the marker: TNS requires bots to identify themselves, and an
    unmarked client is the one that gets rate-limited.
    """
    import httpx

    api_key, tns_id, bot_name = credentials
    marker = f'tns_marker{{"tns_id": {tns_id}, "type": "bot", "name": "{bot_name}"}}'
    payload = {"api_key": api_key, "data": f'{{"objname": "{_tns_objname(name)}"}}'}

    response = httpx.post(
        TNS_OBJECT_URL,
        data=payload,
        headers={"User-Agent": marker},
        timeout=timeout,
        follow_redirects=True,
        proxy=internet_proxy(TNS_OBJECT_URL),
        trust_env=False,
    )
    response.raise_for_status()
    body = response.json()

    data = (body or {}).get("data") or {}
    if not data or data.get("radeg") is None or data.get("decdeg") is None:
        return None

    return ResolvedObject(
        name=name,
        ra_j2000_hours=float(data["radeg"]) / 15.0,
        dec_j2000_degs=float(data["decdeg"]),
        resolver="tns",
        canonical_name=(f"{data.get('name_prefix', '')}{data.get('objname', '')}".strip() or None),
    )


def _tns_objname(name: str) -> str:
    """TNS wants the designation without its prefix or spaces: `SN 2023ixf` -> `2023ixf`."""
    match = _TNS_NAME.match(name)
    return f"{match.group(2)}{match.group(3)}" if match else name.strip()
