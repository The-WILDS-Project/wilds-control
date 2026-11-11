"""Resolving an object name to J2000.

No network: Sesame and TNS are both replaced. What is pinned is the order of enquiry, the
refusals, the provenance, and the budget -- the things a wrong answer would be silent
about.

The case worth stating plainly: a misresolution is indistinguishable from success at every
layer below this. The mount slews normally, guiding locks, and a spectrum is taken of the
wrong object. So "which service answered" is not decoration, and neither is refusing a
name that cannot have a fixed position.
"""

from __future__ import annotations

import threading
import time

import pytest

from wilds.control import object_resolver as res
from wilds.control.object_resolver import (
    MovingTargetError,
    ObjectNameError,
    ResolvedObject,
    SesameMissError,
    is_tns_name,
    resolve_object_name,
)


@pytest.fixture(autouse=True)
def _clear_cache():
    res._cache.clear()
    yield
    res._cache.clear()


@pytest.fixture
def sesame(monkeypatch):
    """Replace Sesame. Records what it was asked; answers unless told to fail."""

    calls: list[str] = []

    class Fake:
        answer: tuple[float, float] | None = (14.06, 54.31)  # ra hours, dec degrees
        fail_with: Exception | None = None

        def __call__(self, name, timeout):
            calls.append(name)
            if self.fail_with is not None:
                raise self.fail_with
            ra, dec = self.answer
            return ResolvedObject(
                name=name, ra_j2000_hours=ra, dec_j2000_degs=dec, resolver="sesame"
            )

    fake = Fake()
    fake.calls = calls
    monkeypatch.setattr(res, "_resolve_via_sesame", fake)
    return fake


@pytest.fixture
def tns(monkeypatch):
    """Replace TNS and give it credentials, so the TNS arm is reachable."""

    calls: list[str] = []

    class Fake:
        answer: ResolvedObject | None = None
        fail_with: Exception | None = None

        def __call__(self, name, credentials, timeout):
            calls.append(name)
            if self.fail_with is not None:
                raise self.fail_with
            return self.answer

    fake = Fake()
    fake.calls = calls
    monkeypatch.setattr(res, "_resolve_via_tns", fake)
    monkeypatch.setattr(res, "_tns_credentials", lambda: ("key", "1234", "WILDS_bot"))
    return fake


class TestMovingTargetsAreRefused:
    """No fixed J2000 exists for these, so a plausible answer is worse than none."""

    @pytest.mark.parametrize(
        "name",
        [
            "C/2023 A3",
            "1P/Halley",
            "73P-C",
            "2024 AB",
            "2024 AB1",
            "(433)",
            "(433) Eros",
            "Mars",
            "ceres",
            "Io",
        ],
        ids=lambda n: n,
    )
    def test_refused_by_name(self, name, sesame):
        with pytest.raises(MovingTargetError):
            resolve_object_name(name)
        assert sesame.calls == [], "a moving target must not reach a catalogue at all"

    @pytest.mark.parametrize(
        "name", ["M31", "NGC 224", "SN2023ixf", "Gaia DR3 12345", "3C 273"], ids=lambda n: n
    )
    def test_real_objects_are_not_mistaken_for_moving_ones(self, name, sesame, tns):
        """A false positive here refuses a real target, which is worse than falling
        through to a catalogue that simply will not find something."""
        resolve_object_name(name)


class TestOrderOfEnquiry:
    def test_tns_wins_a_tns_name_even_though_sesame_was_asked_too(self, sesame, tns):
        """Both are asked at once, and TNS is PREFERRED rather than merely first.

        This inverts what the test here used to assert -- "Sesame must not be asked once
        TNS has answered" -- which was true of the sequential arrangement and is the point
        of the parallel one. Preference, not a race: Sesame is the faster of the two, so
        taking whoever answers soonest would systematically pick the staler source for
        exactly the names TNS exists to answer.
        """
        tns.answer = ResolvedObject(
            "SN2023ixf", 14.06, 54.31, resolver="tns", canonical_name="SN 2023ixf"
        )

        result = resolve_object_name("SN2023ixf")

        assert tns.calls == ["SN2023ixf"]
        assert sesame.calls == ["SN2023ixf"], (
            "Sesame is asked concurrently, so a TNS miss costs no extra round trip"
        )
        assert result.resolver == "tns", "but TNS's answer is the one that counts"

    def test_a_tns_miss_falls_through_to_sesame(self, sesame, tns):
        """A name that merely looks like a transient may be an older object Sesame knows.
        Falling through is safe: Sesame's failure for a fresh transient is 'not found'."""
        tns.answer = None

        result = resolve_object_name("SN1987A")

        assert tns.calls and sesame.calls == ["SN1987A"]
        assert result.resolver == "sesame"

    def test_a_tns_outage_falls_through_rather_than_failing(self, sesame, tns):
        tns.fail_with = RuntimeError("TNS unreachable")

        assert resolve_object_name("AT2024abc").resolver == "sesame"

    def test_missing_tns_credentials_fall_through(self, sesame, monkeypatch):
        """Without a key the resolver is still useful; it is not a reason to refuse."""
        monkeypatch.setattr(res, "_tns_credentials", lambda: None)

        assert resolve_object_name("SN2023ixf").resolver == "sesame"

    def test_a_non_tns_name_never_asks_tns(self, sesame, tns):
        resolve_object_name("M31")
        assert tns.calls == []

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("SN2023ixf", True),
            ("AT2024abc", True),
            ("SN 1987A", True),
            ("M31", False),
            ("NGC 224", False),
        ],
        ids=lambda v: str(v),
    )
    def test_tns_shape_detection(self, name, expected):
        assert is_tns_name(name) is expected


class TestProvenance:
    def test_the_result_says_who_answered(self, sesame, tns):
        tns.answer = ResolvedObject(
            "SN2023ixf", 14.06, 54.31, resolver="tns", canonical_name="SN 2023ixf"
        )
        assert resolve_object_name("SN2023ixf").resolver == "tns"
        res._cache.clear()
        tns.answer = None
        assert resolve_object_name("SN2023ixf").resolver == "sesame"

    def test_coordinates_are_j2000_hours_and_degrees(self, sesame):
        result = resolve_object_name("M31")
        assert result.ra_j2000_hours == 14.06 and result.dec_j2000_degs == 54.31

    def test_the_result_is_serialisable(self, sesame):
        d = resolve_object_name("M31").as_dict()
        assert set(d) >= {"name", "ra_j2000_hours", "dec_j2000_degs", "resolver", "canonical_name"}


class TestFailure:
    def test_an_unresolvable_name_raises(self, sesame):
        sesame.fail_with = SesameMissError("sesame: no database has it (2 mirror(s) agreed)")
        with pytest.raises(ObjectNameError, match="could not be resolved"):
            resolve_object_name("Zaphod Beeblebrox")

    def test_the_error_names_what_was_tried(self, sesame, tns):
        tns.answer = None
        sesame.fail_with = SesameMissError("sesame: no database has it (2 mirror(s) agreed)")
        with pytest.raises(ObjectNameError) as excinfo:
            resolve_object_name("SN2099zzz")
        assert "tns" in str(excinfo.value) and "sesame" in str(excinfo.value)

    def test_an_unexpected_error_does_not_escape(self, sesame):
        """A resolver must not surprise its caller with an arbitrary exception; a plan
        sweeper handling ObjectNameError should not also have to handle httpx."""
        sesame.fail_with = ValueError("something odd")
        with pytest.raises(ObjectNameError):
            resolve_object_name("M31")

    @pytest.mark.parametrize("name", ["", "   ", None], ids=["empty", "blank", "none"])
    def test_no_name_is_an_error_not_a_lookup(self, name, sesame):
        with pytest.raises(ObjectNameError):
            resolve_object_name(name)
        assert sesame.calls == []


class TestCache:
    def test_a_second_ask_does_not_reach_the_service(self, sesame):
        resolve_object_name("M31")
        resolve_object_name("M31")
        assert sesame.calls == ["M31"], "the second answer must come from the cache"

    def test_lookup_is_insensitive_to_case_and_spacing(self, sesame):
        resolve_object_name("M31")
        resolve_object_name("  m31 ")
        assert sesame.calls == ["M31"]

    def test_a_definitive_miss_is_remembered_briefly(self, sesame):
        """So a sweeper does not re-ask CDS on every pass for a name that will not
        resolve -- but only briefly, or a transient reported minutes from now stays
        unresolvable for as long as the miss is held."""
        sesame.fail_with = SesameMissError(
            "sesame: no database has 'Nothing Here' (2 mirror(s) agreed)"
        )
        with pytest.raises(ObjectNameError):
            resolve_object_name("Nothing Here")
        with pytest.raises(ObjectNameError, match="remembered"):
            resolve_object_name("Nothing Here")
        assert len(sesame.calls) == 1

    def test_a_service_outage_is_not_remembered(self, sesame):
        """The bug this exists to prevent, seen live on 2026-08-12: CDS refused a query
        after several in quick succession, `NGC 224` -- which is M31 -- came back
        unresolved, and the miss was cached. It resolved in 0.83s on the next attempt.

        astropy reports both cases as NameResolveError; only the message distinguishes
        them. A service that did not answer says nothing about whether the object exists.
        """
        sesame.fail_with = ObjectNameError(
            "sesame: no mirror answered (cds: ConnectTimeout; cfa: ReadTimeout)"
        )
        with pytest.raises(ObjectNameError):
            resolve_object_name("NGC 224")

        sesame.fail_with = None
        assert resolve_object_name("NGC 224").resolver == "sesame", (
            "must be retried, not remembered"
        )
        assert len(sesame.calls) == 2

    def test_an_unexpected_error_is_not_remembered_either(self, sesame):
        sesame.fail_with = ValueError("something odd")
        with pytest.raises(ObjectNameError):
            resolve_object_name("M31")
        sesame.fail_with = None
        assert resolve_object_name("M31").resolver == "sesame"
        assert len(sesame.calls) == 2

    def test_a_negative_entry_expires_sooner_than_a_positive_one(self):
        assert res.NEGATIVE_TTL_SECONDS < res.TNS_TTL_SECONDS < res.CATALOGUE_TTL_SECONDS

    def test_an_expired_entry_is_asked_again(self, sesame, monkeypatch):
        resolve_object_name("M31")
        monkeypatch.setattr(
            time, "monotonic", lambda: time.perf_counter() + res.CATALOGUE_TTL_SECONDS + 1
        )
        resolve_object_name("M31")
        assert len(sesame.calls) == 2


class TestBudget:
    def test_the_budget_bounds_the_slowest_arm_not_their_sum(self):
        """Every service is asked at once, so the enquiry costs the slowest of them.

        This replaces `SESAME * 2 + TNS <= TOTAL`, which was the right sum when the two
        mirrors were tried in turn behind TNS. Under that arithmetic the Sesame cap had to
        be 4s to fit -- and 4s is below the CfA mirror's FASTEST observed response (9.27s),
        so the fallback could never once have answered. Racing them lets the cap be the
        loser's ceiling instead of a per-enquiry cost.
        """
        assert res.SESAME_TIMEOUT_SECONDS > res.TOTAL_TIMEOUT_SECONDS, (
            "the per-mirror cap is deliberately larger than the whole-enquiry deadline: "
            "it bounds a loser that nobody is waiting on, while TOTAL bounds the caller"
        )
        assert res.TNS_TIMEOUT_SECONDS < res.TOTAL_TIMEOUT_SECONDS

    def test_a_slow_tns_no_longer_starves_sesame(self, sesame, tns, monkeypatch):
        """The reason for asking them together.

        Sequentially, TNS spending the budget left nothing for Sesame and the enquiry
        failed having never asked it -- which is what this test used to pin. Now Sesame was
        started at the same instant, so a TNS that never returns costs its answer nothing.
        """
        started = threading.Event()

        def slow_tns(name, credentials, timeout):
            started.set()
            time.sleep(5)  # still going long after the deadline below

        monkeypatch.setattr(res, "_resolve_via_tns", slow_tns)

        result = resolve_object_name("SN2024zzz", total_timeout=0.4)

        assert started.is_set(), "TNS was asked"
        assert sesame.calls == ["SN2024zzz"], "and so was Sesame, at the same time"
        assert result.resolver == "sesame", "so its answer was there when TNS ran out of budget"

    def test_nothing_is_asked_once_the_deadline_has_passed(self, sesame, tns, monkeypatch):
        """A deadline still means something: if no arm has answered by then, the enquiry
        fails rather than waiting on. Running out of budget says nothing about whether the
        object exists, so it is never remembered."""

        def slow_sesame(name, timeout):
            time.sleep(5)

        monkeypatch.setattr(res, "_resolve_via_sesame", slow_sesame)

        with pytest.raises(ObjectNameError, match="not resolved within"):
            resolve_object_name("M31", total_timeout=0.3)

        hit, _ = res._cache.get("M31")
        assert not hit, "a timeout must not be cached as a miss"


class TestLogging:
    """The returned provenance only helps if something captured it. For a plan sweeper
    working through targets overnight, the log is the record of what was asked, who
    answered, how long it took and what failed -- and it is what answers "why did we
    point there" long after the fact.
    """

    def test_a_resolution_logs_provenance_and_timing(self, sesame, caplog):
        with caplog.at_level("INFO", logger="wilds.control.object_resolver"):
            resolve_object_name("M31")

        line = "\n".join(r.message for r in caplog.records)
        assert "M31" in line
        assert "via sesame" in line, (
            "which service answered must be in the log, not only the return value"
        )
        assert "ra=" in line and "dec=" in line, "the coordinates it resolved to"
        assert "s" in line and "in " in line, "and how long it took"

    def test_the_service_that_answered_is_named(self, sesame, tns, caplog):
        tns.answer = ResolvedObject(
            "SN2023ixf", 14.06, 54.31, resolver="tns", canonical_name="SN 2023ixf"
        )
        with caplog.at_level("INFO", logger="wilds.control.object_resolver"):
            resolve_object_name("SN2023ixf")

        line = "\n".join(r.message for r in caplog.records)
        assert "via tns" in line
        assert "SN 2023ixf" in line, (
            "the canonical name it matched -- an alias match is the likeliest misresolution"
        )

    def test_a_failure_is_logged_with_what_was_tried(self, sesame, caplog):
        sesame.fail_with = SesameMissError("sesame: no database has 'Nope' (2 mirror(s) agreed)")
        with (
            caplog.at_level("WARNING", logger="wilds.control.object_resolver"),
            pytest.raises(ObjectNameError),
        ):
            resolve_object_name("Nope")

        warnings = [r.message for r in caplog.records if r.levelname == "WARNING"]
        assert warnings, "a failed resolution must leave a trace, not only raise"
        assert "sesame" in warnings[0], "and say which services were asked"

    def test_a_failure_says_whether_it_will_be_retried(self, sesame, caplog):
        """The distinction that matters to whoever reads the log at 03:00: a definitive
        miss is a bad name, a non-conclusive one is a service that did not answer."""
        sesame.fail_with = ObjectNameError(
            "sesame: no mirror answered (cds: ConnectTimeout; cfa: ReadTimeout)"
        )
        with (
            caplog.at_level("WARNING", logger="wilds.control.object_resolver"),
            pytest.raises(ObjectNameError),
        ):
            resolve_object_name("NGC 224")

        assert "will retry" in "\n".join(r.message for r in caplog.records)

    def test_a_refusal_is_logged(self, sesame, caplog):
        with (
            caplog.at_level("INFO", logger="wilds.control.object_resolver"),
            pytest.raises(MovingTargetError),
        ):
            resolve_object_name("C/2023 A3")

        assert "moving target" in "\n".join(r.message for r in caplog.records)

    def test_a_cache_hit_does_not_repeat_the_info_line(self, sesame, caplog):
        """Otherwise a sweeper's log implies a network call that never happened."""
        resolve_object_name("M31")
        with caplog.at_level("INFO", logger="wilds.control.object_resolver"):
            resolve_object_name("M31")

        assert [r for r in caplog.records if r.levelname == "INFO"] == []


class TestTheEndpoint:
    """``GET /resolve`` (routers/resolve.py): the answer, and the refusals as the error body."""

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient

        from wilds.control.app import create_app

        with TestClient(create_app()) as c:
            yield c

    def test_it_answers_with_who_said_so(self, client, sesame):
        body = client.get("/resolve", params={"name": "M31"}).json()
        assert body["ra_j2000_hours"] == 14.06 and body["dec_j2000_degs"] == 54.31
        assert body["resolver"] == "sesame" and body["name"] == "M31"

    def test_a_moving_target_is_422(self, client, sesame):
        r = client.get("/resolve", params={"name": "Mars"})
        assert r.status_code == 422 and r.json()["code"] == "invalid"
        assert sesame.calls == []

    def test_an_unknown_name_is_404(self, client, sesame):
        sesame.fail_with = SesameMissError("sesame: no database has it (2 mirror(s) agreed)")
        r = client.get("/resolve", params={"name": "Zaphod"})
        assert r.status_code == 404 and r.json()["code"] == "not_found"

    def test_a_missing_name_is_422(self, client):
        assert client.get("/resolve").status_code == 422
