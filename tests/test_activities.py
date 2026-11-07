import threading
import time
from enum import Enum

import pytest

from wilds.control.activities import Activities, ActivityEvent, Registry


class Move(Enum):
    Slewing = 1
    Parking = 2
    Cancelling = 3


class Other(Enum):
    Slewing = 1


def make() -> Activities:
    return Activities("mount", Move)


def test_start_and_end_track_the_running_set_in_start_order():
    a = make()
    assert a.is_idle() and a.names == []
    a.start(Move.Parking)
    a.start(Move.Slewing)
    assert a.names == ["Parking", "Slewing"]
    assert a.is_active(Move.Slewing) and not a.is_active(Move.Cancelling)
    a.end(Move.Parking)
    assert a.names == ["Slewing"] and not a.is_idle()
    a.end(Move.Slewing)
    assert a.is_idle()


def test_ending_an_activity_that_is_not_running_does_nothing():
    a = make()
    a.end(Move.Slewing)
    assert a.is_idle()


def test_starting_a_running_activity_is_an_error_unless_existing_ok():
    a = make()
    a.start(Move.Slewing, details=["first"])
    with pytest.raises(RuntimeError):
        a.start(Move.Slewing)
    a.start(Move.Slewing, details=["second"], existing_ok=True)
    assert a.snapshot()[0].details == ("first",)  # the original start stands


def test_a_member_of_another_enum_is_refused_even_with_the_same_name():
    with pytest.raises(TypeError):
        make().start(Other.Slewing)


def test_snapshot_reports_wall_clock_start_and_details():
    a = make()
    before = time.time_ns() // 1_000_000
    a.start(Move.Slewing, details=["to M31"])
    (info,) = a.snapshot()
    assert info.name == "Slewing" and info.details == ("to M31",)
    assert before <= info.since_ms <= time.time_ns() // 1_000_000


def test_listeners_see_start_and_end_with_a_duration():
    a, events = make(), []
    a.on_change(events.append)
    a.start(Move.Slewing, details=["x"])
    time.sleep(0.02)
    a.end(Move.Slewing)
    start, end = events
    assert start == ActivityEvent("mount", "Slewing", "start", ("x",))
    assert end.kind == "end" and end.details == ("x",) and end.duration_s >= 0.02


def test_a_failing_listener_cannot_fail_the_operation_it_announces():
    a, seen = make(), []

    def boom(event):
        raise RuntimeError("notifier down")

    a.on_change(boom)
    a.on_change(seen.append)
    a.start(Move.Slewing)
    assert a.is_active(Move.Slewing)
    a.end(Move.Slewing)
    assert [e.kind for e in seen] == ["start", "end"]  # later listeners still run


def test_wait_clear_returns_when_the_activity_ends():
    a = make()
    a.start(Move.Cancelling)
    threading.Timer(0.05, a.end, [Move.Cancelling]).start()
    assert a.wait_clear(Move.Cancelling, timeout=2, interval=0.01) is True


def test_wait_clear_times_out_and_leaves_the_activity_set():
    a = make()
    a.start(Move.Cancelling)
    assert a.wait_clear(Move.Cancelling, timeout=0.05, interval=0.01) is False
    assert a.is_active(Move.Cancelling)


async def test_await_clear_both_ways():
    a = make()
    a.start(Move.Cancelling)
    assert await a.await_clear(Move.Cancelling, timeout=0.05, interval=0.01) is False
    threading.Timer(0.05, a.end, [Move.Cancelling]).start()
    assert await a.await_clear(Move.Cancelling, timeout=2, interval=0.01) is True


def test_registry_lists_only_components_with_something_running():
    r = Registry()
    mount = r.register("mount", Move)
    r.register("stage", Other)
    assert r.snapshot() == {}
    mount.start(Move.Slewing)
    assert list(r.snapshot()) == ["mount"]
    with pytest.raises(ValueError):
        r.register("mount", Move)
    assert r.get("mount") is mount
