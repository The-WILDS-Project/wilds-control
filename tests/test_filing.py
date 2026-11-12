import logging
import os
import threading
import time

import pytest
from fastapi.testclient import TestClient

from wilds.control import config, filing
from wilds.control.app import create_app
from wilds.control.filing import Filing, Probe, StagingFull


@pytest.fixture
def dirs(tmp_path):
    return tmp_path / "staging", tmp_path / "share"


def make(dirs, **kw) -> Filing:
    staging, share = dirs
    staging.mkdir(exist_ok=True)
    cfg = config.Filing(staging_root=staging, share_root=share, sweep_interval_s=0.05, **kw)
    return Filing(cfg)


def put(f: Filing, name: str, data: bytes = b"data") -> None:
    with f.writing(name) as path:
        path.write_bytes(data)


# -- writing --------------------------------------------------------------------------


def test_a_file_appears_in_staging_whole_and_only_when_finished(dirs):
    f = make(dirs)
    with f.writing("dct/20261107/vis.0001.fits") as path:
        path.write_bytes(b"half")
        assert f.waiting() == []  # the sweeper cannot see a file still being written
        assert path.name.startswith(".") and path.name.endswith(".partial")
    assert [p.name for p in f.waiting()] == ["vis.0001.fits"]
    assert f.status().pending == 1 and f.status().pending_bytes == 4


def test_a_failed_write_leaves_nothing(dirs):
    f = make(dirs)
    with pytest.raises(RuntimeError), f.writing("a/b.fits") as path:
        path.write_bytes(b"x")
        raise RuntimeError("camera dropped")
    assert f.waiting() == [] and list(dirs[0].rglob(".*partial")) == []
    assert f.status().pending == 0


@pytest.mark.parametrize("bad", ["../outside.fits", "/etc/passwd", "", "."])
def test_a_name_outside_staging_is_refused(dirs, bad):
    f = make(dirs)
    with pytest.raises(ValueError), f.writing(bad):
        pass


def test_the_staging_disk_at_its_ceiling_refuses_new_files(dirs, monkeypatch):
    f = make(dirs, disk_ceiling=0.5)
    monkeypatch.setattr(
        filing.shutil, "disk_usage", lambda p: type("U", (), {"used": 60, "total": 100})()
    )
    with pytest.raises(StagingFull), f.writing("a.fits"):
        pass
    assert f.status().staging_full is True
    assert "staging disk is 60% full" in " ".join(f.status().warnings())


# -- delivery -------------------------------------------------------------------------


def test_delivery_moves_the_file_to_the_share_and_tidies_staging(dirs):
    f = make(dirs)
    dirs[1].mkdir()
    put(f, "dct/20261107/vis.0001.fits", b"frame")
    assert f.sweep() == 1
    assert (dirs[1] / "dct/20261107/vis.0001.fits").read_bytes() == b"frame"
    assert f.waiting() == [] and not (dirs[0] / "dct").exists()  # empty folders go too
    st = f.status()
    assert st.share == "online" and st.pending == 0 and st.delivered == 1 and st.last_delivery_ms


def test_the_share_down_keeps_files_in_staging_and_they_follow_when_it_returns(dirs, caplog):
    f = make(dirs)
    put(f, "dct/n/a.fits")
    put(f, "dct/n/b.fits")
    with caplog.at_level(logging.INFO, logger="wilds.control.filing"):
        assert f.sweep() == 0 and f.sweep() == 0
        st = f.status()
        assert st.share == "offline" and st.pending == 2
        assert "not reachable" in " ".join(st.warnings()) and "2 file(s)" in " ".join(st.warnings())
        dirs[1].mkdir()  # the share comes back
        assert f.sweep() == 2
    assert sorted(p.name for p in (dirs[1] / "dct/n").iterdir()) == ["a.fits", "b.fits"]
    assert f.status().share == "online" and f.status().warnings() == []
    said = [r.message for r in caplog.records]
    assert sum("not reachable" in m for m in said) == 1  # once per change, not per sweep
    assert sum("is reachable" in m for m in said) == 1


def test_the_oldest_goes_first(dirs):
    f = make(dirs)
    dirs[1].mkdir()
    put(f, "late.fits")
    put(f, "early.fits")
    os.utime(dirs[0] / "early.fits", (1000, 1000))
    order = []
    real = f._deliver
    f._deliver = lambda p: order.append(p.name) or real(p)
    f.sweep()
    assert order == ["early.fits", "late.fits"]


def test_the_share_root_is_never_created(dirs):
    f = make(dirs)
    put(f, "a.fits")
    f.sweep()
    assert not dirs[1].exists()


def test_an_unmounted_mount_point_is_not_a_share(dirs, monkeypatch):
    f = make(dirs, share_is_mount=True)
    dirs[1].mkdir()  # an empty local folder where the mount should be
    put(f, "a.fits")
    assert f.sweep() == 0 and f.status().share == "offline"
    monkeypatch.setattr(os.path, "ismount", lambda p: True)
    f._probe._cached = None
    assert f.sweep() == 1


def test_an_interrupted_copy_never_shows_up_under_the_final_name(dirs, monkeypatch):
    f = make(dirs)
    dirs[1].mkdir()
    put(f, "a.fits", b"x" * 1000)

    def dies_halfway(src, dst):
        with open(dst, "wb") as out:
            out.write(b"x" * 10)
        raise OSError(5, "share dropped")

    monkeypatch.setattr(filing.shutil, "copyfile", dies_halfway)
    assert f.sweep() == 0
    assert not (dirs[1] / "a.fits").exists() and list(dirs[1].glob(".*")) == []  # no junk either
    assert [p.name for p in f.waiting()] == ["a.fits"]  # the source is untouched
    assert "a.fits" in (f.status().last_error or "")
    monkeypatch.undo()
    assert f.sweep() == 1 and (dirs[1] / "a.fits").read_bytes() == b"x" * 1000


def test_a_file_already_on_the_share_with_the_same_bytes_counts_as_delivered(dirs):
    f = make(dirs)
    (dirs[1] / "d").mkdir(parents=True)
    (dirs[1] / "d/a.fits").write_bytes(b"same")
    put(f, "d/a.fits", b"same")
    assert f.sweep() == 1 and f.waiting() == []


def test_a_different_file_with_the_same_name_is_never_overwritten_and_reported_once(dirs, caplog):
    f = make(dirs)
    (dirs[1] / "d").mkdir(parents=True)
    (dirs[1] / "d/a.fits").write_bytes(b"theirs")
    put(f, "d/a.fits", b"ours")
    with caplog.at_level(logging.ERROR, logger="wilds.control.filing"):
        assert f.sweep() == 0 and f.sweep() == 0 and f.sweep() == 0
    assert (dirs[1] / "d/a.fits").read_bytes() == b"theirs"
    assert [p.name for p in f.waiting()] == ["a.fits"]
    assert len([r for r in caplog.records if "not overwriting" in r.message]) == 1
    assert "differs" in f.status().last_error and "Delivery problem" in " ".join(
        f.status().warnings()
    )


# -- the probe ------------------------------------------------------------------------


def test_a_hung_share_costs_one_probe_timeout_and_one_thread():
    release = threading.Event()
    calls = []

    def hangs():
        calls.append(1)
        release.wait(5)
        return True

    probe = Probe(hangs)
    start = time.monotonic()
    assert probe.ok(0.1) is False and time.monotonic() - start < 1
    for _ in range(5):
        assert probe.ok(0.1, fresh=True) is False  # no new thread while one is stuck
    assert len(calls) == 1
    release.set()
    time.sleep(0.2)
    assert probe.ok(0.5, fresh=True) is True and len(calls) == 2


def test_the_probe_answer_is_cached_briefly():
    calls = []
    probe = Probe(lambda: calls.append(1) or True)
    assert probe.ok(1) and probe.ok(1) and probe.ok(1)
    assert len(calls) == 1
    assert probe.ok(1, fresh=True) and len(calls) == 2


# -- the sweeper thread ---------------------------------------------------------------


def test_the_sweeper_delivers_on_its_own_and_after_a_restart(dirs):
    f = make(dirs)
    put(f, "d/a.fits")  # written before the share and the thread exist
    f.start()
    try:
        time.sleep(0.2)
        assert f.status().pending == 1 and f.status().share == "offline"
        dirs[1].mkdir()
        assert f.flush(5)
        assert (dirs[1] / "d/a.fits").exists()
        put(f, "d/b.fits")  # a new file wakes it
        assert f.flush(5) and (dirs[1] / "d/b.fits").exists()
    finally:
        f.stop()

    put(f, "d/c.fits")  # a new process finds what an earlier one left
    again = make(dirs)
    again.start()
    try:
        assert again.flush(5) and (dirs[1] / "d/c.fits").exists()
    finally:
        again.stop()


def test_starting_drops_the_partials_a_crash_left_and_counts_what_waits(dirs):
    f = make(dirs)
    put(f, "d/a.fits", b"whole")
    (dirs[0] / "d/.b.fits.partial").write_bytes(b"torn")
    again = make(dirs)
    again.start()
    try:
        assert not (dirs[0] / "d/.b.fits.partial").exists()
        assert again.status().pending == 1
    finally:
        again.stop()


# -- config and status ----------------------------------------------------------------


def test_defaults_and_home_expansion(config_file):
    assert config.get().filing.share_root.as_posix() == "/lois/data/dct"
    assert config.get().filing.share_is_mount is False
    config_file('[filing]\nstaging_root = "~/s"\nshare_root = "~/share"\n')
    assert config.get().filing.staging_root == config.Path.home() / "s"


@pytest.mark.parametrize(
    "bad", ['staging_root = "relative/dir"', "sweep_interval_s = 0", "disk_ceiling = 1.5"]
)
def test_a_bad_filing_setting_rejects_the_file(config_file, bad, caplog):
    config_file(f'opmode = "ldt-production"\n[filing]\n{bad}\n')
    assert config.get().opmode == "development"  # rejected whole, defaults kept


def test_status_carries_the_filing_and_its_warning_when_the_share_is_down(dirs):
    f = filing.get_filing()
    put(f, "a.fits", b"x" * 2_000_000)
    f.sweep()
    body = TestClient(create_app()).get("/status").json()
    assert body["filing"]["share"] == "offline" and body["filing"]["pending"] == 1
    assert any("not reachable" in w and "2 MB" in w for w in body["warnings"])


def test_the_app_starts_and_stops_the_sweeper(dirs):
    with TestClient(create_app()):
        f = filing.get_filing()
        assert f._thread is not None and f._thread.is_alive()
        put(f, "a.fits")
        dirs[1].mkdir()
        assert f.flush(5)
    assert filing._filing is None  # closed at shutdown
