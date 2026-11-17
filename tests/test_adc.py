"""The ADC: the atmosphere's dispersion and the prism pairs that cancel it (checked against the
optics report), the component (moves, zero points, tracking) and its HTTP side."""

import math
import time

import pytest
from fastapi.testclient import TestClient

from wilds.control import lease, site, telescope
from wilds.control.activities import registry
from wilds.control.adc import (
    AdcBusy,
    AdcRange,
    SimulatedAdc,
    SkyState,
    model,
)
from wilds.control.app import create_app
from wilds.control.errors import ApiError
from wilds.control.jobs import StartedBy, manager
from wilds.control.weather import SITE_TYPICAL, Weather

ARIE = StartedBy(name="Arie", code="K7-42")
SEA_LEVEL = Weather(
    pressure_mbar=1013.0, temperature_c=0.0, humidity_pct=50.0
)  # the report's model


# -- the physics, against the optics report -----------------------------------------------


@pytest.mark.parametrize(("zd", "report_arcsec"), [(20, 1.31), (40, 2.94), (60, 5.98)])
def test_the_atmospheres_dispersion_matches_the_optics_report(zd, report_arcsec):
    # Appendix A.1: 162, 364 and 741 um between 305 and 915 nm at 124 um/arcsec, sea-level model
    assert model.atmospheric_dispersion_arcsec(zd, SEA_LEVEL) == pytest.approx(
        report_arcsec, rel=0.04
    )


def test_the_lowell_site_disperses_about_1_3_times_less_than_sea_level():
    ratio = model.atmospheric_dispersion_arcsec(
        60, SEA_LEVEL
    ) / model.atmospheric_dispersion_arcsec(60, SITE_TYPICAL)
    assert 1.25 < ratio < 1.4  # the report: "overstates the dispersion at the LDT by about 30%"


def test_no_dispersion_at_the_zenith_and_more_lower_down():
    d = [model.atmospheric_dispersion_arcsec(z, SITE_TYPICAL) for z in (0, 10, 30, 50, 60)]
    assert d[0] == pytest.approx(0.0, abs=1e-9) and d == sorted(d)


@pytest.mark.parametrize(
    ("dispersion", "report_phi"), [(1.3, 12.24), (2.9, 29.17), (6.0, 90.0)]
)  # ZD 20, 40 and 60 degrees; the report rounds the dispersions, so 1 degree will do
def test_the_pairs_sit_where_the_report_puts_them(dispersion, report_phi):
    p = model.pairs_for(dispersion, 0.0, 6.0)
    phi = (p.pair1_deg - p.pair2_deg) / 2 % 180
    assert phi == pytest.approx(report_phi, abs=1.0)


def test_pairs_make_the_dispersion_asked_for_in_the_direction_asked_for():
    for dispersion in (0.0, 0.4, 2.0, 5.9):
        for direction in (0.0, 33.0, 180.0, 271.5):
            p = model.pairs_for(dispersion, direction, 6.0)
            d, theta = model.delivered(p.pair1_deg, p.pair2_deg, 6.0)
            assert d == pytest.approx(dispersion, abs=1e-9)
            if dispersion > 0:
                assert (theta - direction + 180) % 360 - 180 == pytest.approx(0.0, abs=1e-6)


def test_more_than_the_adc_can_make_is_flagged_and_gives_its_most():
    p = model.pairs_for(8.0, 10.0, 6.0)
    assert p.saturated and p.dispersion_arcsec == pytest.approx(6.0)
    assert (p.pair1_deg - p.pair2_deg) % 360 == pytest.approx(180.0)  # phi = 90 degrees
    assert not model.pairs_for(6.0, 10.0, 6.0).saturated


# -- where the target is in the sky ---------------------------------------------------------


def test_at_transit_the_parallactic_angle_is_zero_and_zd_is_the_latitude_minus_dec():
    now = time.time()
    ra = site.local_sidereal_deg(now) / 15.0
    h = site.horizontal(ra, 10.0, now)
    assert h.zenith_distance_deg == pytest.approx(site.LATITUDE_DEG - 10.0, abs=1e-3)
    assert h.parallactic_angle_deg == pytest.approx(0.0, abs=1e-3)
    north = site.horizontal(ra, 60.0, now)  # north of the zenith: the pole is below the zenith
    assert abs(north.parallactic_angle_deg) == pytest.approx(180.0, abs=1e-3)


def test_the_parallactic_angle_is_positive_west_of_the_meridian_and_negative_east():
    now = time.time()
    lst = site.local_sidereal_deg(now)
    west = site.horizontal((lst - 45.0) / 15.0, 10.0, now)  # its hour angle is +45
    east = site.horizontal((lst + 45.0) / 15.0, 10.0, now)
    assert west.parallactic_angle_deg > 30 and east.parallactic_angle_deg < -30
    assert west.parallactic_angle_deg == pytest.approx(-east.parallactic_angle_deg, abs=1e-6)


def test_the_zenith_distance_agrees_with_astropy_to_the_precession():
    import astropy.units as u
    from astropy.coordinates import AltAz, EarthLocation, SkyCoord
    from astropy.time import Time

    loc = EarthLocation(
        lat=site.LATITUDE_DEG * u.deg, lon=site.LONGITUDE_DEG * u.deg, height=site.ALTITUDE_M * u.m
    )
    for hours in (0, 4, 9):
        t = time.time() + hours * 3600
        mine = site.horizontal(16.6949, 36.4613, t)
        ref = SkyCoord(ra=16.6949 * u.hourangle, dec=36.4613 * u.deg).transform_to(
            AltAz(obstime=Time(t, format="unix"), location=loc)
        )
        assert mine.zenith_distance_deg == pytest.approx(90 - ref.alt.deg, abs=0.5)
        if mine.zenith_distance_deg > 20:  # near the zenith the azimuth swings with the precession
            assert mine.azimuth_deg == pytest.approx(ref.az.deg, abs=0.5)


# -- the component -----------------------------------------------------------------------


def make(**kw) -> SimulatedAdc:
    sky = kw.pop("sky", lambda: None)
    adc = SimulatedAdc(
        speed_deg_s=kw.pop("speed_deg_s", 1e6), sky=sky, weather=lambda: SEA_LEVEL, **kw
    )
    return adc


async def started(**kw) -> SimulatedAdc:
    adc = make(**kw)
    adc.conf.patch({"settle_s": 0.0}, who=ARIE)
    await adc.startup()
    return adc


async def test_an_unstarted_adc_is_not_operational_and_will_not_move():
    adc = make()
    with pytest.raises(ApiError) as e:
        await adc.move_to(10, 20)
    assert e.value.status == 409 and e.value.body.code == "not_operational"


async def test_startup_references_it_and_reads_where_the_stages_are_without_turning_them():
    adc = make(start_deg=(10.0, 350.0))
    status = await adc.startup()
    assert status.operational and adc.status().referenced and adc.angles == (10.0, 350.0)
    assert not adc.tracking  # tracking stays off until it is asked for


async def test_moving_to_positions_takes_the_short_way_and_any_angle_is_taken_round_the_circle():
    adc = await started()
    assert await adc.move_to(100.0, 460.0) == (100.0, 100.0)
    assert adc.status().dispersion_arcsec == pytest.approx(0.0, abs=1e-9)  # the pairs agree
    await adc.move_to(30.0, 210.0)  # opposite: the most dispersion
    assert adc.status().dispersion_arcsec == pytest.approx(6.0)


async def test_a_dispersion_and_a_direction_move_both_pairs():
    adc = await started()
    await adc.move_dispersion(3.0, 40.0)
    s = adc.status()
    assert s.dispersion_arcsec == pytest.approx(3.0) and s.direction_deg == pytest.approx(40.0)
    with pytest.raises(AdcRange):
        await adc.move_dispersion(6.5, 0.0)
    with pytest.raises(AdcRange):
        await adc.move_dispersion(-1.0, 0.0)


async def test_the_zero_points_shift_what_the_stages_read_not_what_the_user_sees():
    adc = await started()
    adc.cal.patch({"zero1_deg": 10.0, "zero2_deg": -20.0}, who=ARIE)
    await adc.move_to(100.0, 100.0)
    assert adc.angles == (100.0, 100.0)  # the angles are of the pairs
    assert adc._at == (110.0, 80.0)  # the stages read their zero points further on
    s = adc.status()
    assert (s.stage1_deg, s.stage2_deg) == (110.0, 80.0) and (s.zero1_deg, s.zero2_deg) == (
        10.0,
        -20.0,
    )


async def test_a_changed_zero_point_changes_the_angles_at_once_without_a_move():
    adc = await started(start_deg=(110.0, 80.0))
    assert adc.angles == (110.0, 80.0)  # no zero points yet: the pairs read what the stages read
    adc.cal.patch({"zero1_deg": 10.0, "zero2_deg": -20.0}, who=ARIE)
    assert adc.angles == (100.0, 100.0)  # the same stages, a new reference
    s = adc.status()
    assert (s.stage1_deg, s.stage2_deg) == (110.0, 80.0) and (s.pair1_deg, s.pair2_deg) == (
        100.0,
        100.0,
    )
    adc.cal.patch({"zero1_deg": 0.0, "zero2_deg": 0.0}, who=ARIE)
    assert adc.angles == (110.0, 80.0)


async def test_the_pairs_are_unknown_while_they_move_and_a_second_move_is_refused():
    adc = await started(speed_deg_s=100.0)
    task = asyncio.create_task(adc.move_to(90.0, 270.0))
    await asyncio.sleep(0.05)
    assert adc.status().moving and adc.status().pair1_deg is None
    assert adc.activities.names == ["Moving"]
    with pytest.raises(AdcBusy):
        await adc.move_to(0.0, 0.0)
    await task
    assert adc.activities.names == []


import asyncio  # noqa: E402  (the tests above use it too)

# -- tracking ----------------------------------------------------------------------------


class Sky:
    def __init__(self, zd=40.0, q=30.0, rot=10.0):
        self.state = SkyState(zd, q, rot)

    def __call__(self):
        return self.state


async def test_tracking_follows_the_zenith_distance_and_the_parallactic_angle():
    sky = Sky(zd=40.0, q=30.0, rot=10.0)
    adc = await started(sky=sky)
    await adc.start_tracking()
    await adc.track_once()
    t = adc.status().target
    assert t is not None and t.atmosphere_arcsec == pytest.approx(
        model.atmospheric_dispersion_arcsec(40.0, SEA_LEVEL)
    )
    assert t.direction_deg == pytest.approx(20.0)  # the parallactic angle minus the slit PA
    d, theta = model.delivered(*adc.angles, 6.0)
    assert d == pytest.approx(t.dispersion_arcsec) and theta == pytest.approx(20.0)
    assert adc.status().tracking_state == "holding"
    await adc.stop_tracking()


async def test_tracking_moves_only_when_it_is_off_by_more_than_the_dead_band():
    sky = Sky()
    adc = await started(sky=sky)
    moves = []
    original = adc._move
    adc._move = lambda a, b: (moves.append((a, b)), original(a, b))[1]
    await adc.start_tracking()
    await adc.track_once()
    assert len(moves) == 1
    await adc.track_once()  # nothing changed
    assert len(moves) == 1
    sky.state = SkyState(40.1, 30.05, 10.0)  # a hair: well inside 0.25 degrees
    await adc.track_once()
    assert len(moves) == 1
    sky.state = SkyState(50.0, 60.0, 10.0)  # a different sky
    await adc.track_once()
    assert len(moves) == 2
    await adc.stop_tracking()


async def test_the_slit_pa_turns_the_direction():
    sky = Sky(zd=30.0, q=30.0, rot=0.0)
    adc = await started(sky=sky)
    await adc.start_tracking()
    await adc.track_once()
    before = adc.status().direction_deg
    sky.state = SkyState(30.0, 30.0, 25.0)  # the slit turned 25 degrees: so did the direction, back
    await adc.track_once()
    assert (adc.status().direction_deg - before + 180) % 360 - 180 == pytest.approx(-25.0, abs=1e-6)
    await adc.stop_tracking()


async def test_the_direction_offset_is_a_calibration():
    adc = await started(sky=Sky(zd=30.0, q=30.0, rot=10.0))
    adc.cal.patch({"direction_offset_deg": 7.0}, who=ARIE)
    await adc.start_tracking()
    await adc.track_once()
    assert adc.status().target.direction_deg == pytest.approx(27.0)
    await adc.stop_tracking()


async def test_without_a_target_tracking_waits():
    adc = await started()
    await adc.start_tracking()
    await adc.track_once()
    s = adc.status()
    assert s.tracking and s.tracking_state == "waiting" and "no target" in s.tracking_note
    assert s.target is None
    await adc.stop_tracking()
    assert adc.status().tracking_state == "off"


async def test_when_the_atmosphere_is_more_than_it_can_cancel_it_makes_its_most_and_says_so():
    adc = await started(sky=Sky(zd=62.0))  # the sea-level atmosphere: more than 6 arcsec
    await adc.start_tracking()
    await adc.track_once()
    s = adc.status()
    assert s.target.saturated and s.dispersion_arcsec == pytest.approx(6.0)
    assert "more than" in s.tracking_note
    await adc.stop_tracking()


async def test_a_hand_move_is_refused_while_tracking_and_allowed_after():
    adc = await started(sky=Sky())
    await adc.start_tracking()
    with pytest.raises(AdcBusy, match="tracking is on"):
        await adc.move_to(1.0, 2.0)
    with pytest.raises(AdcBusy, match="tracking is on"):
        await adc.move_dispersion(1.0, 2.0)
    await adc.stop_tracking()
    await adc.move_to(1.0, 2.0)


async def test_the_tracking_task_keeps_going_by_itself_and_survives_a_bad_look():
    calls = []

    def sky():
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("the telescope did not answer")
        return SkyState(40.0, 30.0, 10.0)

    adc = await started(sky=sky)
    adc.conf.patch({"update_s": 0.5}, who=ARIE)
    adc._conf.current  # noqa: B018
    await adc.start_tracking()
    await asyncio.sleep(0.2)
    assert adc.status().tracking_state == "error" and "did not answer" in adc.status().tracking_note
    await asyncio.sleep(0.7)
    assert adc.status().tracking_state == "holding" and adc.status().target is not None
    await adc.stop_tracking()


async def test_shutdown_stops_tracking_and_leaves_the_pairs():
    adc = await started(sky=Sky())
    await adc.start_tracking()
    await adc.track_once()
    where = adc.angles
    await adc.shutdown()
    assert not adc.tracking and adc.angles == where


async def test_tracking_needs_the_adc_to_be_started_up():
    adc = make()
    with pytest.raises(ApiError) as e:
        await adc.start_tracking()
    assert e.value.body.code == "not_operational"


# -- HTTP --------------------------------------------------------------------------------


@pytest.fixture
def client(config_file, passphrases_file):
    passphrases_file('[passphrases]\nto = "to-pass"\n')
    lease.state.sessions.clear()
    lease.state.holder_id = lease.state.token = None
    manager.reset()
    registry.clear()
    with TestClient(create_app()) as c:
        yield c
    manager.reset()
    registry.clear()


def hold_lease(client) -> dict[str, str]:
    s = client.post("/sessions", json={"name": "Arie"}).json()
    sid = {"X-Session-Id": s["session_id"]}
    client.post("/control/requests", headers=sid)
    client.post(f"/control/requests/{s['code']}/grant", headers={"X-TO-Passphrase": "to-pass"})
    token = client.post(f"/sessions/{s['session_id']}/heartbeat").json()["lease_token"]
    return {**sid, "X-Lease-Token": token}


def test_the_status_and_the_system_carry_the_adc(client):
    body = client.get("/status").json()
    adc = body["adc"]
    assert adc["kind"] == "simulated" and adc["referenced"] is True and adc["tracking"] is False
    assert adc["dmax_arcsec"] == 6.0 and adc["lifecycle"]["operational"] is True
    assert "adc" in body["system"]["components"]
    assert client.get("/adc").json() == adc


def test_its_settings_and_calibration_are_served_by_the_config_endpoints(client):
    conf = client.get("/config/adc/conf").json()
    assert conf["values"]["dmax_arcsec"] == 6.0
    assert conf["json_schema"]["properties"]["deadband_deg"]["unit"] == "°"
    cal = client.get("/config/adc/cal").json()
    assert cal["values"]["zero1_deg"] == 0.0 and "direction_offset_deg" in cal["values"]


def test_moving_and_tracking_need_the_lease(client):
    assert client.post("/adc/move", json={"pair1_deg": 1, "pair2_deg": 2}).status_code == 403
    assert client.post("/adc/tracking", json={"enabled": True}).status_code == 403


def test_move_the_pairs_by_angles_or_by_dispersion(client):
    h = hold_lease(client)
    r = client.post("/adc/move", json={"pair1_deg": 100, "pair2_deg": 40}, headers=h)
    assert r.status_code == 200 and r.json()["pair1_deg"] == 100 and r.json()["pair2_deg"] == 40
    assert r.json()["dispersion_arcsec"] == pytest.approx(6.0 * math.sin(math.radians(30)))
    r = client.post("/adc/move", json={"dispersion_arcsec": 2.0, "direction_deg": 15}, headers=h)
    assert r.json()["dispersion_arcsec"] == pytest.approx(2.0) and r.json()[
        "direction_deg"
    ] == pytest.approx(15)


def test_a_bad_request_is_422(client):
    h = hold_lease(client)
    for body in (
        {},
        {"pair1_deg": 1},
        {"pair1_deg": 1, "pair2_deg": 2, "direction_deg": 3},
        {"dispersion_arcsec": 2},
    ):
        assert client.post("/adc/move", json=body, headers=h).status_code == 422, body
    r = client.post("/adc/move", json={"dispersion_arcsec": 7.0, "direction_deg": 0}, headers=h)
    assert r.status_code == 422 and r.json()["code"] == "invalid"  # more than the ADC can make


def test_tracking_follows_the_telescopes_target_and_refuses_hand_moves(client):
    h = hold_lease(client)
    tel = telescope.get_telescope()
    tel.slew_s, tel.rotate_deg_s = 0.1, 400.0
    # a target 30 degrees from the zenith, 50 west of the meridian, at a fixed time
    now = time.time()
    tel.utc = lambda: now
    ra = (site.local_sidereal_deg(now) - 50.0) / 15.0
    r = client.post(
        "/telescope/slew",
        json={"name": "T", "ra_hours": ra % 24, "dec_deg": 25.0, "rot_pa": 20.0},
        headers=h,
    )
    assert r.status_code == 202
    time.sleep(0.6)
    on = client.post("/adc/tracking", json={"enabled": True}, headers=h).json()
    assert on["tracking"] is True
    deadline = time.time() + 6
    while time.time() < deadline and client.get("/adc").json()["target"] is None:
        time.sleep(0.1)
    s = client.get("/adc").json()
    t = s["target"]
    assert t is not None and 0 < t["zenith_distance_deg"] < 60
    assert t["parallactic_angle_deg"] > 20  # west of the meridian
    assert t["rot_pa_deg"] == pytest.approx(20.0)
    assert t["atmosphere_arcsec"] > 0.5
    r = client.post("/adc/move", json={"pair1_deg": 1, "pair2_deg": 2}, headers=h)
    assert r.status_code == 409 and r.json()["code"] == "busy"
    off = client.post("/adc/tracking", json={"enabled": False}, headers=h).json()
    assert off["tracking"] is False and off["tracking_state"] == "off"


def test_the_telescope_reports_the_zenith_distance_and_parallactic_angle_of_its_target(client):
    h = hold_lease(client)
    tel = telescope.get_telescope()
    tel.slew_s = 0.1
    t0 = client.get("/status").json()["telescope"]["telemetry"]
    assert (
        t0["zenith_distance_deg"] is None and t0["parallactic_angle_deg"] is None
    )  # no target yet
    client.post(
        "/telescope/slew",
        json={"name": "M13", "ra_hours": 16.6949, "dec_deg": 36.4613, "rot_pa": 0},
        headers=h,
    )
    time.sleep(0.4)
    t1 = client.get("/status").json()["telescope"]["telemetry"]
    assert t1["zenith_distance_deg"] is not None and -180 <= t1["parallactic_angle_deg"] <= 180


# -- an ADC that is not installed -----------------------------------------------------------


def test_the_adc_is_installed_by_default():
    from wilds.control.adc import AdcConf

    assert AdcConf().installed is True


async def test_not_installed_every_command_is_refused_and_nothing_is_started():
    adc = make(sky=Sky())
    adc.conf.patch({"installed": False}, who=ARIE)
    status = await adc.startup()
    assert not status.present and not adc.status().referenced  # nothing was started
    s = adc.status()
    assert s.installed is False and s.tracking is False and s.tracking_state == "off"
    assert "not installed" in s.lifecycle.why_not_operational
    for call in (
        lambda: adc.move_to(1.0, 2.0),
        lambda: adc.move_dispersion(1.0, 2.0),
        lambda: adc.start_tracking(),
    ):
        with pytest.raises(ApiError) as e:
            await call()
        assert e.value.status == 409 and e.value.body.code == "not_installed"
    await adc.track_once()  # a look does nothing
    assert adc.status().target is None


async def test_taking_it_out_ends_tracking_and_putting_it_back_allows_it_again():
    adc = await started(sky=Sky())
    adc.conf.patch({"update_s": 0.5}, who=ARIE)
    await adc.start_tracking()
    await asyncio.sleep(0.1)
    assert adc.tracking
    adc.conf.patch({"installed": False}, who=ARIE)
    assert (
        not adc.tracking and adc.status().tracking_state == "off"
    )  # at once, as far as anyone asks
    await asyncio.sleep(0.8)  # and the task has gone by its next look
    assert adc._task is None
    adc.conf.patch({"installed": True}, who=ARIE)
    assert not adc.tracking  # it does not start by itself
    await adc.start_tracking()
    await adc.track_once()
    assert adc.tracking and adc.status().target is not None
    await adc.stop_tracking()


def test_the_installed_toggle_is_in_the_settings_and_the_system_carries_on_without_the_adc(client):
    conf = client.get("/config/adc/conf").json()
    assert conf["values"]["installed"] is True
    assert conf["json_schema"]["properties"]["installed"]["title"] == "Installed"
    h = hold_lease(client)
    r = client.put(
        "/config/adc/conf",
        json={
            "values": {"installed": False},
            "version": conf["version"],
            "reason": "no ADC fitted",
        },
        headers=h,
    )
    assert r.status_code == 200 and r.json()["values"]["installed"] is False
    body = client.get("/status").json()
    assert body["adc"]["installed"] is False and body["adc"]["tracking"] is False
    system = body["system"]
    assert system["components"]["adc"]["present"] is False
    assert (
        system["operational"] is True and system["opstate"] == "startedup"
    )  # WILDS does not need it
    assert all("adc" not in reason for reason in system["why_not_operational"])
    r = client.post("/adc/move", json={"pair1_deg": 1, "pair2_deg": 2}, headers=h)
    assert r.status_code == 409 and r.json()["code"] == "not_installed"
    r = client.post("/adc/tracking", json={"enabled": True}, headers=h)
    assert r.status_code == 409 and r.json()["code"] == "not_installed"
    # shutting WILDS down and starting it up again leaves the ADC alone too
    assert client.post(
        "/system/adc/startup", headers={"X-TO-Passphrase": "to-pass"}
    ).status_code in (200, 202)
    assert client.get("/status").json()["system"]["operational"] is True
