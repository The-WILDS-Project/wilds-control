"""The guide camera: an abstract base, a simulator, and the one instance the app uses."""

import logging
from pathlib import Path

from .. import config
from ..conf import confs
from ..lifecycle import system
from ..light_path import get_light_path
from .asi import ASICamera
from .base import (
    CameraBusy,
    CaptureSettings,
    CaptureUpdate,
    Cooler,
    CoolerRequest,
    ExposureAborted,
    Frame,
    FrameMeta,
    GuideCameraActivity,
    GuideCameraBase,
    GuideCameraCal,
    GuideCameraConf,
    GuideCameraStatus,
    SettingError,
    SettingsUpdate,
)
from .simulated import SimulatedASICamera

log = logging.getLogger(__name__)

__all__ = [
    "ASICamera",
    "CameraBusy",
    "CaptureSettings",
    "CaptureUpdate",
    "Cooler",
    "CoolerRequest",
    "ExposureAborted",
    "Frame",
    "FrameMeta",
    "GuideCameraActivity",
    "GuideCameraBase",
    "GuideCameraCal",
    "GuideCameraConf",
    "GuideCameraStatus",
    "SettingError",
    "SettingsUpdate",
    "SimulatedASICamera",
    "close_guide_camera",
    "get_guide_camera",
    "reset",
]

_camera: GuideCameraBase | None = None


def get_guide_camera() -> GuideCameraBase:
    """The one camera, made on first use (the app does so at startup) from
    ``[guide_camera] kind`` in config.toml. Use it as a FastAPI ``Depends``; tests override it
    through ``app.dependency_overrides``."""
    global _camera
    if _camera is None:
        kind = config.get().guide_camera.kind
        camera = (
            ASICamera()
            if kind == "asi"
            else SimulatedASICamera(light_blocked=lambda: get_light_path().blocked)
        )
        confs.add(camera.conf)  # for the /config endpoints
        confs.add(camera.cal)
        if (
            isinstance(camera, SimulatedASICamera)
            and config.get().guide_camera.seed_simulated_calibration
        ):
            _seed_simulated_calibration(camera)
        if isinstance(camera, SimulatedASICamera):
            from ..slit import get_slit

            camera.slit_width = lambda: get_slit().width_arcsec
            camera.field = lambda: simulated_field(camera)
            if (sky := _make_sky()) is not None:
                camera.sky = lambda: real_sky(camera, sky)
        system.register(camera.life)  # started up by the system (or by hand), not here
        _camera = camera
    return _camera


def simulated_field(camera: SimulatedASICamera) -> tuple[float, float, float, float, float]:
    """How the simulated telescope has moved the simulated camera's field (see
    ``SimulatedASICamera.field``): the rotator turns the image about the slit (the camera is on
    the instrument), and the pointing offsets shift the stars the other way. Uses the camera's
    true optics (plate scale, orientation, parity, slit pixel), not the stored calibration, which
    is what a calibration has to find."""
    import math

    from ..telescope import get_telescope

    o = camera.optics  # the true geometry, not the stored calibration
    scale, parity, theta0 = o.plate_scale_arcsec_px, o.parity, o.orientation_deg
    centre = (o.slit_x_px, o.slit_y_px)
    t = get_telescope().telemetry()
    pa = t.rot_pa or 0.0
    north = -math.radians(theta0 - parity * pa)  # clockwise on the screen, as slit.js has it
    n_img = (math.sin(north), -math.cos(north))
    e_img = (parity * -math.cos(north), parity * -math.sin(north))
    d1 = sum(o.d1_arcsec for o in t.offsets.values())  # east
    d2 = sum(o.d2_arcsec for o in t.offsets.values())  # north
    # pointing the telescope east moves the stars west: opposite to the offset
    dx = -(d1 * e_img[0] + d2 * n_img[0]) / scale
    dy = -(d1 * e_img[1] + d2 * n_img[1]) / scale
    return dx, dy, -parity * pa, centre[0], centre[1]


def _make_sky():
    """The configured star catalogue as a ``Sky``, or ``None`` (no ``[sky] gaia_root``, or it is
    not a folder): the simulated camera then draws its invented field."""
    from ..sky import CatsHtmSource, FieldFileSource, Sky
    from .simulated import SENSOR

    cfg = config.get().sky
    if cfg.field_file is not None:
        path = cfg.field_file
        if not path.is_absolute() and not path.exists():  # relative to the checkout, not the cwd
            path = Path(__file__).resolve().parents[4] / path
        if path.is_file():
            return Sky(FieldFileSource(path), SENSOR)
        log.warning("[sky] field_file %s is not a file: looking for a catalogue", cfg.field_file)
    if cfg.gaia_root is None:
        return None
    if not cfg.gaia_root.is_dir():
        log.warning(
            "[sky] gaia_root %s is not a folder: using the invented star field", cfg.gaia_root
        )
        return None
    source = CatsHtmSource(str(cfg.gaia_root), cfg.catalog_name, cfg.mag_column, cfg.mag_limit)
    return Sky(source, SENSOR)


def real_sky(camera: SimulatedASICamera, sky):
    """The real stars of the frame for the simulated camera, or ``None`` (no target yet, or the
    catalogue could not be read): see ``SimulatedASICamera.sky`` and ``sky.py``."""
    from ..sky import Geometry, pointing_after_offsets
    from ..telescope import get_telescope

    t = get_telescope().telemetry()
    if t.target is None:
        return None
    o = camera.optics  # the true geometry, not the stored calibration
    geo = Geometry(
        plate_scale_arcsec_px=o.plate_scale_arcsec_px,
        slit_x=o.slit_x_px,
        slit_y=o.slit_y_px,
        orientation_deg=o.orientation_deg,
        parity=o.parity,
        pa_deg=t.rot_pa or 0.0,
    )
    east = sum(o.d1_arcsec for o in t.offsets.values())
    north = sum(o.d2_arcsec for o in t.offsets.values())
    ra, dec = pointing_after_offsets(t.target.ra_hours * 15.0, t.target.dec_deg, east, north)
    return sky.stars_for(ra, dec, geo)


def _seed_simulated_calibration(camera: SimulatedASICamera) -> None:
    """A rough first guess at the geometry (a little off the simulator's true optics, which are
    hidden), so the live view's slit overlay can be drawn before the calibration has measured
    anything: it is close, and the calibration visibly improves it. Only when nothing is set."""
    from ..jobs import StartedBy
    from .simulated import SENSOR

    if camera.cal.current.plate_scale_arcsec_px is not None:
        return
    camera.cal.patch(
        {
            "plate_scale_arcsec_px": 0.25,
            "slit_pixel_x": SENSOR[0] / 2,
            "slit_pixel_y": SENSOR[1] / 2,
            "orientation_deg": 25.0,
            "parity": 1,
            "slit_length_arcsec": 30.0,
            "slit_width_arcsec": 1.0,
        },
        who=StartedBy(name="simulator", code="SIM"),
        reason="simulated defaults",
    )


async def close_guide_camera() -> None:
    global _camera
    camera, _camera = _camera, None
    if camera is not None:
        await camera.close()


def reset() -> None:
    """Forget the instance without closing it (for tests)."""
    global _camera
    _camera = None
