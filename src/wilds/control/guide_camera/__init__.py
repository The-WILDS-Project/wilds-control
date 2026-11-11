"""The guide camera: an abstract base, a simulator, and the one instance the app uses."""

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
            camera.field = lambda: simulated_field(camera)
        system.register(camera.life)  # started up by the system (or by hand), not here
        _camera = camera
    return _camera


def simulated_field(camera: SimulatedASICamera) -> tuple[float, float, float, float, float]:
    """How the simulated telescope has moved the simulated camera's field (see
    ``SimulatedASICamera.field``): the rotator turns the image about the slit (the camera is on
    the instrument), and the pointing offsets shift the stars the other way. Uses the camera's
    calibration (plate scale, orientation, parity, slit pixel) for the geometry; zeros until
    there is a telescope to follow."""
    import math

    from ..telescope import get_telescope
    from .simulated import SENSOR

    cal = camera.cal.current
    scale = cal.plate_scale_arcsec_px or 0.25
    parity = cal.parity or 1
    theta0 = cal.orientation_deg or 0.0
    centre = (
        cal.slit_pixel_x if cal.slit_pixel_x is not None else SENSOR[0] / 2,
        cal.slit_pixel_y if cal.slit_pixel_y is not None else SENSOR[1] / 2,
    )
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


def _seed_simulated_calibration(camera: SimulatedASICamera) -> None:
    """Where the slit is, in the simulated camera's own frame, so the live view's slit overlay
    can be drawn before anyone has measured a real one. Only when nothing is set yet."""
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
            "slit_width_arcsec": 1.5,
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
