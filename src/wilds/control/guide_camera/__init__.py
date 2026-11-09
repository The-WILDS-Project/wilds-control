"""The guide camera: an abstract base, a simulator, and the one instance the app uses."""

from .. import config
from ..conf import confs
from ..lifecycle import system
from .asi import ASICamera
from .base import (
    CameraBusy,
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
        camera = ASICamera() if kind == "asi" else SimulatedASICamera()
        confs.add(camera.conf)  # for the /config endpoints
        confs.add(camera.cal)
        system.register(camera.life)  # started up by the system (or by hand), not here
        _camera = camera
    return _camera


async def close_guide_camera() -> None:
    global _camera
    camera, _camera = _camera, None
    if camera is not None:
        await camera.close()


def reset() -> None:
    """Forget the instance without closing it (for tests)."""
    global _camera
    _camera = None
