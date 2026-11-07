"""The real ZWO ASI294MM Pro. Not written yet: the SDK wrapper comes after the simulator and
the live view have settled the interface. Choosing ``kind = "asi"`` fails at startup, loudly.

The hooks the SDK wrapper will fill: ``sensor_size``, ``_expose``, ``_abort_exposure``
(ASIStopExposure), ``_read_temperature``, ``_read_cooler_power``, ``_write_cooler``
(ASI_COOLER_ON, ASI_TARGET_TEMP).
"""

from typing import ClassVar, NoReturn

import numpy as np

from .base import GuideCameraBase, Settings

_MSG = 'the ZWO ASI SDK wrapper is not written yet; use [guide_camera] kind = "simulated"'


class ASICamera(GuideCameraBase):
    kind: ClassVar[str] = "asi"

    def model_post_init(self, __context) -> None:
        raise NotImplementedError(_MSG)

    @property
    def sensor_size(self) -> tuple[int, int]:
        raise NotImplementedError(_MSG)

    def _expose(self, settings: Settings) -> np.ndarray:
        raise NotImplementedError(_MSG)

    def _abort_exposure(self) -> None:
        raise NotImplementedError(_MSG)

    def _read_temperature(self) -> float:
        raise NotImplementedError(_MSG)

    def _read_cooler_power(self) -> float:
        raise NotImplementedError(_MSG)

    def _write_cooler(self, enabled: bool, setpoint_c: float) -> NoReturn:
        raise NotImplementedError(_MSG)
