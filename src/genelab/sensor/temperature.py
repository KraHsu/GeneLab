"""Temperature-grid sensor (Genesis 1.0 ``TemperatureGrid``).

Wraps :class:`gs.sensors.TemperatureGrid`: a regular grid of temperature samples carried
on a robot link, useful as a coarse thermal model for tasks where contact-driven heat
matters (e.g. friction-warmed end-effectors, exhaust plumes, thermal-aware grasping).
The sensor reports a ``grid_size`` array of temperatures per environment.

This is the minimum-viable wrapper: ``grid_size`` and ``ambient_temperature`` cover the
typical "what temperature does my link feel" use case. The richer Genesis surface
(per-material ``properties_dict``, ``heat_generation``, ``convection_coefficient``) is
intentionally omitted from this initial revision — extend the cfg when a downstream task
needs it.
"""

from dataclasses import dataclass
from typing import Any

import torch

from genelab.sensor.sensor import Sensor, SensorCfg


@dataclass
class TemperatureGridSensorCfg(SensorCfg):
    """Configuration for :class:`TemperatureGridSensor`.

    ``link_name`` selects which link of the named entity carries the grid.
    ``grid_size`` is the ``(nx, ny, nz)`` resolution of the sample grid in the link's
    local frame. ``ambient_temperature`` (degrees C) seeds the grid; leave ``None`` to
    use Genesis's per-material defaults.
    """

    link_name: str = ""
    grid_size: tuple[int, int, int] = (1, 1, 1)
    ambient_temperature: float | None = None
    history_length: int = 0

    def build(self) -> "TemperatureGridSensor":
        return TemperatureGridSensor(self)


@dataclass
class TemperatureGridData:
    """Per-step temperature snapshot.

    ``temperature`` shape mirrors Genesis: ``(num_envs, [history,] nx, ny, nz)`` —
    degrees Celsius at each grid cell.
    """

    temperature: torch.Tensor


class TemperatureGridSensor(Sensor[TemperatureGridData]):
    def __init__(self, cfg: TemperatureGridSensorCfg) -> None:
        super().__init__(cfg)
        self._cfg_typed = cfg
        self._gs_sensor: Any | None = None

    def pre_build_genesis(self, gs_scene: Any, entities: dict[str, Any]) -> None:
        """Construct ``gs.sensors.TemperatureGrid`` options and register with ``gs_scene``."""
        if not self._cfg_typed.link_name:
            raise ValueError(
                f"TemperatureGridSensorCfg(name={self._cfg.name!r}) requires link_name"
            )
        entity = entities[self._cfg.entity_name]
        try:
            link_idx = entity.link_names.index(self._cfg_typed.link_name)
        except ValueError as exc:
            raise ValueError(
                f"sensor {self._cfg.name!r}: link {self._cfg_typed.link_name!r} not in "
                f"link_names={entity.link_names!r}"
            ) from exc

        import genesis as gs

        kwargs: dict[str, Any] = dict(
            entity_idx=entity.gs_handle.idx,
            link_idx_local=link_idx,
            grid_size=self._cfg_typed.grid_size,
            history_length=self._cfg_typed.history_length,
        )
        if self._cfg_typed.ambient_temperature is not None:
            kwargs["ambient_temperature"] = self._cfg_typed.ambient_temperature
        opts = gs.sensors.TemperatureGrid(**kwargs)
        self._gs_sensor = gs_scene.add_sensor(opts)

    def _compute_data(self) -> TemperatureGridData:
        assert self._gs_sensor is not None, (
            f"TemperatureGridSensor(name={self._cfg.name!r}) read before pre_build_genesis"
        )
        return TemperatureGridData(temperature=self._gs_sensor.read())
