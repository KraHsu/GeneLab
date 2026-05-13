"""Terrain height-scan sensor: per-ray ``frame_z - hit_z`` (positive = above terrain).

Composes a ``RayCastSensor`` internally rather than subclassing — keeps the Sensor[T] type
parameter clean (this sensor returns a ``torch.Tensor``, the underlying ray cast returns a
``RayCastData``). With the default flat-plane backend, every output entry is
``link_z - ground_height``; for non-flat ground, swap in a ``RayCastSensor`` subclass that
overrides ``_intersect_world_rays``.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from genelab.sensor.ray_cast import GridPattern, RayCastSensor, RayCastSensorCfg
from genelab.sensor.sensor import Sensor, SensorCfg

if TYPE_CHECKING:
    from genelab.envs.manager_based_rl_env import ManagerBasedRlEnv


@dataclass
class TerrainHeightSensorCfg(SensorCfg):
    link_name: str = ""
    pattern: GridPattern = field(default_factory=GridPattern)
    attach_yaw_only: bool = True
    max_distance: float = 10.0
    ground_height: float = 0.0

    def build(self) -> "TerrainHeightSensor":
        return TerrainHeightSensor(self)


class TerrainHeightSensor(Sensor[torch.Tensor]):
    def __init__(self, cfg: TerrainHeightSensorCfg) -> None:
        super().__init__(cfg)
        self._cfg_typed = cfg
        self._inner = RayCastSensor(
            RayCastSensorCfg(
                name=f"{cfg.name}__inner",
                link_name=cfg.link_name,
                pattern=cfg.pattern,
                attach_yaw_only=cfg.attach_yaw_only,
                max_distance=cfg.max_distance,
                ground_height=cfg.ground_height,
            )
        )

    def bind(self, env: "ManagerBasedRlEnv") -> None:
        super().bind(env)
        self._inner.bind(env)

    def update(self, dt: float) -> None:
        super().update(dt)
        self._inner.update(dt)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        super().reset(env_ids)
        self._inner.reset(env_ids)

    def _compute_data(self) -> torch.Tensor:
        raw = self._inner.data
        return raw.ray_starts_w[..., 2] - raw.hit_pos_w[..., 2]
