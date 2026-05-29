"""Elastomer-displacement tactile sensor (Genesis 1.0 ``ElastomerTaxel``).

Wraps :class:`gs.sensors.ElastomerTaxel`: a point-cloud-sampled tactile sensor with a
deformable elastomer contact model. Probes are sampled from the parent link's mesh
(visual mesh by default; ``use_visual_mesh=False`` falls back to collision geometry)
and report a displacement / contact response governed by the Lamé parameters
``lambda_d`` (volumetric) and ``lambda_s`` (shear).

Use this where the task needs a tactile signal that respects the elastomer's
finite-stiffness contact dynamics — peg-in-hole, grasp-stability, dexterous
manipulation. The lighter-weight :class:`~genelab.sensor.PointCloudTactileSensor`
(over Genesis ``ProximityTaxel``) is a better fit when a stiffness-only model is
enough.
"""

from dataclasses import dataclass
from typing import Any

import torch

from genelab.sensor.sensor import Sensor, SensorCfg


@dataclass
class ElastomerTactileSensorCfg(SensorCfg):
    """Configuration for :class:`ElastomerTactileSensor`.

    ``link_name`` is the parent link the elastomer rides on. ``track_link_names`` is
    the non-empty set of links the elastomer is allowed to deform against (Genesis
    rejects an empty target list, mirroring :class:`~genelab.sensor.ProximitySensor`).
    ``n_sample_points`` controls how densely the link mesh is sampled for the point
    cloud; ``lambda_d`` / ``lambda_s`` are the elastomer's Lamé parameters.
    """

    link_name: str = ""
    probe_local_pos: tuple[tuple[float, float, float], ...] = ((0.0, 0.0, 0.0),)
    probe_local_normal: tuple[float, float, float] = (0.0, 0.0, 1.0)
    probe_radius: float = 0.01
    track_link_names: tuple[str, ...] = ()
    n_sample_points: int = 500
    use_visual_mesh: bool = True
    lambda_d: float = 700.0
    lambda_s: float = 300.0
    history_length: int = 0

    def build(self) -> "ElastomerTactileSensor":
        return ElastomerTactileSensor(self)


@dataclass
class ElastomerTactileData:
    """Per-step elastomer-tactile snapshot.

    ``raw`` holds whatever ``gs.sensors.ElastomerTaxel.read()`` returns —
    ``(num_envs, [history,] ...)`` per Genesis's per-sensor cache layout. Downstream
    rewards in :mod:`genelab.mdp.rewards.tactile` treat the trailing axes as
    per-probe channels and aggregate via ``sum`` / ``norm``; consumers that need a
    specific decomposition (normal vs shear) should slice it themselves.
    """

    raw: torch.Tensor


class ElastomerTactileSensor(Sensor[ElastomerTactileData]):
    def __init__(self, cfg: ElastomerTactileSensorCfg) -> None:
        super().__init__(cfg)
        self._cfg_typed = cfg
        self._gs_sensor: Any | None = None

    def pre_build_genesis(self, gs_scene: Any, entities: dict[str, Any]) -> None:
        """Construct ``gs.sensors.ElastomerTaxel`` options and register with ``gs_scene``."""
        if not self._cfg_typed.link_name:
            raise ValueError(
                f"ElastomerTactileSensorCfg(name={self._cfg.name!r}) requires link_name"
            )
        entity = entities[self._cfg.entity_name]
        try:
            link_idx = entity.link_names.index(self._cfg_typed.link_name)
        except ValueError as exc:
            raise ValueError(
                f"sensor {self._cfg.name!r}: link {self._cfg_typed.link_name!r} not in "
                f"link_names={entity.link_names!r}"
            ) from exc
        if not self._cfg_typed.track_link_names:
            raise ValueError(
                f"ElastomerTactileSensorCfg(name={self._cfg.name!r}) requires at least one "
                "track_link_name — the Genesis ElastomerTaxel needs to know which links "
                "the elastomer can deform against"
            )
        missing = [n for n in self._cfg_typed.track_link_names if n not in entity.link_names]
        if missing:
            raise ValueError(
                f"sensor {self._cfg.name!r}: track_link_names {missing!r} not in "
                f"link_names={entity.link_names!r}"
            )
        track_link_idx = tuple(entity.link_names.index(n) for n in self._cfg_typed.track_link_names)

        import genesis as gs

        opts = gs.sensors.ElastomerTaxel(
            entity_idx=entity.gs_handle.idx,
            link_idx_local=link_idx,
            probe_local_pos=self._cfg_typed.probe_local_pos,
            probe_local_normal=self._cfg_typed.probe_local_normal,
            probe_radius=self._cfg_typed.probe_radius,
            track_link_idx=track_link_idx,
            n_sample_points=self._cfg_typed.n_sample_points,
            use_visual_mesh=self._cfg_typed.use_visual_mesh,
            lambda_d=self._cfg_typed.lambda_d,
            lambda_s=self._cfg_typed.lambda_s,
            history_length=self._cfg_typed.history_length,
        )
        self._gs_sensor = gs_scene.add_sensor(opts)

    def _compute_data(self) -> ElastomerTactileData:
        assert self._gs_sensor is not None, (
            f"ElastomerTactileSensor(name={self._cfg.name!r}) read before pre_build_genesis"
        )
        return ElastomerTactileData(raw=self._gs_sensor.read())
