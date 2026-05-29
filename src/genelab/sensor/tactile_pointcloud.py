"""Point-cloud tactile sensor (Genesis 1.0 ``ProximityTaxel``).

Wraps :class:`gs.sensors.ProximityTaxel`: a point-cloud-sampled tactile sensor with a
stiffness-and-shear contact model — lighter than the deformable-elastomer variant
(:class:`~genelab.sensor.ElastomerTactileSensor`) but still gives per-probe contact
information sampled from the link's mesh.

Use this when the task needs tactile feedback (e.g. for a contact-aware reward) but
doesn't require the elastomer's full deformable response.
"""

from dataclasses import dataclass
from typing import Any

import torch

from genelab.sensor.sensor import Sensor, SensorCfg


@dataclass
class PointCloudTactileSensorCfg(SensorCfg):
    """Configuration for :class:`PointCloudTactileSensor`.

    ``link_name`` is the parent link the probe rides on. ``track_link_names`` is the
    non-empty set of links to sense contact against. ``n_sample_points`` controls
    how many points are sampled from the link mesh; ``stiffness`` and
    ``shear_coupling`` define the contact-response model.
    """

    link_name: str = ""
    probe_local_pos: tuple[tuple[float, float, float], ...] = ((0.0, 0.0, 0.0),)
    probe_local_normal: tuple[float, float, float] = (0.0, 0.0, 1.0)
    probe_radius: float = 0.01
    track_link_names: tuple[str, ...] = ()
    n_sample_points: int = 500
    use_visual_mesh: bool = True
    stiffness: float = 100.0
    shear_coupling: float = 0.0
    history_length: int = 0

    def build(self) -> "PointCloudTactileSensor":
        return PointCloudTactileSensor(self)


@dataclass
class PointCloudTactileData:
    """Per-step point-cloud-tactile snapshot.

    ``raw`` is whatever ``gs.sensors.ProximityTaxel.read()`` returns —
    ``(num_envs, [history,] ...)`` per Genesis's per-sensor cache layout. The
    primitives in :mod:`genelab.mdp.rewards.tactile` consume this generically;
    consumers needing a specific decomposition (normal vs shear) should slice it.
    """

    raw: torch.Tensor


class PointCloudTactileSensor(Sensor[PointCloudTactileData]):
    def __init__(self, cfg: PointCloudTactileSensorCfg) -> None:
        super().__init__(cfg)
        self._cfg_typed = cfg
        self._gs_sensor: Any | None = None

    def pre_build_genesis(self, gs_scene: Any, entities: dict[str, Any]) -> None:
        """Construct ``gs.sensors.ProximityTaxel`` options and register with ``gs_scene``."""
        if not self._cfg_typed.link_name:
            raise ValueError(
                f"PointCloudTactileSensorCfg(name={self._cfg.name!r}) requires link_name"
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
                f"PointCloudTactileSensorCfg(name={self._cfg.name!r}) requires at least one "
                "track_link_name — the Genesis ProximityTaxel needs to know which links "
                "to sense contact against"
            )
        missing = [n for n in self._cfg_typed.track_link_names if n not in entity.link_names]
        if missing:
            raise ValueError(
                f"sensor {self._cfg.name!r}: track_link_names {missing!r} not in "
                f"link_names={entity.link_names!r}"
            )
        track_link_idx = tuple(entity.link_names.index(n) for n in self._cfg_typed.track_link_names)

        import genesis as gs

        opts = gs.sensors.ProximityTaxel(
            entity_idx=entity.gs_handle.idx,
            link_idx_local=link_idx,
            probe_local_pos=self._cfg_typed.probe_local_pos,
            probe_local_normal=self._cfg_typed.probe_local_normal,
            probe_radius=self._cfg_typed.probe_radius,
            track_link_idx=track_link_idx,
            n_sample_points=self._cfg_typed.n_sample_points,
            use_visual_mesh=self._cfg_typed.use_visual_mesh,
            stiffness=self._cfg_typed.stiffness,
            shear_coupling=self._cfg_typed.shear_coupling,
            history_length=self._cfg_typed.history_length,
        )
        self._gs_sensor = gs_scene.add_sensor(opts)

    def _compute_data(self) -> PointCloudTactileData:
        assert self._gs_sensor is not None, (
            f"PointCloudTactileSensor(name={self._cfg.name!r}) read before pre_build_genesis"
        )
        return PointCloudTactileData(raw=self._gs_sensor.read())
