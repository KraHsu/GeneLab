"""Surface-distance proximity sensor (Genesis 1.0 ``SurfaceDistanceProbe``).

Wraps :class:`gs.sensors.SurfaceDistanceProbe`: a set of point probes rigidly attached to a
robot link that report the distance to the nearest surface among a configurable set of
tracked links. Use this for "how close is point X on my body to terrain / obstacle / other
robot link" queries — proximity gating, soft self-collision avoidance, time-to-contact
estimates.

Probe positions are specified in the parent link's local frame, and ``probe_radius``
caps the search distance: any surface beyond ``probe_radius`` from a probe reports
``probe_radius`` itself. ``track_link_names`` selects which links count as targets;
Genesis requires at least one target (the wrapper raises if it's empty).
"""

from dataclasses import dataclass
from typing import Any

import torch

from genelab.sensor.sensor import Sensor, SensorCfg


@dataclass
class ProximitySensorCfg(SensorCfg):
    """Configuration for :class:`ProximitySensor`.

    ``link_name`` selects the parent link the probes ride on. ``track_link_names`` is the
    non-empty set of "what to detect against" links (resolved against the same entity's
    ``link_names``); a typical use is to track terrain / world-collision links from a
    foot probe. ``probe_radius`` is both the per-probe search radius and the saturation
    value returned when no surface lies within it.
    """

    link_name: str = ""
    probe_local_pos: tuple[tuple[float, float, float], ...] = ((0.0, 0.0, 0.0),)
    probe_radius: float = 10.0
    track_link_names: tuple[str, ...] = ()
    history_length: int = 0

    def build(self) -> "ProximitySensor":
        return ProximitySensor(self)


@dataclass
class ProximityData:
    """Per-step proximity snapshot.

    ``distance`` shape mirrors Genesis: ``(num_envs, [history,] num_probes)`` of
    distances in metres, clipped to ``cfg.probe_radius`` when no tracked surface is
    within reach.
    """

    distance: torch.Tensor


class ProximitySensor(Sensor[ProximityData]):
    def __init__(self, cfg: ProximitySensorCfg) -> None:
        super().__init__(cfg)
        self._cfg_typed = cfg
        self._gs_sensor: Any | None = None

    def pre_build_genesis(self, gs_scene: Any, entities: dict[str, Any]) -> None:
        """Construct ``gs.sensors.SurfaceDistanceProbe`` options and register with ``gs_scene``."""
        if not self._cfg_typed.link_name:
            raise ValueError(f"ProximitySensorCfg(name={self._cfg.name!r}) requires link_name")
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
                f"ProximitySensorCfg(name={self._cfg.name!r}) requires at least one "
                "track_link_name — the Genesis SurfaceDistanceProbe needs to know which "
                "links to measure distance against"
            )
        missing = [n for n in self._cfg_typed.track_link_names if n not in entity.link_names]
        if missing:
            raise ValueError(
                f"sensor {self._cfg.name!r}: track_link_names {missing!r} not in "
                f"link_names={entity.link_names!r}"
            )
        track_link_idx = tuple(entity.link_names.index(n) for n in self._cfg_typed.track_link_names)

        import genesis as gs

        opts = gs.sensors.SurfaceDistanceProbe(
            entity_idx=entity.gs_handle.idx,
            link_idx_local=link_idx,
            probe_local_pos=self._cfg_typed.probe_local_pos,
            probe_radius=self._cfg_typed.probe_radius,
            track_link_idx=track_link_idx,
            history_length=self._cfg_typed.history_length,
        )
        self._gs_sensor = gs_scene.add_sensor(opts)

    def _compute_data(self) -> ProximityData:
        assert self._gs_sensor is not None, (
            f"ProximitySensor(name={self._cfg.name!r}) read before pre_build_genesis"
        )
        return ProximityData(distance=self._gs_sensor.read())
