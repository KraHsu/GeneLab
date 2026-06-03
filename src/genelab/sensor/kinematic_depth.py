"""Kinematic contact-depth-probe sensor (Genesis 1.0 ``ContactDepthProbe``).

Wraps :class:`gs.sensors.ContactDepthProbe`: a set of point probes rigidly attached to a
robot link that report **penetration depth** (metres) against surrounding geometry, via SDF
queries around each probe position rather than the physics solver's contact impulses. Where
:class:`~genelab.sensor.KinematicContactSensor` (``ContactProbe``) returns a Genesis-thresholded
``bool`` mask, this returns the underlying continuous depth — non-negative, ``0`` when the probe
is clear of geometry and growing with inter-penetration.

This is geometric / SDF tactile (penetration only), not a deformable-skin FEM or optical
tactile model. ``probe_radius_noise > 0`` enables Genesis's dual-radius sim2real scheme (the
measured branch perturbs each probe radius per step).

Probe positions are in the parent link's local frame. ``history_length=0`` (default) reads only
the current step; ``H > 0`` returns a rolling ring buffer with a leading history axis.
"""

from dataclasses import dataclass
from typing import Any

import torch

from genelab.sensor.sensor import Sensor, SensorCfg


@dataclass
class KinematicDepthSensorCfg(SensorCfg):
    """Configuration for :class:`KinematicDepthSensor`.

    ``link_name`` resolves through the named entity's ``link_names`` and selects the parent
    link the probes ride on. ``probe_local_pos`` is the tuple of probe positions in that link's
    local frame. ``probe_radius`` is the SDF query radius; ``probe_radius_noise`` (metres) turns
    on the dual-radius sim2real perturbation. ``history_length`` forwards to the Genesis sensor
    constructor — leave at ``0`` for single-step reads.
    """

    link_name: str = ""
    probe_local_pos: tuple[tuple[float, float, float], ...] = ((0.0, 0.0, 0.0),)
    probe_radius: float = 0.01
    probe_radius_noise: float = 0.0
    history_length: int = 0

    def build(self) -> "KinematicDepthSensor":
        return KinematicDepthSensor(self)


@dataclass
class KinematicDepthData:
    """Per-step contact-depth-probe snapshot.

    ``depth`` is the ``float`` tensor returned by ``gs.sensors.ContactDepthProbe.read()`` —
    shape ``(num_envs, [history,] num_probes)``, penetration depth in metres per probe.
    Non-negative: ``0`` when clear of geometry, growing with inter-penetration. ``raw`` aliases
    ``depth`` so the shape-agnostic reward primitives in :mod:`genelab.mdp.rewards.tactile`
    (``contact_intensity_l2``, ``contact_count``) see a ``.raw`` field.
    """

    depth: torch.Tensor

    @property
    def raw(self) -> torch.Tensor:
        """Alias for :attr:`depth` — kept for the shape-agnostic reward primitives."""
        return self.depth


class KinematicDepthSensor(Sensor[KinematicDepthData]):
    def __init__(self, cfg: KinematicDepthSensorCfg) -> None:
        super().__init__(cfg)
        self._cfg_typed = cfg
        self._gs_sensor: Any | None = None

    def pre_build_genesis(self, gs_scene: Any, entities: dict[str, Any]) -> None:
        """Construct ``gs.sensors.ContactDepthProbe`` options and register with ``gs_scene``.

        Resolves ``cfg.link_name`` against the named entity's ``link_names``; raises
        :class:`ValueError` for an empty or unknown name. The returned Genesis sensor handle is
        stashed for per-step reads in :meth:`_compute_data`.
        """
        if not self._cfg_typed.link_name:
            raise ValueError(f"KinematicDepthSensorCfg(name={self._cfg.name!r}) requires link_name")
        entity = entities[self._cfg.entity_name]
        try:
            link_idx = entity.link_names.index(self._cfg_typed.link_name)
        except ValueError as exc:
            raise ValueError(
                f"sensor {self._cfg.name!r}: link {self._cfg_typed.link_name!r} not in "
                f"link_names={entity.link_names!r}"
            ) from exc

        import genesis as gs

        opts = gs.sensors.ContactDepthProbe(
            entity_idx=entity.gs_handle.idx,
            link_idx_local=link_idx,
            probe_local_pos=self._cfg_typed.probe_local_pos,
            probe_radius=self._cfg_typed.probe_radius,
            probe_radius_noise=self._cfg_typed.probe_radius_noise,
            history_length=self._cfg_typed.history_length,
        )
        self._gs_sensor = gs_scene.add_sensor(opts)

    def _compute_data(self) -> KinematicDepthData:
        assert self._gs_sensor is not None, (
            f"KinematicDepthSensor(name={self._cfg.name!r}) read before pre_build_genesis"
        )
        return KinematicDepthData(depth=self._gs_sensor.read())
