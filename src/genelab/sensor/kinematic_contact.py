"""Kinematic contact-probe sensor (Genesis 1.0 ``ContactProbe``).

Wraps :class:`gs.sensors.ContactProbe`: a set of point probes rigidly attached to a robot
link that report contact depth (or binary contact when read against a threshold). Use this
for "did this point on the robot touch something" without engaging the tactile-stack
dynamics that the deformable-elastomer sensors carry.

Probe positions are specified in the parent link's local frame; binary contact is reported
whenever the per-probe contact depth exceeds ``contact_threshold`` (Genesis default
``1e-4`` m). ``history_length=0`` (default) reads only the current step; setting it to
``H > 0`` returns a rolling ``(num_envs, H, num_probes)`` ring buffer that Genesis
maintains on the GPU.
"""

from dataclasses import dataclass
from typing import Any

import torch

from genelab.sensor.sensor import Sensor, SensorCfg


@dataclass
class KinematicContactSensorCfg(SensorCfg):
    """Configuration for :class:`KinematicContactSensor`.

    ``link_name`` resolves through ``env.link_names`` and selects the parent link the
    probes ride on. ``probe_local_pos`` is the tuple of probe positions in the link's
    local frame. ``history_length`` forwards to the Genesis sensor constructor — leave
    at ``0`` for single-step reads.
    """

    link_name: str = ""
    probe_local_pos: tuple[tuple[float, float, float], ...] = ((0.0, 0.0, 0.0),)
    probe_radius: float = 0.01
    contact_threshold: float = 1e-4
    history_length: int = 0

    def build(self) -> "KinematicContactSensor":
        return KinematicContactSensor(self)


@dataclass
class KinematicContactData:
    """Per-step contact-probe snapshot.

    ``depth`` is whatever Genesis ``ContactProbe.read`` returns (per-probe contact depth
    in metres, ``(num_envs, [history,] num_probes)``). ``in_contact`` thresholds the
    depth at ``cfg.contact_threshold`` so consumers that only care about the boolean
    don't have to repeat the comparison.
    """

    depth: torch.Tensor
    in_contact: torch.Tensor


class KinematicContactSensor(Sensor[KinematicContactData]):
    def __init__(self, cfg: KinematicContactSensorCfg) -> None:
        super().__init__(cfg)
        self._cfg_typed = cfg
        self._gs_sensor: Any | None = None

    def pre_build_genesis(self, gs_scene: Any, entities: dict[str, Any]) -> None:
        """Construct ``gs.sensors.ContactProbe`` options and register with ``gs_scene``.

        Resolves ``cfg.link_name`` against the named entity's ``link_names``; raises
        :class:`ValueError` for an empty or unknown name. The returned Genesis sensor
        handle is stashed for per-step reads in :meth:`_compute_data`.
        """
        if not self._cfg_typed.link_name:
            raise ValueError(
                f"KinematicContactSensorCfg(name={self._cfg.name!r}) requires link_name"
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

        opts = gs.sensors.ContactProbe(
            entity_idx=entity.gs_handle.idx,
            link_idx_local=link_idx,
            probe_local_pos=self._cfg_typed.probe_local_pos,
            probe_radius=self._cfg_typed.probe_radius,
            contact_threshold=self._cfg_typed.contact_threshold,
            history_length=self._cfg_typed.history_length,
        )
        self._gs_sensor = gs_scene.add_sensor(opts)

    def _compute_data(self) -> KinematicContactData:
        assert self._gs_sensor is not None, (
            f"KinematicContactSensor(name={self._cfg.name!r}) read before pre_build_genesis"
        )
        depth = self._gs_sensor.read()
        in_contact = depth > self._cfg_typed.contact_threshold
        return KinematicContactData(depth=depth, in_contact=in_contact)
