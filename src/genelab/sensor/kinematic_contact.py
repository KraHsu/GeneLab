"""Kinematic contact-probe sensor (Genesis 1.0 ``ContactProbe``).

Wraps :class:`gs.sensors.ContactProbe`: a set of point probes rigidly attached to a robot
link that report binary contact. Genesis thresholds internally at ``contact_threshold``
(default ``1e-4`` m of inter-penetration) and ``ContactProbe.read()`` returns a
``(num_envs, [history,] num_probes)`` ``bool`` tensor — the wrapper exposes this as
``data.in_contact``. Use this for "did this point on the robot touch something" without
engaging the tactile-stack dynamics that the deformable-elastomer sensors carry.

Probe positions are in the parent link's local frame. ``history_length=0`` (default)
reads only the current step; ``H > 0`` returns a rolling ring buffer with a leading
history axis.
"""

from dataclasses import dataclass
from typing import Any

import torch

from genelab.sensor._entity import prebuild_link_names
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

    ``in_contact`` is the ``bool`` tensor returned by ``gs.sensors.ContactProbe.read()`` —
    shape ``(num_envs, [history,] num_probes)``. Genesis already applies
    ``cfg.contact_threshold`` internally, so this is the final boolean mask.
    """

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
        link_names = prebuild_link_names(entity)
        try:
            link_idx = link_names.index(self._cfg_typed.link_name)
        except ValueError as exc:
            raise ValueError(
                f"sensor {self._cfg.name!r}: link {self._cfg_typed.link_name!r} not in "
                f"link_names={link_names!r}"
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
        return KinematicContactData(in_contact=self._gs_sensor.read())
