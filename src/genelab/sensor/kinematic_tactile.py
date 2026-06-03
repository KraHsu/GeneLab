"""Kinematic taxel tactile sensor (Genesis 1.0 ``KinematicTaxel``).

Wraps :class:`gs.sensors.KinematicTaxel`: a set of point taxels rigidly attached to a robot
link that estimate **contact force and torque** per probe from a kinematic spring-damper model
driven by SDF penetration depth and the relative motion of the probe and the geometry it
touches. Per Genesis, for each taxel::

    v_n = dot(relative_velocity, probe_normal) * probe_normal
    v_t = relative_velocity - v_n
    s   = penetration ** normal_exponent
    F   = (-normal_stiffness * s * probe_normal) - (normal_damping * s * v_n) - (shear_scalar * v_t)
    T   = cross(probe_local_pos, F) - twist_scalar * dot(relative_angular_velocity, probe_normal) * probe_normal

i.e. a geometric / SDF estimate, **not** the solver's impulse force and **not** a deformable-skin
FEM / optical tactile model. The contact normal is carried implicitly by the direction of
``force`` (its normal component aligns with ``probe_local_normal``). Outputs are in the parent
link's local frame.

Probe positions / normals are in the parent link's local frame. ``history_length=0`` (default)
reads only the current step; ``H > 0`` returns a rolling ring buffer with a leading history axis.
"""

from dataclasses import dataclass
from typing import Any

import torch

from genelab.sensor.sensor import Sensor, SensorCfg


@dataclass
class KinematicTactileSensorCfg(SensorCfg):
    """Configuration for :class:`KinematicTactileSensor`.

    ``link_name`` is the parent link the taxels ride on. ``probe_local_pos`` are the taxel
    positions and ``probe_local_normal`` the shared contact normal, both in that link's local
    frame. ``probe_radius`` is the SDF query radius; ``probe_radius_noise`` (metres) turns on the
    dual-radius sim2real perturbation. The ``normal_*`` / ``shear_scalar`` / ``twist_scalar``
    fields parameterize the spring-damper model (see module docstring). ``normal_exponent`` must
    be ``>= 1.0`` (``1.0`` linear, ``1.5`` Hertzian). ``history_length`` forwards to the Genesis
    sensor constructor — leave at ``0`` for single-step reads.
    """

    link_name: str = ""
    probe_local_pos: tuple[tuple[float, float, float], ...] = ((0.0, 0.0, 0.0),)
    probe_local_normal: tuple[float, float, float] = (0.0, 0.0, 1.0)
    probe_radius: float = 0.01
    probe_radius_noise: float = 0.0
    normal_stiffness: float = 1000.0
    normal_damping: float = 1.0
    normal_exponent: float = 1.0
    shear_scalar: float = 1.0
    twist_scalar: float = 1.0
    history_length: int = 0

    def build(self) -> "KinematicTactileSensor":
        return KinematicTactileSensor(self)


@dataclass
class KinematicTactileData:
    """Per-step kinematic-taxel snapshot.

    Genesis's ``KinematicTaxel.read()`` returns ``KinematicTaxelData(force=…, torque=…)`` with
    two per-probe vector channels, each of shape ``(num_envs, [history,] num_probes, 3)`` in the
    parent link's local frame. The wrapper extracts both as tensors. ``raw`` aliases ``force`` so
    the generic primitives in :mod:`genelab.mdp.rewards.tactile` (``contact_intensity_l2``,
    ``contact_count``, ``slip_penalty``) still see a ``.raw`` field whose last axis is the
    ``(tangent_x, tangent_y, normal)`` force.
    """

    force: torch.Tensor
    torque: torch.Tensor

    @property
    def raw(self) -> torch.Tensor:
        """Alias for :attr:`force` — kept for the shape-agnostic reward primitives."""
        return self.force


class KinematicTactileSensor(Sensor[KinematicTactileData]):
    def __init__(self, cfg: KinematicTactileSensorCfg) -> None:
        super().__init__(cfg)
        self._cfg_typed = cfg
        self._gs_sensor: Any | None = None

    def pre_build_genesis(self, gs_scene: Any, entities: dict[str, Any]) -> None:
        """Construct ``gs.sensors.KinematicTaxel`` options and register with ``gs_scene``.

        Resolves ``cfg.link_name`` against the named entity's ``link_names``; raises
        :class:`ValueError` for an empty or unknown name. The returned Genesis sensor handle is
        stashed for per-step reads in :meth:`_compute_data`.
        """
        if not self._cfg_typed.link_name:
            raise ValueError(
                f"KinematicTactileSensorCfg(name={self._cfg.name!r}) requires link_name"
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

        opts = gs.sensors.KinematicTaxel(
            entity_idx=entity.gs_handle.idx,
            link_idx_local=link_idx,
            probe_local_pos=self._cfg_typed.probe_local_pos,
            probe_local_normal=self._cfg_typed.probe_local_normal,
            probe_radius=self._cfg_typed.probe_radius,
            probe_radius_noise=self._cfg_typed.probe_radius_noise,
            normal_stiffness=self._cfg_typed.normal_stiffness,
            normal_damping=self._cfg_typed.normal_damping,
            normal_exponent=self._cfg_typed.normal_exponent,
            shear_scalar=self._cfg_typed.shear_scalar,
            twist_scalar=self._cfg_typed.twist_scalar,
            history_length=self._cfg_typed.history_length,
        )
        self._gs_sensor = gs_scene.add_sensor(opts)

    def _compute_data(self) -> KinematicTactileData:
        assert self._gs_sensor is not None, (
            f"KinematicTactileSensor(name={self._cfg.name!r}) read before pre_build_genesis"
        )
        raw = self._gs_sensor.read()
        # Genesis ``KinematicTaxel.read()`` returns ``KinematicTaxelData(force=…, torque=…)``;
        # extracting the two tensors keeps the wrapper's contract a plain dataclass of tensors.
        return KinematicTactileData(force=raw.force, torque=raw.torque)
