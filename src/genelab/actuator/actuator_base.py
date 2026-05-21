"""Actuator base class — Isaac Lab-style electromechanical model.

An :class:`ActuatorBase` owns a regex-matched subset of an articulation's actuated joints
and decides *how* a joint-position target reaches Genesis:

* ``ImplicitPDActuator`` lets Genesis solve the PD law internally via ``set_dofs_kp/kv`` +
  ``control_dofs_position``. ``compute`` returns ``None``.
* ``IdealPDActuator`` and ``DCMotorActuator`` compute torque in Python and push it through
  ``control_dofs_force``. ``compute`` returns the per-joint effort tensor.

Articulation construction is two-phase:

* ``ActuatorBaseCfg`` is pure data, frozen by the user.
* ``ActuatorBase.__init__`` is called by :class:`genelab.entity.Articulation` after joint
  enumeration so the actuator knows its ``joint_ids`` / ``dof_ids`` slices.
"""

from dataclasses import dataclass
from typing import Any, Literal

import torch


@dataclass
class ActuatorBaseCfg:
    """Common actuator configuration shared by every subclass.

    ``target_names_expr`` matches joint names via regex. ``stiffness`` / ``damping`` accept
    a scalar (applied to every matched joint) or a dict ``regex -> value`` (last-match wins,
    consistent with :meth:`Articulation.build_per_joint_tensor`).

    ``class_type`` is filled in by ``__post_init__`` on every concrete subclass.
    """

    target_names_expr: tuple[str, ...] = ()
    stiffness: float | dict[str, float] = 0.0
    damping: float | dict[str, float] = 0.0
    effort_limit: float | None = None
    velocity_limit: float | None = None
    armature: float | None = None
    friction: float | None = None
    action_scale: float = 0.25
    class_type: "type[ActuatorBase] | None" = None


class ActuatorBase:
    """Base class for joint-group electromechanical models.

    Subclasses set :attr:`channel` and implement :meth:`compute`. ``ImplicitPDActuator``
    returns ``None`` from compute and relies on the simulator-side PD law; everything else
    returns a torque tensor that ``Articulation`` writes via ``control_dofs_force``.
    """

    channel: Literal["implicit_pd", "force"] = "implicit_pd"

    def __init__(
        self,
        cfg: ActuatorBaseCfg,
        *,
        name: str,
        joint_ids: torch.Tensor,
        dof_ids: torch.Tensor,
        joint_names: list[str],
        num_envs: int,
        device: str,
    ) -> None:
        if joint_ids.numel() != dof_ids.numel():
            raise ValueError(
                f"Actuator(name={name!r}): joint_ids ({joint_ids.numel()}) and dof_ids "
                f"({dof_ids.numel()}) must have the same length"
            )
        self.cfg = cfg
        self.name = name
        self._joint_ids = joint_ids.to(device=device, dtype=torch.long)
        self._dof_ids = dof_ids.to(device=device, dtype=torch.long)
        self._joint_names = list(joint_names)
        self._num_envs = num_envs
        self._device = device
        self._num_joints = joint_ids.numel()
        self._stiffness = self._fanout_scalar_or_dict(cfg.stiffness)
        self._damping = self._fanout_scalar_or_dict(cfg.damping)
        self._effort_limit = (
            torch.full((self._num_joints,), float(cfg.effort_limit), device=device)
            if cfg.effort_limit is not None
            else None
        )
        self._velocity_limit = (
            torch.full((self._num_joints,), float(cfg.velocity_limit), device=device)
            if cfg.velocity_limit is not None
            else None
        )

    # ------------------------------------------------------------------ public API

    def initialize(self, gs_handle: Any) -> None:
        """Push static actuator parameters (kp / kv / force_range / armature / friction) to Genesis.

        Subclasses override to choose whether to write the PD gains to the sim or to zero
        them out (force channel). Common helpers ``_write_pd_gains``, ``_write_force_range``,
        ``_write_armature``, ``_write_friction`` are reusable.
        """
        self._write_force_range(gs_handle)
        self._write_armature(gs_handle)
        self._write_friction(gs_handle)

    def compute(
        self,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
        target_pos: torch.Tensor,
        target_vel: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """Return per-joint effort, or ``None`` for implicit (sim-side) PD.

        ``joint_pos`` / ``joint_vel`` / ``target_pos`` are this actuator's slices with shape
        ``(num_envs, num_joints)``. ``target_vel`` is reserved for future velocity-tracking
        actuator models (currently unused by every shipped subclass).
        """
        raise NotImplementedError

    # ------------------------------------------------------------------ internals

    def _fanout_scalar_or_dict(self, value: float | dict[str, float]) -> torch.Tensor:
        """Resolve a scalar / regex-keyed dict to a ``(num_joints,)`` tensor."""
        if isinstance(value, dict):
            import re

            out = torch.zeros(self._num_joints, device=self._device)
            for pattern, v in value.items():
                try:
                    regex = re.compile(pattern)
                except re.error:
                    regex = re.compile(re.escape(pattern))
                for i, name in enumerate(self._joint_names):
                    if regex.fullmatch(name) or regex.search(name):
                        out[i] = float(v)
            return out
        return torch.full((self._num_joints,), float(value), device=self._device)

    def _write_force_range(self, gs_handle: Any) -> None:
        if self._effort_limit is None:
            return
        setter = getattr(gs_handle, "set_dofs_force_range", None)
        if setter is None:
            return
        lower = -self._effort_limit
        upper = self._effort_limit
        try:
            setter(lower, upper, self._dof_ids)
        except TypeError:
            try:
                setter(lower, upper, dofs_idx_local=self._dof_ids)
            except TypeError:
                pass

    def _write_armature(self, gs_handle: Any) -> None:
        if self.cfg.armature is None:
            return
        setter = getattr(gs_handle, "set_dofs_armature", None)
        if setter is None:
            return
        values = torch.full((self._num_joints,), float(self.cfg.armature), device=self._device)
        try:
            setter(values, self._dof_ids)
        except TypeError:
            try:
                setter(values, dofs_idx_local=self._dof_ids)
            except TypeError:
                pass

    def _write_friction(self, gs_handle: Any) -> None:
        if self.cfg.friction is None:
            return
        setter = getattr(gs_handle, "set_dofs_frictionloss", None) or getattr(
            gs_handle, "set_dofs_friction", None
        )
        if setter is None:
            return
        values = torch.full((self._num_joints,), float(self.cfg.friction), device=self._device)
        try:
            setter(values, self._dof_ids)
        except TypeError:
            try:
                setter(values, dofs_idx_local=self._dof_ids)
            except TypeError:
                pass

    def _write_pd_gains(
        self,
        gs_handle: Any,
        *,
        kp_values: torch.Tensor,
        kv_values: torch.Tensor,
    ) -> None:
        """Push ``kp`` / ``kv`` tensors to Genesis via ``set_dofs_kp`` / ``set_dofs_kv``.

        Wraps each call in a ``TypeError``-fallback because some Genesis versions accept
        ``(values, dof_ids)`` positionally while others require ``dofs_idx_local=…``.
        Skipped per-channel when the matching gain tensor is empty.

        Used by :class:`~genelab.actuator.ideal_pd.IdealPDActuator` (``kp = kv = zeros`` to
        disable the simulator-side PD so Python-side ``compute`` drives ``control_dofs_force``)
        and :class:`~genelab.actuator.implicit_pd.ImplicitPDActuator` (``kp = self._stiffness``,
        ``kv = self._damping`` so Genesis's internal PD runs natively). Per ADR-0003 / R2.4.
        """
        set_kp = getattr(gs_handle, "set_dofs_kp", None)
        set_kv = getattr(gs_handle, "set_dofs_kv", None)
        if set_kp is not None and self._stiffness.numel() > 0:
            try:
                set_kp(kp_values, self._dof_ids)
            except TypeError:
                set_kp(kp_values, dofs_idx_local=self._dof_ids)
        if set_kv is not None and self._damping.numel() > 0:
            try:
                set_kv(kv_values, self._dof_ids)
            except TypeError:
                set_kv(kv_values, dofs_idx_local=self._dof_ids)

    # ------------------------------------------------------------------ properties

    @property
    def joint_ids(self) -> torch.Tensor:
        """Local joint indices (into the articulation's actuated joint list)."""
        return self._joint_ids

    @property
    def dof_ids(self) -> torch.Tensor:
        """Global Genesis DoF indices."""
        return self._dof_ids

    @property
    def joint_names(self) -> list[str]:
        return list(self._joint_names)

    @property
    def num_joints(self) -> int:
        return self._num_joints

    @property
    def stiffness(self) -> torch.Tensor:
        return self._stiffness

    @property
    def damping(self) -> torch.Tensor:
        return self._damping

    @property
    def effort_limit(self) -> torch.Tensor | None:
        return self._effort_limit

    @property
    def velocity_limit(self) -> torch.Tensor | None:
        return self._velocity_limit
