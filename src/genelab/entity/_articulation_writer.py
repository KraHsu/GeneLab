"""Articulation write-back — the write seam of ``Articulation`` (ADR-0019).

Owns the methods that push state into the Genesis sim: joint / root write-back,
reset to the home pose, and the per-substep actuator target routing. ``Articulation``
composes one (built at ``bind``) and delegates. The cached joint-position target and
the actuator dict are shared **by reference** with ``Articulation`` (mutated in place /
read live), so the articulation's own ``_joint_pos_target`` / ``actuators`` stay
authoritative.
"""

from typing import Any

import torch

from genelab.actuator import ActuatorBase
from genelab.entity._torch import to_tensor


class ArticulationWriter:
    """Pushes state into the Genesis handle for ``Articulation`` (write seam).

    Created at ``Articulation.bind`` once the handle, actuated DoF index, default pose,
    joint-target buffer, and actuators are available.
    """

    def __init__(
        self,
        gs_handle: Any,
        *,
        actuated_dof_idx: torch.Tensor,
        default_joint_pos: torch.Tensor,
        joint_pos_target: torch.Tensor,
        actuators: dict[str, ActuatorBase],
        device: str,
    ) -> None:
        self._gs_handle = gs_handle
        self._actuated_dof_idx = actuated_dof_idx
        self._default_joint_pos = default_joint_pos
        # Shared by reference with Articulation: mutated in place by reset /
        # write_joint_targets_partial, so Articulation._joint_pos_target stays current.
        self._joint_pos_target = joint_pos_target
        self._actuators = actuators
        self._device = device

    def write_joint_state(
        self,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
        env_ids: torch.Tensor,
    ) -> None:
        """Write actuated-joint positions / velocities for ``env_ids``."""
        if env_ids.numel() == 0:
            return
        robot = self._gs_handle
        set_pos = getattr(robot, "set_dofs_position", None)
        set_vel = getattr(robot, "set_dofs_velocity", None)
        # ``envs_idx=`` is a newer Genesis kwarg; fall back to non-batched API on older builds.
        if set_pos is not None:
            try:
                set_pos(joint_pos, self._actuated_dof_idx, envs_idx=env_ids)
            except TypeError:
                set_pos(joint_pos, self._actuated_dof_idx)
        if set_vel is not None:
            try:
                set_vel(joint_vel, self._actuated_dof_idx, envs_idx=env_ids)
            except TypeError:
                set_vel(joint_vel, self._actuated_dof_idx)

    def write_root_state(
        self,
        root_pos: torch.Tensor,
        root_quat: torch.Tensor,
        root_lin_vel_w: torch.Tensor,
        root_ang_vel_w: torch.Tensor,
        env_ids: torch.Tensor,
    ) -> None:
        """Write floating-base pose + velocity for ``env_ids``."""
        if env_ids.numel() == 0:
            return
        robot = self._gs_handle
        for fn_name, value in (
            ("set_pos", root_pos),
            ("set_quat", root_quat),
            ("set_vel", root_lin_vel_w),
            ("set_ang", root_ang_vel_w),
        ):
            fn = getattr(robot, fn_name, None)
            if fn is None:
                continue
            try:
                fn(value, envs_idx=env_ids)
            except TypeError:
                fn(value)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Reset actuated joints to default pose + zero velocity for ``env_ids``."""
        if env_ids.numel() == 0:
            return
        default = self._default_joint_pos.unsqueeze(0).expand(env_ids.numel(), -1).contiguous()
        zeros_v = torch.zeros_like(default)
        self.write_joint_state(default, zeros_v, env_ids)
        # Also reset the cached joint target so the next implicit-PD step uses the home pose.
        # ``index_copy_`` sidesteps Genesis's strict aliasing check on indexed assignment.
        self._joint_pos_target.index_copy_(0, env_ids.long(), default.clone())

    def write_joint_targets_partial(
        self, local_joint_ids: torch.Tensor, target: torch.Tensor
    ) -> None:
        """Stash a slice of joint position targets, then dispatch each actuator.

        ``local_joint_ids`` indexes into the articulation's actuated joint list (same space
        as :attr:`Articulation.joint_names`). ``target`` has shape ``(num_envs,
        len(local_joint_ids))``. For every actuator group: implicit-PD actuators receive the
        slice via ``control_dofs_position``; force-channel actuators get their effort from
        :meth:`ActuatorBase.compute` and the result is pushed via ``control_dofs_force``.
        """
        if local_joint_ids.numel() == 0:
            return
        self._joint_pos_target.index_copy_(1, local_joint_ids, target)
        robot = self._gs_handle
        if robot is None:
            return

        # The RobotState cache is only refreshed once per env step, but this method runs
        # once per sim substep inside ``ManagerBasedRlEnv.step``. Reading the stale cache
        # here feeds the force-channel PD law state up to ``decimation * dt_sim`` old, which
        # zeros closed-loop damping inside the substep window and makes the joints visibly
        # oscillate. Pull fresh DoF state directly from sim for the force-channel branch.
        fresh_pos: torch.Tensor | None = None
        fresh_vel: torch.Tensor | None = None
        if any(a.channel != "implicit_pd" for a in self._actuators.values()):
            fresh_pos = to_tensor(robot.get_dofs_position(), self._device)
            fresh_vel = to_tensor(robot.get_dofs_velocity(), self._device)

        for actuator in self._actuators.values():
            jids = actuator.joint_ids
            dofs = actuator.dof_ids
            target_slice = self._joint_pos_target.index_select(1, jids)
            if actuator.channel == "implicit_pd":
                ctrl = getattr(robot, "control_dofs_position", None) or getattr(
                    robot, "set_dofs_position_target", None
                )
                if ctrl is None:
                    continue
                try:
                    ctrl(target_slice, dofs)
                except TypeError:
                    ctrl(target_slice)
            else:
                assert fresh_pos is not None and fresh_vel is not None
                joint_pos = fresh_pos.index_select(-1, dofs)
                joint_vel = fresh_vel.index_select(-1, dofs)
                effort = actuator.compute(joint_pos, joint_vel, target_slice)
                if effort is None:
                    continue
                effort = actuator.apply_deadzone(effort)
                ctrl = getattr(robot, "control_dofs_force", None)
                if ctrl is None:
                    continue
                try:
                    ctrl(effort, dofs)
                except TypeError:
                    ctrl(effort)
