"""Per-step articulation state — the read seam of ``Articulation`` (ADR-0019).

``RobotState`` is the cached buffer MDP / sensor code reads through the env; it is
re-exported from :mod:`genelab.entity.articulation` (and ``genelab.entity``) so the
public import path is unchanged. ``ArticulationState`` owns that buffer and the
per-step :meth:`refresh` that fills it from the Genesis handle; ``Articulation``
composes it and delegates ``refresh`` / ``data`` to it.
"""

from typing import Any

import torch

from genelab.entity._torch import quat_rotate_inverse, to_tensor


class RobotState:
    """Cached per-step articulation state. Refreshed by ``Articulation.refresh``."""

    def __init__(
        self,
        num_envs: int,
        num_dofs: int,
        num_links: int,
        device: str,
    ) -> None:
        def z(*shape: int) -> torch.Tensor:
            return torch.zeros(*shape, device=device)

        self.root_pos = z(num_envs, 3)
        self.root_quat = z(num_envs, 4)
        self.root_quat[:, 0] = 1.0
        self.root_lin_vel_w = z(num_envs, 3)
        self.root_ang_vel_w = z(num_envs, 3)
        self.root_lin_vel_b = z(num_envs, 3)
        self.root_ang_vel_b = z(num_envs, 3)
        self.projected_gravity_b = z(num_envs, 3)
        self.projected_gravity_b[:, 2] = -1.0
        self.joint_pos = z(num_envs, num_dofs)
        self.joint_vel = z(num_envs, num_dofs)
        self.link_pos = z(num_envs, num_links, 3)
        self.link_quat_w = z(num_envs, num_links, 4)
        self.link_quat_w[..., 0] = 1.0
        self.link_lin_vel_w = z(num_envs, num_links, 3)
        self.link_ang_vel_w = z(num_envs, num_links, 3)
        # Per-env, per-DoF encoder bias. Zero by default — populated by the
        # ``encoder_bias`` DR event (``genelab.mdp.dr.encoder_bias``). When non-zero,
        # ``JointPositionAction.process_actions`` subtracts it from the PD target
        # so the real joint sits ``bias`` away from the policy's nominal command;
        # ``mdp.joint_pos_rel`` returns the raw ``joint_pos − default`` and surfaces
        # that offset to the policy — mjlab parity for joint-encoder-bias sim2real DR.
        self.encoder_bias = z(num_envs, num_dofs)
        # Per-env, per-actuated-DoF realized actuator torque (Genesis control force),
        # refreshed each step via ``get_dofs_control_force``. Used by
        # ``mdp.applied_torque_l2``. Zero on platforms / fake envs without the getter.
        self.applied_torque = z(num_envs, num_dofs)


class ArticulationState:
    """Owns the cached :class:`RobotState` and the per-step refresh that fills it.

    Behaviour-preserving extraction of ``Articulation.refresh`` and its state buffer
    (ADR-0019 read seam). Created at ``Articulation.bind`` once the joints / links and
    actuated DoF index are known.
    """

    def __init__(
        self,
        gs_handle: Any,
        *,
        num_envs: int,
        num_dofs: int,
        num_links: int,
        device: str,
        actuated_dof_idx: torch.Tensor,
    ) -> None:
        self._gs_handle = gs_handle
        self._device = device
        self._num_envs = num_envs
        self._actuated_dof_idx = actuated_dof_idx
        self._data = RobotState(num_envs, num_dofs, num_links, device)
        # Cached constant gravity vector, broadcast across envs. Built once here to keep
        # ``refresh`` allocation-free on the hot path (called every step + every reset).
        self._gravity_w = torch.zeros(num_envs, 3, device=device)
        self._gravity_w[:, 2] = -1.0

    @property
    def data(self) -> RobotState:
        return self._data

    def refresh(self) -> None:
        rs = self._data
        robot = self._gs_handle
        try:
            rs.root_pos.copy_(to_tensor(robot.get_pos(), self._device))
            rs.root_quat.copy_(to_tensor(robot.get_quat(), self._device))
            rs.root_lin_vel_w.copy_(to_tensor(robot.get_vel(), self._device))
            rs.root_ang_vel_w.copy_(to_tensor(robot.get_ang(), self._device))
            joint_pos_full = to_tensor(robot.get_dofs_position(), self._device)
            joint_vel_full = to_tensor(robot.get_dofs_velocity(), self._device)
            rs.joint_pos.copy_(joint_pos_full.index_select(-1, self._actuated_dof_idx))
            rs.joint_vel.copy_(joint_vel_full.index_select(-1, self._actuated_dof_idx))
        except AttributeError:
            pass
        # Realized actuator torque (Genesis control force), sliced to actuated DoFs.
        # Guarded separately so a missing getter (fake envs / older Genesis) leaves
        # ``applied_torque`` at zero without disturbing the joint-state read above.
        get_control_force = getattr(robot, "get_dofs_control_force", None)
        if get_control_force is not None:
            try:
                force_full = to_tensor(get_control_force(), self._device)
                rs.applied_torque.copy_(force_full.index_select(-1, self._actuated_dof_idx))
            except Exception:
                pass
        for attr, target in (
            ("get_links_pos", "link_pos"),
            ("get_links_quat", "link_quat_w"),
            ("get_links_vel", "link_lin_vel_w"),
            ("get_links_ang", "link_ang_vel_w"),
        ):
            getter = getattr(robot, attr, None)
            if getter is None:
                continue
            try:
                value = getter()
            except Exception:
                continue
            tensor = to_tensor(value, self._device)
            # Genesis may return either (num_envs, num_links, ...) or the bare (num_links, ...)
            # shape on older single-env builds; ``expand`` broadcasts without allocating.
            if tensor.dim() == 2:
                tensor = tensor.unsqueeze(0).expand(self._num_envs, -1, -1)
            getattr(rs, target).copy_(tensor)
        rs.root_lin_vel_b = quat_rotate_inverse(rs.root_quat, rs.root_lin_vel_w)
        rs.root_ang_vel_b = quat_rotate_inverse(rs.root_quat, rs.root_ang_vel_w)
        rs.projected_gravity_b = quat_rotate_inverse(rs.root_quat, self._gravity_w)
