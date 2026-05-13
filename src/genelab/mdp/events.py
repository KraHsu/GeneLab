"""Reusable event term functions (resets, periodic disturbances)."""

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from genelab.envs.manager_based_rl_env import ManagerBasedRlEnv


def reset_root_state_uniform(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor | None,
    pose_range: dict[str, tuple[float, float]] | None = None,
    velocity_range: dict[str, tuple[float, float]] | None = None,
) -> None:
    """Randomize the floating base pose / velocity within the given ranges."""
    if env_ids is None or env_ids.numel() == 0:
        return
    pose_range = pose_range or {}
    velocity_range = velocity_range or {}
    n = int(env_ids.numel())
    init_pos = torch.tensor(env.cfg.robot.init_pos, device=env.device).expand(n, -1).clone()
    for axis, idx in (("x", 0), ("y", 1), ("z", 2)):
        if axis in pose_range:
            lo, hi = pose_range[axis]
            init_pos[:, idx] += torch.empty(n, device=env.device).uniform_(lo, hi)
    set_pos = getattr(env.robot, "set_pos", None)
    if set_pos is not None:
        try:
            set_pos(init_pos, envs_idx=env_ids)
        except TypeError:
            set_pos(init_pos)
    vel = torch.zeros(n, 3, device=env.device)
    ang = torch.zeros(n, 3, device=env.device)
    for axis, idx in (("x", 0), ("y", 1), ("z", 2)):
        if axis in velocity_range:
            lo, hi = velocity_range[axis]
            vel[:, idx] = torch.empty(n, device=env.device).uniform_(lo, hi)
    set_vel = getattr(env.robot, "set_vel", None)
    set_ang = getattr(env.robot, "set_ang", None)
    if set_vel is not None:
        try:
            set_vel(vel, envs_idx=env_ids)
        except TypeError:
            set_vel(vel)
    if set_ang is not None:
        try:
            set_ang(ang, envs_idx=env_ids)
        except TypeError:
            set_ang(ang)


def reset_joints_to_default(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor | None,
    pos_jitter: float = 0.0,
    vel_jitter: float = 0.0,
) -> None:
    if env_ids is None or env_ids.numel() == 0:
        return
    n = int(env_ids.numel())
    pos = env.default_joint_pos.unsqueeze(0).expand(n, -1).clone()
    if pos_jitter > 0:
        pos += torch.empty_like(pos).uniform_(-pos_jitter, pos_jitter)
    vel = torch.zeros_like(pos)
    if vel_jitter > 0:
        vel += torch.empty_like(vel).uniform_(-vel_jitter, vel_jitter)
    actuated_idx = getattr(env, "_actuated_dof_idx", None)
    set_pos = getattr(env.robot, "set_dofs_position", None)
    set_vel = getattr(env.robot, "set_dofs_velocity", None)
    if set_pos is not None:
        try:
            set_pos(pos, actuated_idx, envs_idx=env_ids)
        except TypeError:
            set_pos(pos, actuated_idx)
    if set_vel is not None:
        try:
            set_vel(vel, actuated_idx, envs_idx=env_ids)
        except TypeError:
            set_vel(vel, actuated_idx)


def push_by_setting_velocity(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor | None,
    velocity_range: dict[str, tuple[float, float]] | None = None,
) -> None:
    if env_ids is None or env_ids.numel() == 0:
        return
    velocity_range = velocity_range or {}
    n = int(env_ids.numel())
    vel = torch.zeros(n, 3, device=env.device)
    for axis, idx in (("x", 0), ("y", 1), ("z", 2)):
        if axis in velocity_range:
            lo, hi = velocity_range[axis]
            vel[:, idx] = torch.empty(n, device=env.device).uniform_(lo, hi)
    set_vel = getattr(env.robot, "set_vel", None)
    if set_vel is not None:
        try:
            set_vel(vel, envs_idx=env_ids)
        except TypeError:
            set_vel(vel)
