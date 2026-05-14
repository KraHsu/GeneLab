"""Reusable event term functions (resets, periodic disturbances)."""

from typing import TYPE_CHECKING

import torch

from genelab.utils.math import quat_from_euler_xyz, quat_mul

if TYPE_CHECKING:
    from genelab.envs.manager_based_rl_env import ManagerBasedRlEnv


def reset_root_state_uniform(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor | None,
    pose_range: dict[str, tuple[float, float]] | None = None,
    velocity_range: dict[str, tuple[float, float]] | None = None,
) -> None:
    """Randomize the floating base pose / velocity within the given ranges.

    ``pose_range`` accepts ``x``, ``y``, ``z`` (position offsets, m) and ``roll``, ``pitch``,
    ``yaw`` (rotation offsets, rad, layered on top of the configured ``init_quat``).
    ``velocity_range`` accepts ``x``, ``y``, ``z`` (linear, m/s) and ``roll``, ``pitch``,
    ``yaw`` (angular, rad/s).
    """
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
    if any(axis in pose_range for axis in ("roll", "pitch", "yaw")):
        base_quat = torch.tensor(env.cfg.robot.init_quat, device=env.device).expand(n, -1).clone()
        zeros = torch.zeros(n, device=env.device)
        rpy = []
        for axis in ("roll", "pitch", "yaw"):
            if axis in pose_range:
                lo, hi = pose_range[axis]
                rpy.append(torch.empty(n, device=env.device).uniform_(lo, hi))
            else:
                rpy.append(zeros)
        rand_quat = quat_from_euler_xyz(rpy[0], rpy[1], rpy[2])
        new_quat = quat_mul(base_quat, rand_quat)
        set_quat = getattr(env.robot, "set_quat", None)
        if set_quat is not None:
            try:
                set_quat(new_quat, envs_idx=env_ids)
            except TypeError:
                set_quat(new_quat)
    vel = torch.zeros(n, 3, device=env.device)
    ang = torch.zeros(n, 3, device=env.device)
    for axis, idx in (("x", 0), ("y", 1), ("z", 2)):
        if axis in velocity_range:
            lo, hi = velocity_range[axis]
            vel[:, idx] = torch.empty(n, device=env.device).uniform_(lo, hi)
    for axis, idx in (("roll", 0), ("pitch", 1), ("yaw", 2)):
        if axis in velocity_range:
            lo, hi = velocity_range[axis]
            ang[:, idx] = torch.empty(n, device=env.device).uniform_(lo, hi)
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
    env.articulation.write_joint_state(pos, vel, env_ids)


def push_by_setting_velocity(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor | None,
    velocity_range: dict[str, tuple[float, float]] | None = None,
) -> None:
    """Push selected envs by overwriting linear (x/y/z) and angular (roll/pitch/yaw) base velocity."""
    if env_ids is None or env_ids.numel() == 0:
        return
    velocity_range = velocity_range or {}
    n = int(env_ids.numel())
    vel = torch.zeros(n, 3, device=env.device)
    ang = torch.zeros(n, 3, device=env.device)
    for axis, idx in (("x", 0), ("y", 1), ("z", 2)):
        if axis in velocity_range:
            lo, hi = velocity_range[axis]
            vel[:, idx] = torch.empty(n, device=env.device).uniform_(lo, hi)
    for axis, idx in (("roll", 0), ("pitch", 1), ("yaw", 2)):
        if axis in velocity_range:
            lo, hi = velocity_range[axis]
            ang[:, idx] = torch.empty(n, device=env.device).uniform_(lo, hi)
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
