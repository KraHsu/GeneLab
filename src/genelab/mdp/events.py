"""Reusable event term functions (resets, periodic disturbances)."""

from typing import TYPE_CHECKING

import torch

from genelab.mdp._helpers import asset_articulation, asset_handle
from genelab.utils.math import quat_from_euler_xyz, quat_mul

if TYPE_CHECKING:
    from genelab.envs.manager_based_rl_env import ManagerBasedRlEnv
    from genelab.managers.scene_entity_cfg import SceneEntityCfg


def reset_root_state_uniform(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor | None,
    pose_range: dict[str, tuple[float, float]] | None = None,
    velocity_range: dict[str, tuple[float, float]] | None = None,
    asset_cfg: "SceneEntityCfg | None" = None,
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
    handle = asset_handle(env, asset_cfg)
    entity_cfg = asset_articulation(env, asset_cfg).cfg
    init_pos = torch.tensor(entity_cfg.init_pos, device=env.device).expand(n, -1).clone()
    for axis, idx in (("x", 0), ("y", 1), ("z", 2)):
        if axis in pose_range:
            lo, hi = pose_range[axis]
            init_pos[:, idx] += torch.empty(n, device=env.device).uniform_(lo, hi)
    set_pos = getattr(handle, "set_pos", None)
    if set_pos is not None:
        try:
            set_pos(init_pos, envs_idx=env_ids)
        except TypeError:
            set_pos(init_pos)
    if any(axis in pose_range for axis in ("roll", "pitch", "yaw")):
        base_quat = torch.tensor(entity_cfg.init_quat, device=env.device).expand(n, -1).clone()
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
        set_quat = getattr(handle, "set_quat", None)
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
    set_vel = getattr(handle, "set_vel", None)
    set_ang = getattr(handle, "set_ang", None)
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
    asset_cfg: "SceneEntityCfg | None" = None,
) -> None:
    if env_ids is None or env_ids.numel() == 0:
        return
    n = int(env_ids.numel())
    articulation = asset_articulation(env, asset_cfg)
    pos = articulation.default_joint_pos.unsqueeze(0).expand(n, -1).clone()
    if pos_jitter > 0:
        pos += torch.empty_like(pos).uniform_(-pos_jitter, pos_jitter)
    vel = torch.zeros_like(pos)
    if vel_jitter > 0:
        vel += torch.empty_like(vel).uniform_(-vel_jitter, vel_jitter)
    articulation.write_joint_state(pos, vel, env_ids)


def reset_joints_by_offset(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor | None,
    position_range: tuple[float, float] = (0.0, 0.0),
    velocity_range: tuple[float, float] = (0.0, 0.0),
    asset_cfg: "SceneEntityCfg | None" = None,
) -> None:
    """Reset joints to ``default + U(*position_range)`` with ``U(*velocity_range)`` velocity.

    mjlab parity for ``envs/mdp/events.reset_joints_by_offset``. Differs from
    :func:`reset_joints_to_default`:

    * Accepts asymmetric ``(lo, hi)`` ranges rather than a symmetric jitter scalar.
    * Same range applies uniformly to all joints (mjlab supports per-joint via
      ``SceneEntityCfg``; once :class:`SceneEntityCfg` lands in P5 we can extend here).
    * Clamps the sampled position to ``env.articulation.joint_pos_limits`` so
      large offsets can't violate the joint limits — mirrors mjlab's
      ``soft_joint_pos_limits`` clamp.

    ``position_range=(0, 0)`` and ``velocity_range=(0, 0)`` reproduces the mjlab call
    site that resets to bare default pose with zero velocity.
    """
    if env_ids is None or env_ids.numel() == 0:
        return
    n = int(env_ids.numel())
    articulation = asset_articulation(env, asset_cfg)
    pos = articulation.default_joint_pos.unsqueeze(0).expand(n, -1).clone()
    lo, hi = position_range
    if lo != 0.0 or hi != 0.0:
        pos += torch.empty_like(pos).uniform_(lo, hi)
        limits = articulation.joint_pos_limits
        if limits.numel() > 0 and limits.shape[0] == pos.shape[1]:
            # Only clamp when finite (Genesis returns ±inf for floating-base DoFs
            # but those are stripped from joint_pos_limits at bind time).
            pos = pos.clamp(min=limits[:, 0], max=limits[:, 1])
    vel = torch.zeros_like(pos)
    lo, hi = velocity_range
    if lo != 0.0 or hi != 0.0:
        vel += torch.empty_like(vel).uniform_(lo, hi)
    articulation.write_joint_state(pos, vel, env_ids)


def push_by_setting_velocity(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor | None,
    velocity_range: dict[str, tuple[float, float]] | None = None,
    asset_cfg: "SceneEntityCfg | None" = None,
) -> None:
    """Push selected envs by overwriting linear (x/y/z) and angular (roll/pitch/yaw) base velocity."""
    if env_ids is None or env_ids.numel() == 0:
        return
    velocity_range = velocity_range or {}
    n = int(env_ids.numel())
    handle = asset_handle(env, asset_cfg)
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
    set_vel = getattr(handle, "set_vel", None)
    set_ang = getattr(handle, "set_ang", None)
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
