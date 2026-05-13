"""Reusable reward term functions for locomotion tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from genelab.envs.manager_based_rl_env import ManagerBasedRlEnv


def track_linear_velocity_xy_exp(
    env: "ManagerBasedRlEnv", command_name: str, std: float = 0.5
) -> torch.Tensor:
    """``exp(-||cmd_xy - vel_xy||^2 / std^2)``."""
    cmd = env.command_manager.get_command(command_name)[:, :2]
    vel = env.robot_state.root_lin_vel_b[:, :2]
    err = torch.sum((cmd - vel) ** 2, dim=-1)
    return torch.exp(-err / (std ** 2))


def track_angular_velocity_z_exp(
    env: "ManagerBasedRlEnv", command_name: str, std: float = 0.5
) -> torch.Tensor:
    cmd = env.command_manager.get_command(command_name)[:, 2]
    vel = env.robot_state.root_ang_vel_b[:, 2]
    err = (cmd - vel) ** 2
    return torch.exp(-err / (std ** 2))


def action_rate_l2(env: "ManagerBasedRlEnv") -> torch.Tensor:
    return torch.sum(
        (env.action_manager.action - env.action_manager.prev_action) ** 2, dim=-1
    )


def joint_acc_l2(env: "ManagerBasedRlEnv") -> torch.Tensor:
    # Approximate joint acceleration as joint_vel change per step (best-effort).
    vel = env.robot_state.joint_vel
    return torch.sum(vel ** 2, dim=-1) * 0.0  # placeholder — proper accel would need history


def flat_orientation_l2(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Penalise tilt: the xy components of body-frame gravity should be zero."""
    return torch.sum(env.robot_state.projected_gravity_b[:, :2] ** 2, dim=-1)


def joint_pos_limits(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """L2 of joint-position excursion past ±π (cheap stand-in for true limit penalty)."""
    excess = (env.robot_state.joint_pos.abs() - 3.14).clamp(min=0.0)
    return torch.sum(excess ** 2, dim=-1)


def feet_air_time(
    env: "ManagerBasedRlEnv",
    threshold: float = 0.4,
) -> torch.Tensor:
    """Stub: reward proportional to mean foot-link height above ground.

    A faithful air-time reward needs contact sensors; until those are wired into the env we
    approximate with the lowest foot z. ``threshold`` controls the height where reward saturates.
    """
    foot_names = env.cfg.robot.foot_link_names
    if not foot_names or not env.link_names:
        return torch.zeros(env.num_envs, device=env.device)
    indices = [env.link_names.index(n) for n in foot_names if n in env.link_names]
    if not indices:
        return torch.zeros(env.num_envs, device=env.device)
    foot_z = env.robot_state.link_pos[:, indices, 2]
    height = foot_z.mean(dim=-1).clamp(0.0, threshold)
    return height / threshold
