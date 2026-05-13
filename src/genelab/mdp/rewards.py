"""Reusable reward term functions for locomotion tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from genelab.mdp.commands.motion_command import MotionCommand
from genelab.utils.math import quat_error_magnitude

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


# --------------------------------------------------------------------- motion imitation

def _motion_command(env: "ManagerBasedRlEnv", command_name: str) -> MotionCommand:
    term = env.command_manager._terms[command_name]  # pyright: ignore[reportPrivateUsage]
    return cast(MotionCommand, term)


def _body_index_filter(
    cmd: MotionCommand, body_names: tuple[str, ...] | None
) -> list[int]:
    return [
        i
        for i, name in enumerate(cmd.cfg.body_names)
        if (body_names is None) or (name in body_names)
    ]


def motion_global_anchor_position_error_exp(
    env: "ManagerBasedRlEnv", command_name: str, std: float
) -> torch.Tensor:
    """``exp(-||p_ref - p_robot||^2 / std^2)`` on the anchor body in world frame."""
    cmd = _motion_command(env, command_name)
    error = torch.sum((cmd.anchor_pos_w - cmd.robot_anchor_pos_w) ** 2, dim=-1)
    return torch.exp(-error / (std * std))


def motion_global_anchor_orientation_error_exp(
    env: "ManagerBasedRlEnv", command_name: str, std: float
) -> torch.Tensor:
    """Geodesic rotation error on the anchor body, mapped through a Gaussian kernel."""
    cmd = _motion_command(env, command_name)
    error = quat_error_magnitude(cmd.anchor_quat_w, cmd.robot_anchor_quat_w) ** 2
    return torch.exp(-error / (std * std))


def motion_relative_body_position_error_exp(
    env: "ManagerBasedRlEnv",
    command_name: str,
    std: float,
    body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
    """Multi-body L2 position error against the anchor-aligned reference frames."""
    cmd = _motion_command(env, command_name)
    indexes = _body_index_filter(cmd, body_names)
    error = torch.sum(
        (cmd.body_pos_relative_w[:, indexes] - cmd.robot_body_pos_w[:, indexes]) ** 2,
        dim=-1,
    )
    return torch.exp(-error.mean(dim=-1) / (std * std))


def motion_relative_body_orientation_error_exp(
    env: "ManagerBasedRlEnv",
    command_name: str,
    std: float,
    body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
    """Multi-body geodesic rotation error against the anchor-aligned reference frames."""
    cmd = _motion_command(env, command_name)
    indexes = _body_index_filter(cmd, body_names)
    error = (
        quat_error_magnitude(
            cmd.body_quat_relative_w[:, indexes],
            cmd.robot_body_quat_w[:, indexes],
        )
        ** 2
    )
    return torch.exp(-error.mean(dim=-1) / (std * std))


def motion_global_body_linear_velocity_error_exp(
    env: "ManagerBasedRlEnv",
    command_name: str,
    std: float,
    body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
    """L2 world-frame linear-velocity error across the tracked bodies."""
    cmd = _motion_command(env, command_name)
    indexes = _body_index_filter(cmd, body_names)
    error = torch.sum(
        (cmd.body_lin_vel_w[:, indexes] - cmd.robot_body_lin_vel_w[:, indexes]) ** 2,
        dim=-1,
    )
    return torch.exp(-error.mean(dim=-1) / (std * std))


def motion_global_body_angular_velocity_error_exp(
    env: "ManagerBasedRlEnv",
    command_name: str,
    std: float,
    body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
    """L2 world-frame angular-velocity error across the tracked bodies."""
    cmd = _motion_command(env, command_name)
    indexes = _body_index_filter(cmd, body_names)
    error = torch.sum(
        (cmd.body_ang_vel_w[:, indexes] - cmd.robot_body_ang_vel_w[:, indexes]) ** 2,
        dim=-1,
    )
    return torch.exp(-error.mean(dim=-1) / (std * std))
