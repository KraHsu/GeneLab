"""Reusable termination term functions."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import torch

from genelab.mdp.commands.motion_command import MotionCommand

if TYPE_CHECKING:
    from genelab.envs.manager_based_rl_env import ManagerBasedRlEnv


def time_out(env: "ManagerBasedRlEnv") -> torch.Tensor:
    return env.episode_length_buf >= env.max_episode_length


def bad_orientation(
    env: "ManagerBasedRlEnv",
    limit_angle: float = math.radians(70.0),
) -> torch.Tensor:
    """True when the body z-axis tilts more than ``limit_angle`` from world up.

    Body-frame projected gravity z = -cos(tilt). So |projected_gravity_b.z| < cos(limit_angle).
    """
    cos_limit = math.cos(limit_angle)
    gravity_z = env.robot_state.projected_gravity_b[:, 2]
    return gravity_z > -cos_limit


def root_height_below(env: "ManagerBasedRlEnv", min_height: float) -> torch.Tensor:
    return env.robot_state.root_pos[:, 2] < min_height


# --------------------------------------------------------------------- motion imitation

def _motion_command(env: "ManagerBasedRlEnv", command_name: str) -> MotionCommand:
    term = env.command_manager._terms[command_name]  # pyright: ignore[reportPrivateUsage]
    return cast(MotionCommand, term)


def bad_anchor_pos_z_only(
    env: "ManagerBasedRlEnv", command_name: str, threshold: float
) -> torch.Tensor:
    """True when the robot anchor z drifts further than ``threshold`` from the reference clip."""
    cmd = _motion_command(env, command_name)
    return torch.abs(cmd.anchor_pos_w[:, -1] - cmd.robot_anchor_pos_w[:, -1]) > threshold


def bad_anchor_ori(
    env: "ManagerBasedRlEnv", command_name: str, threshold: float
) -> torch.Tensor:
    """True when the tilt error between robot anchor and reference exceeds ``threshold``.

    Uses the body-frame gravity z-component as a tilt proxy (simpler than a full geodesic
    distance and matches mjlab's ``bad_anchor_ori`` semantically).
    """
    from genelab.utils.math import quat_apply_inverse

    cmd = _motion_command(env, command_name)
    gravity_w = torch.tensor([0.0, 0.0, -1.0], device=env.device).expand(env.num_envs, 3)
    motion_g_b = quat_apply_inverse(cmd.anchor_quat_w, gravity_w)
    robot_g_b = quat_apply_inverse(cmd.robot_anchor_quat_w, gravity_w)
    return (motion_g_b[:, 2] - robot_g_b[:, 2]).abs() > threshold


def bad_motion_body_pos_z_only(
    env: "ManagerBasedRlEnv",
    command_name: str,
    threshold: float,
    body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
    """True when any selected body's vertical position deviates past ``threshold``."""
    cmd = _motion_command(env, command_name)
    if body_names is None:
        indexes = list(range(len(cmd.cfg.body_names)))
    else:
        indexes = [i for i, n in enumerate(cmd.cfg.body_names) if n in body_names]
    if not indexes:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    error = torch.abs(
        cmd.body_pos_relative_w[:, indexes, -1] - cmd.robot_body_pos_w[:, indexes, -1]
    )
    return torch.any(error > threshold, dim=-1)
