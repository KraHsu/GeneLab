"""Reusable termination term functions."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

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
