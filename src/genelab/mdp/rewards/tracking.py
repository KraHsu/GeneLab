"""Command- and motion-tracking reward terms.

Velocity-command tracking (``track_*``) plus the global/relative motion-tracking
error terms, which are defined in :mod:`genelab.mdp.motion_tracking` and
re-exported here so they remain reachable as ``genelab.mdp.rewards.motion_*``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from genelab.mdp._helpers import asset_state as _asset_state
from genelab.mdp.motion_tracking import (
    motion_global_anchor_orientation_error_exp as motion_global_anchor_orientation_error_exp,
    motion_global_anchor_position_error_exp as motion_global_anchor_position_error_exp,
    motion_global_body_angular_velocity_error_exp as motion_global_body_angular_velocity_error_exp,
    motion_global_body_linear_velocity_error_exp as motion_global_body_linear_velocity_error_exp,
    motion_relative_body_orientation_error_exp as motion_relative_body_orientation_error_exp,
    motion_relative_body_position_error_exp as motion_relative_body_position_error_exp,
)

if TYPE_CHECKING:
    from genelab.contracts import EnvContext
    from genelab.managers.scene_entity_cfg import SceneEntityCfg


def track_linear_velocity_xy_exp(
    env: EnvContext,
    command_name: str,
    std: float = 0.5,
    asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """``exp(-(||cmd_xy - vel_xy||² + vel_z²) / std²)``.

    mjlab parity: assumes the commanded z-velocity is zero, so any non-zero
    vertical motion contributes to the tracking error. Discourages vertical
    bouncing alongside xy-tracking.
    """
    cmd = env.command_manager.get_command(command_name)[:, :2]
    vel = _asset_state(env, asset_cfg).root_lin_vel_b
    xy_err = torch.sum((cmd - vel[:, :2]) ** 2, dim=-1)
    z_err = vel[:, 2] ** 2
    return torch.exp(-(xy_err + z_err) / (std**2))


def track_angular_velocity_z_exp(
    env: EnvContext,
    command_name: str,
    std: float = 0.5,
    asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """``exp(-((cmd_z − vel_z)² + ||vel_xy||²) / std²)``.

    mjlab parity: assumes the commanded xy angular velocities are zero, so any
    pitching/rolling rate contributes to the error term. Discourages tumbling
    alongside yaw-tracking.
    """
    cmd = env.command_manager.get_command(command_name)[:, 2]
    vel = _asset_state(env, asset_cfg).root_ang_vel_b
    z_err = (cmd - vel[:, 2]) ** 2
    xy_err = torch.sum(vel[:, :2] ** 2, dim=-1)
    return torch.exp(-(z_err + xy_err) / (std**2))
