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

    reference parity: assumes the commanded z-velocity is zero, so any non-zero
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

    reference parity: assumes the commanded xy angular velocities are zero, so any
    pitching/rolling rate contributes to the error term. Discourages tumbling
    alongside yaw-tracking.
    """
    cmd = env.command_manager.get_command(command_name)[:, 2]
    vel = _asset_state(env, asset_cfg).root_ang_vel_b
    z_err = (cmd - vel[:, 2]) ** 2
    xy_err = torch.sum(vel[:, :2] ** 2, dim=-1)
    return torch.exp(-(z_err + xy_err) / (std**2))


def velocity_tracking_error_l1(
    env: EnvContext,
    command_name: str,
    axes: tuple[int, ...] = (0, 1),
    asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Sum of ``|cmd[axis] − lin_vel_b[axis]|`` over the selected linear axes.

    Complement to the exp tracking kernel, which saturates at large error — once a
    policy fully abandons an axis (e.g. lateral velocity on a skid-steer platform), the
    exp gradient is ≈ 0 and nothing pulls it back. Used with a negative weight, this L1
    error keeps a constant-magnitude gradient at any distance from the command. ``axes``
    scopes the penalty to the abandoned axis so well-tracked axes aren't double-counted
    on top of the exp reward.
    """
    cmd = env.command_manager.get_command(command_name)
    vel = _asset_state(env, asset_cfg).root_lin_vel_b
    axes_t = torch.tensor(axes, dtype=torch.long, device=vel.device)
    err = cmd.index_select(-1, axes_t) - vel.index_select(-1, axes_t)
    return err.abs().sum(dim=-1)


def angular_velocity_tracking_error_l1(
    env: EnvContext,
    command_name: str,
    asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """``|cmd_wz − ang_vel_z|`` — linear-in-error yaw-tracking penalty.

    Angular counterpart of :func:`velocity_tracking_error_l1`, for the same failure mode:
    the exp yaw kernel's gradient vanishes once a policy fully abandons pure-rotation
    commands (it can still score on mixed commands, masking the collapse), so a
    constant-magnitude pull on the yaw error is needed to bring in-place rotation back.
    """
    cmd = env.command_manager.get_command(command_name)[:, 2]
    vel = _asset_state(env, asset_cfg).root_ang_vel_b
    return (cmd - vel[:, 2]).abs()
