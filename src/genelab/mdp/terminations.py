"""Reusable termination term functions."""

import math
from typing import TYPE_CHECKING, cast

import torch

from genelab.mdp._helpers import (
    asset_articulation as _asset_articulation,
    asset_state as _asset_state,
    contact_sensor as _contact_sensor,
)
from genelab.mdp.commands.motion_command import MotionCommand

if TYPE_CHECKING:
    from genelab.contracts import EnvContext
    from genelab.managers.scene_entity_cfg import SceneEntityCfg


def time_out(env: "EnvContext") -> torch.Tensor:
    return env.episode_length_buf >= env.max_episode_length


def bad_orientation(
    env: "EnvContext",
    limit_angle: float = math.radians(70.0),
    asset_cfg: "SceneEntityCfg | None" = None,
) -> torch.Tensor:
    """True when the body z-axis tilts more than ``limit_angle`` from world up.

    Body-frame projected gravity z = -cos(tilt). So |projected_gravity_b.z| < cos(limit_angle).
    """
    cos_limit = math.cos(limit_angle)
    gravity_z = _asset_state(env, asset_cfg).projected_gravity_b[:, 2]
    return gravity_z > -cos_limit


def root_height_below(
    env: "EnvContext", min_height: float, asset_cfg: "SceneEntityCfg | None" = None
) -> torch.Tensor:
    return _asset_state(env, asset_cfg).root_pos[:, 2] < min_height


def joint_pos_out_of_limit(
    env: "EnvContext", asset_cfg: "SceneEntityCfg | None" = None
) -> torch.Tensor:
    """True for any env where an actuated joint left its position limits.

    Reads the same per-joint ``(lower, upper)`` limits as the ``joint_pos_limits``
    reward (``env.joint_pos_limits``, sliced to actuated DoFs at bind time). Joints
    with ±∞ limits (continuous joints) never trip it. Use as a safety termination so
    a policy can't learn to exploit a joint slammed against its hard stop.
    """
    limits = _asset_articulation(env, asset_cfg).joint_pos_limits  # (J, 2)
    joint_pos = _asset_state(env, asset_cfg).joint_pos  # (B, J)
    below = joint_pos < limits[:, 0].unsqueeze(0)
    above = joint_pos > limits[:, 1].unsqueeze(0)
    return torch.any(below | above, dim=-1)


def joint_vel_out_of_limit(
    env: "EnvContext", asset_cfg: "SceneEntityCfg | None" = None
) -> torch.Tensor:
    """True for any env where an actuated joint exceeds its velocity-limit magnitude.

    Reads ``env.joint_vel_limits`` (``ArticulationCfg.joint_vel_limit``, ``+∞`` when
    unset → never trips) and ``robot_state.joint_vel``. A safety termination for
    runaway joint speeds; inert until a task declares a velocity limit.
    """
    limit = _asset_articulation(env, asset_cfg).joint_vel_limits  # (J,)
    speed = torch.abs(_asset_state(env, asset_cfg).joint_vel)  # (B, J)
    return torch.any(speed > limit.unsqueeze(0), dim=-1)


def contact_force_limit(env: "EnvContext", sensor_name: str, max_force: float) -> torch.Tensor:
    """True for any env where a tracked link's net contact-force magnitude exceeds ``max_force``.

    Reads ``force_norm`` ``(B, N)`` from the named :class:`~genelab.sensor.ContactSensor`
    (``N`` = the links it tracks). A safety termination for impact spikes — e.g. a
    base or knee slamming the ground harder than the real hardware could survive.
    """
    force_norm = _contact_sensor(env, sensor_name).data.force_norm  # (B, N)
    return torch.any(force_norm > max_force, dim=-1)


# --------------------------------------------------------------------- motion imitation


def _motion_command(env: "EnvContext", command_name: str) -> MotionCommand:
    term = env.command_manager._terms[command_name]  # pyright: ignore[reportPrivateUsage]
    return cast(MotionCommand, term)


def bad_anchor_pos_z_only(env: "EnvContext", command_name: str, threshold: float) -> torch.Tensor:
    """True when the robot anchor z drifts further than ``threshold`` from the reference clip."""
    cmd = _motion_command(env, command_name)
    return torch.abs(cmd.anchor_pos_w[:, -1] - cmd.robot_anchor_pos_w[:, -1]) > threshold


def bad_anchor_ori(env: "EnvContext", command_name: str, threshold: float) -> torch.Tensor:
    """True when the tilt error between robot anchor and reference exceeds ``threshold``.

    Uses the body-frame gravity z-component as a tilt proxy (simpler than a full geodesic
    distance and matches the reference's ``bad_anchor_ori`` semantically).
    """
    from genelab.utils.math import quat_apply_inverse

    cmd = _motion_command(env, command_name)
    gravity_w = torch.tensor([0.0, 0.0, -1.0], device=env.device).expand(env.num_envs, 3)
    motion_g_b = quat_apply_inverse(cmd.anchor_quat_w, gravity_w)
    robot_g_b = quat_apply_inverse(cmd.robot_anchor_quat_w, gravity_w)
    return (motion_g_b[:, 2] - robot_g_b[:, 2]).abs() > threshold


def bad_motion_body_pos_z_only(
    env: "EnvContext",
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
