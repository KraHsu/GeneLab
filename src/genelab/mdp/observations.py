"""Reusable observation term functions (body-frame velocities, joint state, commands)."""

from typing import TYPE_CHECKING, cast

import torch

from genelab.mdp._helpers import asset_articulation, asset_state
from genelab.mdp.commands.motion_command import MotionCommand
from genelab.sensor.contact import ContactSensor
from genelab.sensor.force_torque import ForceTorqueSensor
from genelab.utils.math import matrix_from_quat, subtract_frame_transforms

if TYPE_CHECKING:
    from genelab.envs.manager_based_rl_env import ManagerBasedRlEnv
    from genelab.managers.scene_entity_cfg import SceneEntityCfg


def base_lin_vel(
    env: "ManagerBasedRlEnv", asset_cfg: "SceneEntityCfg | None" = None
) -> torch.Tensor:
    """Body-frame linear velocity of the floating base."""
    return asset_state(env, asset_cfg).root_lin_vel_b


def base_ang_vel(
    env: "ManagerBasedRlEnv", asset_cfg: "SceneEntityCfg | None" = None
) -> torch.Tensor:
    """Body-frame angular velocity of the floating base."""
    return asset_state(env, asset_cfg).root_ang_vel_b


def projected_gravity(
    env: "ManagerBasedRlEnv", asset_cfg: "SceneEntityCfg | None" = None
) -> torch.Tensor:
    """Gravity vector projected into the body frame (proxy for IMU orientation)."""
    return asset_state(env, asset_cfg).projected_gravity_b


def joint_pos_rel(
    env: "ManagerBasedRlEnv", asset_cfg: "SceneEntityCfg | None" = None
) -> torch.Tensor:
    """Joint positions minus default pose.

    Mirrors mjlab's default ``joint_pos_rel`` (no bias term). When the env's
    startup events include ``genelab.mdp.dr.encoder_bias``,
    :class:`~genelab.mdp.actions.joint_position.JointPositionAction` subtracts
    the bias from its PD target so the real joint sits ``-bias`` away from the
    commanded reference. The obs then surfaces that physical offset directly,
    which is what the policy must learn to compensate for. Adding the bias here
    too would cancel the perturbation in the observation and silently neutralise
    the encoder-bias DR.
    """
    return (
        asset_state(env, asset_cfg).joint_pos - asset_articulation(env, asset_cfg).default_joint_pos
    )


def joint_vel_rel(
    env: "ManagerBasedRlEnv", asset_cfg: "SceneEntityCfg | None" = None
) -> torch.Tensor:
    """Joint velocities (default is zero, so just the raw vel)."""
    return asset_state(env, asset_cfg).joint_vel


def last_action(env: "ManagerBasedRlEnv") -> torch.Tensor:
    return env.action_manager.action


def generated_commands(env: "ManagerBasedRlEnv", command_name: str) -> torch.Tensor:
    return env.command_manager.get_command(command_name)


def sensor_data(env: "ManagerBasedRlEnv", sensor_name: str) -> torch.Tensor:
    """Return the per-step cached tensor of the named sensor."""
    return env.sensors[sensor_name].data


def _contact_sensor(env: "ManagerBasedRlEnv", sensor_name: str) -> ContactSensor:
    sensor = env.sensors[sensor_name]
    if not isinstance(sensor, ContactSensor):
        raise TypeError(
            f"sensor {sensor_name!r} is not a ContactSensor (got {type(sensor).__name__})"
        )
    return sensor


def foot_air_time(env: "ManagerBasedRlEnv", sensor_name: str) -> torch.Tensor:
    """Current air time per foot (zero while in contact)."""
    return _contact_sensor(env, sensor_name).data.current_air_time


def foot_contact(env: "ManagerBasedRlEnv", sensor_name: str) -> torch.Tensor:
    """Binary contact mask per foot as a float tensor."""
    return _contact_sensor(env, sensor_name).data.found.float()


def foot_contact_forces(env: "ManagerBasedRlEnv", sensor_name: str) -> torch.Tensor:
    """Per-foot contact force, compressed via ``sign(f) * log1p(|f|)`` and flattened to ``(B, N*3)``."""
    force = _contact_sensor(env, sensor_name).data.force
    return (force.sign() * torch.log1p(force.abs())).reshape(force.shape[0], -1)


def joint_force_torque(env: "ManagerBasedRlEnv", sensor_name: str) -> torch.Tensor:
    """Per-joint reaction force/torque from a ``ForceTorqueSensor``, shape ``(B, num_joints)``."""
    sensor = env.sensors[sensor_name]
    if not isinstance(sensor, ForceTorqueSensor):
        raise TypeError(
            f"sensor {sensor_name!r} is not a ForceTorqueSensor (got {type(sensor).__name__})"
        )
    return sensor.data.force


def height_scan(env: "ManagerBasedRlEnv", sensor_name: str) -> torch.Tensor:
    """Per-ray heights from a ``TerrainHeightSensor`` (positive = above terrain)."""
    out = env.sensors[sensor_name].data
    if not isinstance(out, torch.Tensor):
        raise TypeError(
            f"sensor {sensor_name!r} does not return a height tensor (got {type(out).__name__})"
        )
    return out


# --------------------------------------------------------------------- motion imitation


def _motion_command(env: "ManagerBasedRlEnv", command_name: str) -> MotionCommand:
    term = env.command_manager._terms[command_name]  # pyright: ignore[reportPrivateUsage]
    return cast(MotionCommand, term)


def motion_anchor_pos_b(env: "ManagerBasedRlEnv", command_name: str) -> torch.Tensor:
    """Anchor-frame reference position expressed in the robot's anchor frame."""
    cmd = _motion_command(env, command_name)
    pos, _ = subtract_frame_transforms(
        cmd.robot_anchor_pos_w,
        cmd.robot_anchor_quat_w,
        cmd.anchor_pos_w,
        cmd.anchor_quat_w,
    )
    return pos.view(env.num_envs, -1)


def motion_anchor_ori_b(env: "ManagerBasedRlEnv", command_name: str) -> torch.Tensor:
    """6D anchor-frame orientation (first two columns of the rotation matrix)."""
    cmd = _motion_command(env, command_name)
    _, ori = subtract_frame_transforms(
        cmd.robot_anchor_pos_w,
        cmd.robot_anchor_quat_w,
        cmd.anchor_pos_w,
        cmd.anchor_quat_w,
    )
    mat = matrix_from_quat(ori)
    return mat[..., :2].reshape(mat.shape[0], -1)


def robot_body_pos_b(env: "ManagerBasedRlEnv", command_name: str) -> torch.Tensor:
    """Per-body positions in the robot's anchor frame (privileged critic obs)."""
    cmd = _motion_command(env, command_name)
    num_bodies = len(cmd.cfg.body_names)
    pos_b, _ = subtract_frame_transforms(
        cmd.robot_anchor_pos_w[:, None, :].expand(-1, num_bodies, -1),
        cmd.robot_anchor_quat_w[:, None, :].expand(-1, num_bodies, -1),
        cmd.robot_body_pos_w,
        cmd.robot_body_quat_w,
    )
    return pos_b.reshape(env.num_envs, -1)


def robot_body_ori_b(env: "ManagerBasedRlEnv", command_name: str) -> torch.Tensor:
    """Per-body 6D orientations in the robot's anchor frame."""
    cmd = _motion_command(env, command_name)
    num_bodies = len(cmd.cfg.body_names)
    _, ori_b = subtract_frame_transforms(
        cmd.robot_anchor_pos_w[:, None, :].expand(-1, num_bodies, -1),
        cmd.robot_anchor_quat_w[:, None, :].expand(-1, num_bodies, -1),
        cmd.robot_body_pos_w,
        cmd.robot_body_quat_w,
    )
    mat = matrix_from_quat(ori_b)
    return mat[..., :2].reshape(mat.shape[0], -1)
