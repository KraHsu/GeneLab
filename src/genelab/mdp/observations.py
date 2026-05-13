"""Reusable observation term functions (body-frame velocities, joint state, commands)."""

from typing import TYPE_CHECKING, cast

import torch

from genelab.mdp.commands.motion_command import MotionCommand
from genelab.utils.math import matrix_from_quat, subtract_frame_transforms

if TYPE_CHECKING:
    from genelab.envs.manager_based_rl_env import ManagerBasedRlEnv


def base_lin_vel(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Body-frame linear velocity of the floating base."""
    return env.robot_state.root_lin_vel_b


def base_ang_vel(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Body-frame angular velocity of the floating base."""
    return env.robot_state.root_ang_vel_b


def projected_gravity(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Gravity vector projected into the body frame (proxy for IMU orientation)."""
    return env.robot_state.projected_gravity_b


def joint_pos_rel(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Joint positions minus default pose."""
    return env.robot_state.joint_pos - env.default_joint_pos


def joint_vel_rel(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Joint velocities (default is zero, so just the raw vel)."""
    return env.robot_state.joint_vel


def last_action(env: "ManagerBasedRlEnv") -> torch.Tensor:
    return env.action_manager.action


def generated_commands(env: "ManagerBasedRlEnv", command_name: str) -> torch.Tensor:
    return env.command_manager.get_command(command_name)


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
