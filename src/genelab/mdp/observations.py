"""Reusable observation term functions (body-frame velocities, joint state, commands)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

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
