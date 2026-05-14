"""Cart-pole reward / termination / event terms reused by single and double tasks."""

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from genelab.envs.manager_based_rl_env import ManagerBasedRlEnv


_JOINT_INDEX_CACHE: dict[int, dict[str, int]] = {}


def _joint_index(env: "ManagerBasedRlEnv", joint_name: str) -> int:
    cache = _JOINT_INDEX_CACHE.setdefault(id(env), {})
    idx = cache.get(joint_name)
    if idx is None:
        idx = env.joint_names.index(joint_name)
        cache[joint_name] = idx
    return idx


def alive_bonus(env: "ManagerBasedRlEnv") -> torch.Tensor:
    return torch.ones(env.num_envs, device=env.device)


def cart_position_l2(env: "ManagerBasedRlEnv", joint_name: str = "cart_slide") -> torch.Tensor:
    return env.robot_state.joint_pos[:, _joint_index(env, joint_name)].square()


def cart_velocity_l2(env: "ManagerBasedRlEnv", joint_name: str = "cart_slide") -> torch.Tensor:
    return env.robot_state.joint_vel[:, _joint_index(env, joint_name)].square()


def pole_upright(env: "ManagerBasedRlEnv", joint_name: str = "pole_hinge") -> torch.Tensor:
    pole_angle = env.robot_state.joint_pos[:, _joint_index(env, joint_name)]
    return torch.cos(pole_angle)


def pole_angle_l2(env: "ManagerBasedRlEnv", joint_name: str = "pole_hinge") -> torch.Tensor:
    return env.robot_state.joint_pos[:, _joint_index(env, joint_name)].square()


def pole_velocity_l2(env: "ManagerBasedRlEnv", joint_name: str = "pole_hinge") -> torch.Tensor:
    return env.robot_state.joint_vel[:, _joint_index(env, joint_name)].square()


def double_pole_upright(
    env: "ManagerBasedRlEnv",
    joint_names: tuple[str, ...] = ("pole_1_hinge", "pole_2_hinge"),
) -> torch.Tensor:
    """Average of ``cos(angle)`` across the listed pole hinges."""
    indices = [_joint_index(env, n) for n in joint_names]
    angles = env.robot_state.joint_pos[:, indices]
    return torch.cos(angles).mean(dim=-1)


def double_pole_alignment(
    env: "ManagerBasedRlEnv",
    joint_names: tuple[str, ...] = ("pole_1_hinge", "pole_2_hinge"),
) -> torch.Tensor:
    """``(pole_2 - pole_1)^2`` so PPO learns to keep the chain colinear."""
    indices = [_joint_index(env, n) for n in joint_names]
    angles = env.robot_state.joint_pos[:, indices]
    return (angles[:, 1] - angles[:, 0]).square()


def double_pole_velocity_l2(
    env: "ManagerBasedRlEnv",
    joint_names: tuple[str, ...] = ("pole_1_hinge", "pole_2_hinge"),
) -> torch.Tensor:
    indices = [_joint_index(env, n) for n in joint_names]
    return env.robot_state.joint_vel[:, indices].square().sum(dim=-1)


def cart_position_exceeds(
    env: "ManagerBasedRlEnv",
    limit: float,
    joint_name: str = "cart_slide",
) -> torch.Tensor:
    return env.robot_state.joint_pos[:, _joint_index(env, joint_name)].abs() > limit


def pole_angle_exceeds(
    env: "ManagerBasedRlEnv",
    limit: float,
    joint_name: str = "pole_hinge",
) -> torch.Tensor:
    return env.robot_state.joint_pos[:, _joint_index(env, joint_name)].abs() > limit


def any_pole_angle_exceeds(
    env: "ManagerBasedRlEnv",
    limits: dict[str, float],
) -> torch.Tensor:
    """``done`` if any of the listed pole hinges exceeds its per-joint limit."""
    done = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for joint_name, limit in limits.items():
        idx = _joint_index(env, joint_name)
        done |= env.robot_state.joint_pos[:, idx].abs() > limit
    return done


def push_cart_by_setting_joint_velocity(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor | None,
    velocity_range: tuple[float, float],
    joint_name: str = "cart_slide",
) -> None:
    """Interval-mode disturbance: overwrite the cart's slide velocity."""
    if env_ids is None or env_ids.numel() == 0:
        return
    n = int(env_ids.numel())
    idx = _joint_index(env, joint_name)
    lo, hi = velocity_range
    sample = torch.empty(n, device=env.device).uniform_(lo, hi)
    current = env.robot_state.joint_vel.clone()
    current[env_ids, idx] = sample
    target_vel = current[env_ids]
    target_pos = env.robot_state.joint_pos[env_ids]
    env.articulation.write_joint_state(target_pos, target_vel, env_ids)
