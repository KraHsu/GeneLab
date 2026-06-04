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
        idx = env.articulations["robot"].joint_names.index(joint_name)
        cache[joint_name] = idx
    return idx


def alive_bonus(env: "ManagerBasedRlEnv") -> torch.Tensor:
    return torch.ones(env.num_envs, device=env.device)


def cart_position_l2(env: "ManagerBasedRlEnv", joint_name: str = "cart_slide") -> torch.Tensor:
    return env.articulations["robot"].data.joint_pos[:, _joint_index(env, joint_name)].square()


def cart_velocity_l2(env: "ManagerBasedRlEnv", joint_name: str = "cart_slide") -> torch.Tensor:
    return env.articulations["robot"].data.joint_vel[:, _joint_index(env, joint_name)].square()


def pole_upright(env: "ManagerBasedRlEnv", joint_name: str = "pole_hinge") -> torch.Tensor:
    pole_angle = env.articulations["robot"].data.joint_pos[:, _joint_index(env, joint_name)]
    return torch.cos(pole_angle)


def pole_angle_l2(env: "ManagerBasedRlEnv", joint_name: str = "pole_hinge") -> torch.Tensor:
    return env.articulations["robot"].data.joint_pos[:, _joint_index(env, joint_name)].square()


def pole_velocity_l2(env: "ManagerBasedRlEnv", joint_name: str = "pole_hinge") -> torch.Tensor:
    return env.articulations["robot"].data.joint_vel[:, _joint_index(env, joint_name)].square()


def double_pole_upright(
    env: "ManagerBasedRlEnv",
    joint_names: tuple[str, ...] = ("pole_1_hinge", "pole_2_hinge"),
) -> torch.Tensor:
    """Average of ``cos(angle)`` across the listed pole hinges."""
    indices = [_joint_index(env, n) for n in joint_names]
    angles = env.articulations["robot"].data.joint_pos[:, indices]
    return torch.cos(angles).mean(dim=-1)


def double_pole_alignment(
    env: "ManagerBasedRlEnv",
    joint_names: tuple[str, ...] = ("pole_1_hinge", "pole_2_hinge"),
) -> torch.Tensor:
    """``(pole_2 - pole_1)^2`` so PPO learns to keep the chain colinear."""
    indices = [_joint_index(env, n) for n in joint_names]
    angles = env.articulations["robot"].data.joint_pos[:, indices]
    return (angles[:, 1] - angles[:, 0]).square()


def double_pole_velocity_l2(
    env: "ManagerBasedRlEnv",
    joint_names: tuple[str, ...] = ("pole_1_hinge", "pole_2_hinge"),
) -> torch.Tensor:
    indices = [_joint_index(env, n) for n in joint_names]
    return env.articulations["robot"].data.joint_vel[:, indices].square().sum(dim=-1)


def cart_position_exceeds(
    env: "ManagerBasedRlEnv",
    limit: float,
    joint_name: str = "cart_slide",
) -> torch.Tensor:
    return env.articulations["robot"].data.joint_pos[:, _joint_index(env, joint_name)].abs() > limit


def pole_angle_exceeds(
    env: "ManagerBasedRlEnv",
    limit: float,
    joint_name: str = "pole_hinge",
) -> torch.Tensor:
    return env.articulations["robot"].data.joint_pos[:, _joint_index(env, joint_name)].abs() > limit


def any_pole_angle_exceeds(
    env: "ManagerBasedRlEnv",
    limits: dict[str, float],
) -> torch.Tensor:
    """``done`` if any of the listed pole hinges exceeds its per-joint limit."""
    done = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for joint_name, limit in limits.items():
        idx = _joint_index(env, joint_name)
        done |= env.articulations["robot"].data.joint_pos[:, idx].abs() > limit
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
    current = env.articulations["robot"].data.joint_vel.clone()
    current[env_ids, idx] = sample
    target_vel = current[env_ids]
    target_pos = env.articulations["robot"].data.joint_pos[env_ids]
    env.articulations["robot"].write_joint_state(target_pos, target_vel, env_ids)


# ---- "recall the target" memory task ---------------------------------------------
# A per-episode cart target is shown only for the first ``cue_steps``, then hidden. With
# frequent pushes knocking the cart around, a memoryless policy cannot return to the (now
# unobserved) target — it must have remembered it. This is the task recurrence is for.

_CART_TARGET_ATTR = "_cart_memory_target"


def _cart_target_buf(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Per-env cart target, lazily allocated and stashed on the env object.

    Created on first access (which is the episode-0 reset, so ``reset_cart_target`` fills it
    before any observation/reward reads it). A different ``num_envs`` always means a freshly
    built env, hence a fresh attribute — no resize path is needed.
    """
    buf = getattr(env, _CART_TARGET_ATTR, None)
    if buf is None:
        buf = torch.zeros(env.num_envs, device=env.device)
        setattr(env, _CART_TARGET_ATTR, buf)
    return buf


def reset_cart_target(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor | None,
    target_range: tuple[float, float] = (-1.0, 1.0),
) -> None:
    """Reset-mode: sample a fresh hidden cart target for each reset env."""
    if env_ids is None or env_ids.numel() == 0:
        return
    buf = _cart_target_buf(env)
    lo, hi = target_range
    buf[env_ids] = torch.empty(int(env_ids.numel()), device=env.device).uniform_(lo, hi)


def cart_target_cue(env: "ManagerBasedRlEnv", cue_steps: int = 30) -> torch.Tensor:
    """Observation term: the cart target during the first ``cue_steps``, then ``0``.

    After the cue window the term is identical (zero) regardless of the true target, so a
    memoryless policy has no way to recover it — only a recurrent policy that stored it
    during the cue window can keep tracking it.
    """
    buf = _cart_target_buf(env)
    active = (env.episode_length_buf < cue_steps).to(buf.dtype)
    return (buf * active).unsqueeze(-1)


def cart_target_tracking_l2(
    env: "ManagerBasedRlEnv", joint_name: str = "cart_slide"
) -> torch.Tensor:
    """Squared distance of the cart from its (hidden) target — the thing to minimise."""
    buf = _cart_target_buf(env)
    idx = _joint_index(env, joint_name)
    x = env.articulations["robot"].data.joint_pos[:, idx]
    return (x - buf).square()
