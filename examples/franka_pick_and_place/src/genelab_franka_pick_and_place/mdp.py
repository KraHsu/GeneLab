"""MDP terms specific to the Franka pick-and-place task.

The terms here read or write the cube's pose through ``env.scene.rigid_objects["cube"]``
and store/sample a per-env goal buffer cached on the env instance. The pattern mirrors
``genelab.mdp.events.reset_root_state_uniform`` — call the Genesis batched API first and
fall back to the non-batched signature on older builds.
"""

from typing import TYPE_CHECKING

import torch

from genelab.entity._torch import to_tensor

from genelab_franka_pick_and_place.constants import (
    CUBE_DROP_Z,
    CUBE_ENTITY_NAME,
    CUBE_HALF,
    CUBE_NOMINAL_XY,
    CUBE_XY_RANGE,
    DISTANCE_THRESHOLD,
    EE_LINK,
    GOAL_AIR_PROB,
    GOAL_XY_RANGE,
    GOAL_Z_MAX,
)

if TYPE_CHECKING:
    from genelab.envs.manager_based_rl_env import ManagerBasedRlEnv


_GOAL_ATTR = "_franka_pp_goal_w"
_EE_INDEX_CACHE: dict[int, int] = {}


def _ee_index(env: "ManagerBasedRlEnv") -> int:
    cached = _EE_INDEX_CACHE.get(id(env))
    if cached is None:
        cached = env.articulation.link_names.index(EE_LINK)
        _EE_INDEX_CACHE[id(env)] = cached
    return cached


def _cube_handle(env: "ManagerBasedRlEnv"):
    return env.scene.rigid_objects[CUBE_ENTITY_NAME].gs_handle


def _ensure_goal_buffer(env: "ManagerBasedRlEnv") -> torch.Tensor:
    buf = getattr(env, _GOAL_ATTR, None)
    if buf is None or buf.shape[0] != env.num_envs:
        buf = torch.zeros(env.num_envs, 3, device=env.device)
        # Seed with the nominal cube position so observations are well-defined before
        # ``resample_goal_uniform`` ever fires (e.g. the very first ``compute`` call
        # on startup before the reset event runs).
        buf[:, 0] = CUBE_NOMINAL_XY[0]
        buf[:, 1] = CUBE_NOMINAL_XY[1]
        buf[:, 2] = CUBE_HALF
        setattr(env, _GOAL_ATTR, buf)
    return buf


def _cube_pos(env: "ManagerBasedRlEnv") -> torch.Tensor:
    raw = _cube_handle(env).get_pos()
    pos = to_tensor(raw, env.device)
    if pos.dim() == 1:
        pos = pos.unsqueeze(0).expand(env.num_envs, -1)
    return pos


def _ee_pos(env: "ManagerBasedRlEnv") -> torch.Tensor:
    return env.robot_state.link_pos[:, _ee_index(env), :]


# ----------------------------------------------------------------- observations


def cube_pos_obs(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """3-vector world position of the cube, per env."""
    return _cube_pos(env)


def ee_pos_obs(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """3-vector world position of the Franka hand link, per env."""
    return _ee_pos(env)


def goal_pos_obs(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """3-vector world goal position (per-env target), refreshed at reset."""
    return _ensure_goal_buffer(env)


def cube_to_goal_vec_obs(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """``goal - cube`` (3-vector). Gives the policy a directional cue at zero noise."""
    return _ensure_goal_buffer(env) - _cube_pos(env)


# ----------------------------------------------------------------- rewards


def ee_to_cube_distance(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """L2 distance from EE to cube. Apply with a NEGATIVE weight (panda-gym dense form)."""
    return torch.linalg.vector_norm(_ee_pos(env) - _cube_pos(env), dim=-1)


def cube_to_goal_distance(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """L2 distance from cube to goal. Apply with a NEGATIVE weight."""
    return torch.linalg.vector_norm(_cube_pos(env) - _ensure_goal_buffer(env), dim=-1)


def success_bonus(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """``1.0`` while ``d(cube, goal) < DISTANCE_THRESHOLD`` (sparse top-up)."""
    d = cube_to_goal_distance(env)
    return (d < DISTANCE_THRESHOLD).to(d.dtype)


# ----------------------------------------------------------------- terminations


def cube_dropped(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """``True`` once the cube has fallen below ``CUBE_DROP_Z`` (escaped the world)."""
    return _cube_pos(env)[:, 2] < CUBE_DROP_Z


# ----------------------------------------------------------------- events


def reset_cube_uniform(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor | None,
    xy_range: tuple[float, float] = CUBE_XY_RANGE,
    nominal_xy: tuple[float, float] = CUBE_NOMINAL_XY,
    z: float = CUBE_HALF,
) -> None:
    """Reset the cube to ``nominal_xy + U(*xy_range)`` (per env) and zero its velocity."""
    if env_ids is None or env_ids.numel() == 0:
        return
    n = int(env_ids.numel())
    lo, hi = xy_range
    new_pos = torch.empty(n, 3, device=env.device)
    new_pos[:, 0] = nominal_xy[0] + torch.empty(n, device=env.device).uniform_(lo, hi)
    new_pos[:, 1] = nominal_xy[1] + torch.empty(n, device=env.device).uniform_(lo, hi)
    new_pos[:, 2] = float(z)
    handle = _cube_handle(env)
    set_pos = getattr(handle, "set_pos", None)
    if set_pos is not None:
        try:
            set_pos(new_pos, envs_idx=env_ids)
        except TypeError:
            set_pos(new_pos)
    zeros = torch.zeros(n, 3, device=env.device)
    for fn_name in ("set_vel", "set_ang"):
        fn = getattr(handle, fn_name, None)
        if fn is None:
            continue
        try:
            fn(zeros, envs_idx=env_ids)
        except TypeError:
            fn(zeros)


def resample_goal_uniform(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor | None,
    xy_range: tuple[float, float] = GOAL_XY_RANGE,
    nominal_xy: tuple[float, float] = CUBE_NOMINAL_XY,
    z_max: float = GOAL_Z_MAX,
    air_prob: float = GOAL_AIR_PROB,
    ground_z: float = CUBE_HALF,
) -> None:
    """Sample new goals for ``env_ids`` into the env's goal buffer.

    With probability ``1 - air_prob`` the goal sits on the ground at ``z = ground_z``
    (panda-gym's "stay on the table" branch); otherwise ``z`` is uniform in
    ``[ground_z, z_max]``.
    """
    if env_ids is None or env_ids.numel() == 0:
        return
    buf = _ensure_goal_buffer(env)
    n = int(env_ids.numel())
    lo, hi = xy_range
    xy = torch.empty(n, 2, device=env.device)
    xy[:, 0] = nominal_xy[0] + torch.empty(n, device=env.device).uniform_(lo, hi)
    xy[:, 1] = nominal_xy[1] + torch.empty(n, device=env.device).uniform_(lo, hi)
    z_air = torch.empty(n, device=env.device).uniform_(ground_z, z_max)
    in_air = torch.rand(n, device=env.device) < air_prob
    z = torch.where(in_air, z_air, torch.full_like(z_air, ground_z))
    buf[env_ids, 0] = xy[:, 0]
    buf[env_ids, 1] = xy[:, 1]
    buf[env_ids, 2] = z
