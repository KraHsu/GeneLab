"""Reorient curriculums (run on episode reset).

Two adaptive schedules ported from the mjlab reference:

- ``reorient_success_curriculum`` — tightens the goal success tolerance from loose to tight
  as the policy reliably reaches goals. This gives the heavily-regularized aligned reward
  weights an early learning signal (without it the strict tolerance yields zero goal-reaches
  and the policy collapses to a static hold).
- ``adaptive_episode_curriculum`` — scales the cube velocity disturbance up as episodes
  survive longer, ramping the sim2real robustness pressure.

Both maintain a single global scalar (shared across envs) and report it for logging.
"""

from typing import TYPE_CHECKING

import torch

from genelab_wuji.reorient.mdp._state import (
    disturbance_scale_value,
    success_curriculum_value,
)

if TYPE_CHECKING:
    from genelab.contracts import EnvContext


def _as_ids(env: "EnvContext", env_ids: torch.Tensor | slice | None) -> torch.Tensor:
    if env_ids is None:
        return torch.arange(env.num_envs, device=env.device)
    if isinstance(env_ids, slice):
        return torch.arange(env.num_envs, device=env.device)[env_ids]
    return env_ids


def reorient_success_curriculum(
    env: "EnvContext",
    env_ids: torch.Tensor | slice | None,
    command_name: str = "reorient_command",
    count_threshold: int = 3,
    delta_per_loop: float = 0.08,
    threshold_start: float = 0.8,
    threshold_end: float = 0.2,
) -> dict[str, float]:
    """Advance the global success progress and set the command's success tolerance.

    Per reset, envs whose ended episode reached ``>= count_threshold`` goals nudge the
    progress up, the rest nudge it down (scaled by ``delta_per_loop / num_envs``). The
    success threshold interpolates ``threshold_start`` (loose) -> ``threshold_end`` (tight).
    """
    ids = _as_ids(env, env_ids)
    cmd = env.command_manager.get_term(command_name)
    reached = cmd.goal_reach_snapshot[ids].float()  # type: ignore[attr-defined]
    delta_env = delta_per_loop / env.num_envs
    delta = torch.where(reached >= count_threshold, delta_env, -delta_env).sum()
    value = success_curriculum_value(env)
    value[0] = float((value[0] + delta).clamp(0.0, 1.0))
    threshold = threshold_start + float(value[0]) * (threshold_end - threshold_start)
    cmd.cfg.success_threshold = threshold  # type: ignore[attr-defined]
    return {
        "value": float(value[0]),
        "success_threshold": threshold,
        "mean_goals_reached": float(reached.mean()) if reached.numel() else 0.0,
    }


def adaptive_episode_curriculum(
    env: "EnvContext",
    env_ids: torch.Tensor | slice | None,
    target_steps: int = 800,
    inc_rate: float = 0.1,
    dec_rate: float = 0.2,
    min_scale: float = 0.05,
) -> dict[str, float]:
    """Scale disturbance intensity up when episodes survive to ``target_steps``, else down."""
    ids = _as_ids(env, env_ids)
    steps = env.episode_length_buf[ids].float()
    progress = (steps / target_steps).clamp(0.0, 1.0)
    delta = torch.where(
        steps >= target_steps,
        inc_rate / env.num_envs,
        -(dec_rate / env.num_envs) * (1.0 - progress),
    ).sum()
    scale = disturbance_scale_value(env)
    scale[0] = float((scale[0] + delta).clamp(min_scale, 1.0))
    return {"disturbance_scale": float(scale[0]), "mean_episode_steps": float(steps.mean())}
