"""Backend-agnostic deterministic evaluation.

``run_evaluation`` drives a loaded :class:`~genelab.rl.backends.base.InferenceSetup`
through a vectorized rollout until at least ``cfg.episodes`` complete episodes have
been collected, then aggregates per-episode return, length, and (optional) success
into a :class:`EvalResult`. ``rl/eval_task.py`` writes the result to ``eval.json``.

The success-rate signal follows a gymnasium-style convention: a task opts in by
publishing a per-env ``bool`` tensor at ``extras["is_success"]`` from its
``ManagerBasedRlEnv`` step. When absent across the whole rollout, ``success_rate``
is ``None`` and the field is serialized as JSON ``null``.
"""

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from genelab.rl.backends.base import InferenceSetup


@dataclass
class EvalConfig:
    """Settings for one :func:`run_evaluation` call."""

    num_envs: int = 64
    episodes: int = 100
    seed: int = 0
    deterministic: bool = True
    max_steps: int | None = None  # safety cap; None = no cap


@dataclass
class EvalMetrics:
    return_mean: float
    return_std: float
    length_mean: float
    success_rate: float | None  # None when no ``is_success`` was reported during rollout


@dataclass
class EvalResult:
    metrics: EvalMetrics
    num_episodes: int
    wall_clock_seconds: float


def _as_float_list(values: Any) -> list[float]:
    """Convert a tensor / array / sequence to a plain ``list[float]`` on CPU."""
    import torch

    if isinstance(values, torch.Tensor):
        return [float(v) for v in values.detach().to("cpu").tolist()]
    return [float(v) for v in values]


def _coerce_per_env_bool(values: Any, num_envs: int) -> list[bool] | None:
    """Coerce a per-env ``is_success`` tensor / array / list to ``list[bool]`` length ``num_envs``.

    Returns ``None`` when the shape is unexpected (so the caller treats this step as
    "no signal" rather than aborting the whole eval).
    """
    import torch

    if isinstance(values, torch.Tensor):
        if values.numel() != num_envs:
            return None
        return [bool(v) for v in values.view(-1).detach().to("cpu").tolist()]
    try:
        lst = list(values)
    except TypeError:
        return None
    if len(lst) != num_envs:
        return None
    return [bool(v) for v in lst]


def run_evaluation(
    setup: "InferenceSetup",
    cfg: EvalConfig,
    num_envs: int,
) -> EvalResult:
    """Roll out ``setup.policy`` until ``cfg.episodes`` complete episodes are collected.

    Vectorized: every env contributes to the episode count as it finishes. Returns
    are accumulated per env; on each ``done[i]``, ``returns_so_far[i]`` is committed
    to the global episode list and reset to zero. ``success_rate`` is the fraction of
    completed episodes where ``info["is_success"][i]`` was ``True`` at the step that
    ended them — or ``None`` when no step ever produced an ``is_success`` signal.
    """
    import torch

    adapter = setup.adapter
    policy = setup.policy
    reset_hidden = setup.reset_hidden

    obs, _ = adapter.reset()
    returns_running = torch.zeros(num_envs, dtype=torch.float64)
    lengths_running = torch.zeros(num_envs, dtype=torch.long)
    episode_returns: list[float] = []
    episode_lengths: list[int] = []
    episode_successes: list[bool] = []
    saw_is_success = False

    start = time.perf_counter()
    step_idx = 0
    with torch.inference_mode():
        while len(episode_returns) < cfg.episodes:
            actions = policy(obs)
            obs, reward, dones, info = adapter.step(actions)
            # A recurrent policy must drop hidden state for envs that just auto-reset, so
            # the next step starts each finished episode from a zero state.
            if reset_hidden is not None:
                reset_hidden(dones)

            rewards_list = _as_float_list(reward)
            for i, r in enumerate(rewards_list):
                returns_running[i] += r
                lengths_running[i] += 1

            successes_this_step: list[bool] | None = None
            if isinstance(info, dict) and "is_success" in info:
                successes_this_step = _coerce_per_env_bool(info["is_success"], num_envs)
                if successes_this_step is not None:
                    saw_is_success = True

            dones_list: list[bool]
            if isinstance(dones, torch.Tensor):
                dones_list = [bool(d) for d in dones.view(-1).detach().to("cpu").tolist()]
            else:
                dones_list = [bool(d) for d in dones]

            for i, done in enumerate(dones_list):
                if not done:
                    continue
                episode_returns.append(float(returns_running[i].item()))
                episode_lengths.append(int(lengths_running[i].item()))
                if successes_this_step is not None:
                    episode_successes.append(bool(successes_this_step[i]))
                returns_running[i] = 0.0
                lengths_running[i] = 0
                if len(episode_returns) >= cfg.episodes:
                    break

            step_idx += 1
            if cfg.max_steps is not None and step_idx >= cfg.max_steps:
                break

    wall_clock = time.perf_counter() - start

    if not episode_returns:
        # No complete episode collected; return zeros so the caller can decide.
        metrics = EvalMetrics(
            return_mean=0.0,
            return_std=0.0,
            length_mean=0.0,
            success_rate=None,
        )
        return EvalResult(metrics=metrics, num_episodes=0, wall_clock_seconds=wall_clock)

    returns_t = torch.tensor(episode_returns[: cfg.episodes], dtype=torch.float64)
    lengths_t = torch.tensor(episode_lengths[: cfg.episodes], dtype=torch.float64)
    success_rate: float | None
    if saw_is_success and episode_successes:
        completed = episode_successes[: cfg.episodes]
        success_rate = float(sum(completed) / len(completed)) if completed else None
    else:
        success_rate = None

    metrics = EvalMetrics(
        return_mean=float(returns_t.mean().item()),
        return_std=float(returns_t.std(unbiased=False).item()) if returns_t.numel() > 1 else 0.0,
        length_mean=float(lengths_t.mean().item()),
        success_rate=success_rate,
    )
    return EvalResult(
        metrics=metrics,
        num_episodes=int(returns_t.numel()),
        wall_clock_seconds=wall_clock,
    )
