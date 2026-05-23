"""Shared helpers used by every RL backend.

This module exists to break the import cycle between ``rl.runner`` and
``rl.backends`` documented in ADR-0001. Each backend
(``rl.backends.{rsl_rl, skrl, sb3}``) imports the helpers from here;
``rl.runner`` re-exports them so callers of the public path
(``from genelab.rl.runner import build_env, resolve_env_cfg, …``)
continue to work unchanged.

Function bodies were moved verbatim from ``rl/runner.py``; logic is
intentionally identical so this refactor is purely structural.
"""

import dataclasses
import datetime as dt
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from genelab.bridges.base import Bridge
from genelab.registry import TASKS

if TYPE_CHECKING:
    from genelab.envs.manager_based_rl_env import ManagerBasedRlEnv

__all__ = [
    "build_bridges",
    "build_env",
    "close_bridges",
    "make_random_policy",
    "make_zero_policy",
    "resolve_env_cfg",
    "resolve_log_dir",
    "run_play_loop",
    "save_run_params",
]

_logger = logging.getLogger(__name__)


def build_bridges(env_cfg: Any) -> list[Bridge]:
    """Instantiate every bridge declared on ``env_cfg.bridges_cfg``.

    Entries with ``class_type=None`` are skipped (matches manager-term convention).
    """
    cfg_dict = getattr(env_cfg, "bridges_cfg", {}) or {}
    bridges: list[Bridge] = []
    for name, bridge_cfg in cfg_dict.items():
        cls = getattr(bridge_cfg, "class_type", None)
        if cls is None:
            continue
        try:
            bridges.append(cls(bridge_cfg))
        except Exception:
            _logger.exception("failed to instantiate bridge %r; skipping", name)
    return bridges


def close_bridges(bridges: list[Bridge], env: "ManagerBasedRlEnv") -> None:
    """Call ``on_close`` on every bridge; swallow per-bridge exceptions.

    Runs inside the play loop's ``finally`` so a misbehaving bridge can't block
    ``env.close()``.
    """
    for bridge in bridges:
        try:
            bridge.on_close(env)
        except Exception:
            _logger.exception("bridge %r raised in on_close; continuing teardown", bridge)


def run_play_loop(
    env: "ManagerBasedRlEnv",
    wrapped: Any,
    policy: Any,
    bridges: list[Bridge],
    *,
    max_steps: int | None,
    prof_step: Any,
) -> None:
    """Inner play loop. Extracted from :func:`play_task` so the bridge lifecycle
    contract (``pre_step → policy → step → post_step``) is unit-testable without
    spinning up a Genesis scene.
    """
    import torch

    obs, _ = wrapped.reset()
    step = 0
    while True:
        for bridge in bridges:
            bridge.pre_step(env)
        with torch.inference_mode():
            actions = policy(obs)
        obs, _, _, _ = wrapped.step(actions)
        for bridge in bridges:
            bridge.post_step(env)
        if env.viewer_closed:
            break
        if prof_step is not None:
            prof_step()
        step += 1
        if max_steps is not None and step >= max_steps:
            break


def make_zero_policy(num_envs: int, num_actions: int, device: Any) -> Any:
    """Return a policy that always emits zero actions. Backend-agnostic."""
    import torch

    shape = (num_envs, num_actions)

    def _zero_policy(_obs: Any) -> "torch.Tensor":
        return torch.zeros(shape, device=device)

    return _zero_policy


def make_random_policy(num_envs: int, num_actions: int, device: Any) -> Any:
    """Return a policy that emits uniform random actions in ``[-1, 1]``. Backend-agnostic."""
    import torch

    shape = (num_envs, num_actions)

    def _random_policy(_obs: Any) -> "torch.Tensor":
        return 2.0 * torch.rand(shape, device=device) - 1.0

    return _random_policy


def resolve_env_cfg(task_id: str, play: bool) -> Any:
    """Pull the train or play env cfg off a registered task."""
    task = TASKS.get(task_id)
    task_cfg = getattr(task, "cfg", None)
    if task_cfg is None:
        raise ValueError(f"task {task_id!r} has no .cfg attribute")
    env_cfg = task_cfg.play_env if play and task_cfg.play_env is not None else task_cfg.env
    if env_cfg is None:
        raise ValueError(f"task {task_id!r} has no env config")
    return env_cfg


def build_env(env_cfg: Any) -> "ManagerBasedRlEnv":
    from genelab.envs.manager_based_rl_env import ManagerBasedRlEnv

    return ManagerBasedRlEnv(env_cfg)


def resolve_log_dir(log_root: Path, experiment_name: str, run_name: str) -> Path:
    timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    suffix = f"_{run_name}" if run_name else ""
    return log_root / experiment_name / f"{timestamp}{suffix}"


def save_run_params(log_dir: Path, env_cfg: Any, agent_cfg: Any) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    params_dir = log_dir / "params"
    params_dir.mkdir(exist_ok=True)

    def _dump(obj: Any) -> Any:
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return {k: _dump(v) for k, v in asdict(obj).items()}
        if isinstance(obj, dict):
            return {k: _dump(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_dump(v) for v in obj]
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        return repr(obj)

    (params_dir / "env.json").write_text(json.dumps(_dump(env_cfg), indent=2))
    (params_dir / "agent.json").write_text(json.dumps(_dump(agent_cfg), indent=2))
