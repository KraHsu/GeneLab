"""``genelab eval`` — deterministic rollout evaluation that writes ``eval.json``.

Loads ``checkpoint`` against the task's play env (falling back to the train env
when no ``play_env`` is defined), drives a vectorized rollout via the backend's
``make_inference_setup``, and writes the result in the schema documented under
ROADMAP §M1.1. Designed so eval numbers are reproducible across the three
backends (``rsl_rl`` / ``skrl`` / ``sb3``) by routing through the shared
:func:`genelab.rl.evaluator.run_evaluation`.
"""

import datetime as dt
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from genelab.cache import ensure_project_cache
from genelab.registry import TASKS
from genelab.rl.backends.base import PlayContext, ProfileArgs
from genelab.rl.evaluator import EvalConfig, EvalResult, run_evaluation
from genelab.rl.runner import build_env, resolve_env_cfg


def eval_task(
    task_id: str,
    checkpoint: Path,
    *,
    num_envs: int = 64,
    episodes: int = 100,
    seed: int = 0,
    deterministic: bool = True,
    max_steps: int | None = None,
    out_path: Path | None = None,
) -> tuple[EvalResult, dict[str, Any]]:
    """Run deterministic eval and return ``(result, payload)``.

    ``payload`` is the dict serialized to ``out_path`` (when provided); returning it
    lets callers inspect the eval result programmatically (used by ``EvalCallback``).
    The schema is:

        {
            "task": str,
            "checkpoint": str,
            "num_episodes": int,
            "metrics": { "return_mean", "return_std", "length_mean", "success_rate" },
            "wall_clock_seconds": float,
            "seed": int,
            "deterministic": bool,
            "evaluated_at": ISO-8601 str,
        }
    """
    from genelab.rl.backends import select_backend

    ensure_project_cache()
    task = TASKS.get(task_id)
    task_cfg = getattr(task, "cfg", None)
    if task_cfg is None:
        raise SystemExit(f"task {task_id!r} has no .cfg attribute")
    agent_cfg = getattr(task_cfg, "agent", None)
    if agent_cfg is None:
        raise SystemExit(
            f"task {task_id!r} did not register an agent cfg; eval requires a trainable task"
        )

    env_cfg = resolve_env_cfg(task_id, play=True)
    env_cfg.simulation.num_envs = int(num_envs)
    env_cfg.seed = int(seed)
    env = build_env(env_cfg)

    backend = select_backend(agent_cfg)
    ctx = PlayContext(
        task_id=task_id,
        env=env,
        env_cfg=env_cfg,
        agent_cfg=agent_cfg,
        checkpoint=checkpoint,
        kind="trained",
        deterministic=deterministic,
        max_steps=max_steps,
        bridges=[],
        profile=ProfileArgs(),
    )
    try:
        setup = backend.make_inference_setup(ctx)
        result = run_evaluation(
            setup,
            EvalConfig(
                num_envs=int(num_envs),
                episodes=int(episodes),
                seed=int(seed),
                deterministic=bool(deterministic),
                max_steps=max_steps,
            ),
            num_envs=int(num_envs),
        )
    finally:
        env.close()

    payload: dict[str, Any] = {
        "task": task_id,
        "checkpoint": str(checkpoint),
        "num_episodes": result.num_episodes,
        "metrics": asdict(result.metrics),
        "wall_clock_seconds": result.wall_clock_seconds,
        "seed": int(seed),
        "deterministic": bool(deterministic),
        "evaluated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2))
    return result, payload
