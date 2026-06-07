"""``eval_task`` — evaluate a registered task's checkpoint and return ``(result, payload)``.

Backend-agnostic eval orchestration: resolve the task's play env, build it, run a
deterministic rollout through the chosen backend's ``make_inference_setup``, and
build the ``eval.json`` payload. Lives in the ``rl`` layer (its only dependencies
are ``cache`` / ``registry`` / ``rl.*``); both the ``genelab eval`` CLI command
and the in-training ``EvalCallback`` call it, so it must not live above them in
``cli`` (which would re-form the ``rl.eval_callback -> cli._eval`` layering
violation).
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
    # Function-local so importing this module does not pull in ``rl.runner`` (which
    # reaches back into ``rl.eval_callback`` -> here): keeps the import graph acyclic.
    from genelab.rl.backends import select_backend
    from genelab.rl.runner import build_env, resolve_env_cfg

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
    # Eval is always headless (no human-in-the-loop) — force ``vis=False`` so
    # tasks whose ``play_env`` was configured for viewer playback still run on
    # CI / remote servers without a display.
    env_cfg.simulation.vis = False
    # Some tasks set ``episode_length_s = 1e9`` in their play_env for infinite
    # viewer playback. Eval can't run with that — episodes never truncate and
    # the rollout never collects ``episodes`` complete trajectories. Clamp to
    # a defensive 30 s cap so eval makes progress; if a task legitimately
    # needs longer episodes, set its play_env to a finite value.
    if env_cfg.episode_length_s > 30.0:
        env_cfg.episode_length_s = 30.0
    # Eval collects ``episodes`` complete trajectories by relying on the env to
    # auto-reset each terminated sub-env (``run_evaluation`` commits an episode on
    # every ``done[i]`` and never resets the env itself). A play_env configured for
    # interactive teleop sets ``auto_reset = False`` so a human-driven robot is not
    # yanked back on a stumble — but under eval that means a terminated env is never
    # reset, so the rollout spews degenerate ~1-step "episodes" off the frozen
    # terminal state and ``length_mean`` / ``return_mean`` collapse. Force it on here,
    # alongside the other play→eval overrides above.
    env_cfg.auto_reset = True
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
