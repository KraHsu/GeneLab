"""Play / train dispatch for registered GeneLab tasks.

Factored out of ``cli/__init__.py`` (ADR-0004 / ROADMAP §9 PR R4.3 — final
sub-PR of the CLI dispatcher decomposition). The ``play`` / ``train`` Typer
command callbacks in ``cli/__init__.py`` are the only callers; this module
imports the distributed helpers from ``cli/_distributed.py`` and the agent-kind
picker from ``cli/_interactive.py``, and references nothing in ``genelab.cli``
itself at runtime, so no import cycle is introduced.

The profiler-kwarg coercion (``_coerce_prof_kwargs`` + its ``_parse_*`` helpers)
and the ``_AGENT_KINDS`` set live here rather than in ``cli/__init__.py`` (where
ADR-0004 originally placed them) because the two dispatch functions are their
only users; co-locating keeps this module a self-contained leaf and avoids a
runtime ``cli -> _dispatch -> cli`` cycle.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, cast

from genelab.cli._distributed import _relaunch_under_torchrun, _resolve_per_rank_num_envs
from genelab.cli._interactive import pick_agent_kind

if TYPE_CHECKING:
    from genelab.registry import Runnable

# These keep their leading underscores (they are CLI-package-private, re-exported
# through ``cli/__init__.py``) but are this module's external API — listing them in
# ``__all__`` marks them exported so they are not flagged as unused at the def site.
__all__ = [
    "_dispatch_play",
    "_dispatch_train",
]

_AGENT_KINDS: Final[frozenset[str]] = frozenset({"zero", "random", "trained"})


def _parse_bool(raw: str | None) -> bool | None:
    if raw is None:
        return None
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _parse_int(raw: str | None) -> int | None:
    return int(raw) if raw is not None else None


def _parse_path(raw: str | None) -> Path | None:
    return Path(raw) if raw is not None else None


def _coerce_prof_kwargs(prof_args: dict[str, str]) -> dict[str, Any]:
    """Translate the raw string dict produced by ``split_prof_keys`` into typed kwargs."""
    return {
        "prof": _parse_bool(prof_args.get("prof")),
        "prof_out": _parse_path(prof_args.get("prof_out")),
        "prof_wait": _parse_int(prof_args.get("prof_wait")),
        "prof_warmup": _parse_int(prof_args.get("prof_warmup")),
        "prof_active": _parse_int(prof_args.get("prof_active")),
        "prof_repeat": _parse_int(prof_args.get("prof_repeat")),
        "prof_record_shapes": _parse_bool(prof_args.get("prof_record_shapes")),
        "prof_with_stack": _parse_bool(prof_args.get("prof_with_stack")),
    }


def _is_rl_play_cfg(env_cfg: Any) -> bool:
    """Whether ``env_cfg`` can back the RL play helper (``play_task`` → ``build_env`` →
    ``ManagerBasedRlEnv``).

    Scene-playback demos (Rubiks, Wuji, downstream non-RL tasks) subclass the *non-RL*
    base ``ManagerBasedEnvCfg`` and lack the RL surface (``decimation``,
    ``episode_length_s``, ``actions_cfg``, …), so they must run their own ``task.play()``
    instead — routing them through the RL helper crashes in env construction. See P8/P9.
    """
    if env_cfg is None:
        return False
    from genelab.envs.manager_based_rl_env import ManagerBasedRlEnvCfg

    return isinstance(env_cfg, ManagerBasedRlEnvCfg)


def _dispatch_play(task: Runnable, runner_args: dict[str, str], prof_args: dict[str, str]) -> None:
    task_cfg = getattr(task, "cfg", None)
    agent_cfg = getattr(task_cfg, "agent", None) if task_cfg is not None else None
    checkpoint_raw = runner_args.get("checkpoint")
    agent_raw = runner_args.get("agent")
    if agent_raw is not None and agent_raw not in _AGENT_KINDS:
        picked_agent = pick_agent_kind()
        if picked_agent is None or picked_agent not in _AGENT_KINDS:
            raise SystemExit(f"--agent must be one of {{zero, random, trained}}; got {agent_raw!r}")
        agent_raw = picked_agent
        runner_args["agent"] = picked_agent
    # play is always single-process; either flag is accepted but mapped through the same
    # resolver so the mutual-exclusion guard fires on misuse.
    num_envs_per_rank = _resolve_per_rank_num_envs(runner_args, gpus=1)

    # ``--max-steps`` is the hard, genelab-enforced playback cap: when set it always wins
    # over the soft ``env.simulation.steps`` config and over the viewer gate (the loop
    # stops after this many steps even with a window open). Threaded into every play path
    # — the RL helper (``play_task``) and each task's own ``play()`` — so the semantics are
    # identical regardless of which runner backs the task. ``None`` leaves the soft config
    # in charge (headless: ``simulation.steps``; viewer: run until the window closes).
    max_steps_raw = runner_args.get("max_steps")
    max_steps = int(max_steps_raw) if max_steps_raw is not None else None

    # The CLI's already-overridden cfg (play_env when configured): TASKS.get returns a
    # fresh task each call, so the runner re-resolving would discard the --vis / --gpu /
    # --a.env.* overrides applied above.
    play_env_cfg = getattr(task_cfg, "play_env", None)
    if play_env_cfg is None:
        play_env_cfg = getattr(task_cfg, "env", None)

    # Scene-playback demos (non-RL: env cfg is a base ManagerBasedEnvCfg) can't go through
    # the RL play helper — ManagerBasedRlEnv requires an RL cfg surface they don't have, so
    # play_task → build_env would crash in env construction. Run their own .play() instead.
    # The RL-only options below have no meaning for a fixed scene demo, so warn and ignore
    # them rather than crashing. See P8/P9.
    if not _is_rl_play_cfg(play_env_cfg):
        ignored = [
            name
            for name, present in (
                ("--checkpoint", checkpoint_raw is not None),
                ("--num-envs", num_envs_per_rank is not None),
                ("--agent", agent_raw is not None),
                ("--prof*", bool(prof_args)),
            )
            if present
        ]
        if ignored:
            task_name = getattr(task_cfg, "name", None) or "task"
            print(
                f"warning: {task_name} is a non-RL scene-playback task; ignoring "
                f"{', '.join(ignored)} and running its built-in playback.",
                file=sys.stderr,
            )
        task.play(max_steps=max_steps)
        return

    if (
        checkpoint_raw is None
        and num_envs_per_rank is None
        and agent_raw is None
        and agent_cfg is None
        and not prof_args
    ):
        task.play(max_steps=max_steps)
        return
    from genelab.rl import AgentKind, play_task

    task_id = getattr(task_cfg, "name", None)
    if not isinstance(task_id, str):
        raise SystemExit("task config is missing 'name'; cannot route through RL play helper")
    play_task(
        task_id,
        env_cfg=play_env_cfg,
        checkpoint=Path(checkpoint_raw) if checkpoint_raw is not None else None,
        num_envs=num_envs_per_rank,
        agent=cast("AgentKind | None", agent_raw),
        max_steps=max_steps,
        **_coerce_prof_kwargs(prof_args),
    )


def _dispatch_train(task: Runnable, runner_args: dict[str, str], prof_args: dict[str, str]) -> None:
    task_cfg = getattr(task, "cfg", None)
    agent_cfg = getattr(task_cfg, "agent", None) if task_cfg is not None else None
    if agent_cfg is None:
        task.train()
        return
    from genelab.rl import select_backend, train_task

    # The backend is chosen by the agent cfg type (RSL-RL, skrl, ...); an
    # unregistered type raises a clear error here instead of deep in the runner.
    try:
        backend = select_backend(agent_cfg)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    task_id = getattr(task_cfg, "name", None)
    if not isinstance(task_id, str):
        raise SystemExit("task config is missing 'name'; cannot route through RL train helper")

    gpus_raw = runner_args.pop("gpus", None)
    gpus = int(gpus_raw) if gpus_raw is not None else 1
    num_envs_per_rank = _resolve_per_rank_num_envs(runner_args, gpus=gpus)
    if gpus > 1:
        if backend.name != "rsl_rl":
            raise SystemExit(
                f"multi-GPU training (--gpus {gpus}) is only supported by the RSL-RL "
                f"backend; this task uses the {backend.name!r} backend"
            )
        if "TORCHELASTIC_RUN_ID" not in os.environ:
            _relaunch_under_torchrun(
                gpus, agent_cfg, runner_args, num_envs_per_rank, task_id=task_id
            )
            return

    max_iter_raw = runner_args.get("max_iterations")
    seed_raw = runner_args.get("seed")
    log_dir_raw = runner_args.get("log_dir")
    from genelab.rl.eval_callback import EvalCallbackCfg

    eval_callback = EvalCallbackCfg.from_args(runner_args)
    train_task(
        task_id,
        agent_cfg,
        # Pass the CLI's already-overridden cfg: TASKS.get returns a fresh task
        # each call, so the runner re-resolving would discard --gpu / --a.env.* .
        env_cfg=getattr(task_cfg, "env", None),
        num_envs=num_envs_per_rank,
        max_iterations=int(max_iter_raw) if max_iter_raw is not None else None,
        seed=int(seed_raw) if seed_raw is not None else None,
        log_dir=Path(log_dir_raw) if log_dir_raw is not None else None,
        eval_callback=eval_callback,
        **_coerce_prof_kwargs(prof_args),
    )
