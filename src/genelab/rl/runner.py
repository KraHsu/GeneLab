"""Train / play entry points.

``train_task`` / ``play_task`` are backend-agnostic dispatchers: they resolve the
env config, build the ``ManagerBasedRlEnv`` (and bridges, for play), then hand a
:class:`~genelab.rl.backends.base.TrainContext` / ``PlayContext`` to the backend
that owns the task's agent config (see :mod:`genelab.rl.backends`). The RSL-RL and
skrl library-specific code lives in ``genelab.rl.backends.rsl_rl`` / ``.skrl``.

Per ADR-0001, the helpers shared by every backend (bridge lifecycle, rollout
loop, log-dir resolution, run-param dumps, zero/random policies) live in
:mod:`genelab.rl._helpers`; we re-export them here so external callers using
``from genelab.rl.runner import build_env, resolve_env_cfg, …`` continue to
work without change. Backends import the helpers directly from
``rl._helpers`` to keep ``rl.backends.* → rl.runner → rl.backends.*`` from
re-forming as a static import cycle.
"""

from pathlib import Path
from typing import Any

from genelab.cache import ensure_project_cache
from genelab.registry import TASKS
from genelab.rl._helpers import (
    build_bridges,
    build_env,
    close_bridges,
    make_random_policy,  # noqa: F401  (re-exported for the public runner.py API)
    make_zero_policy,  # noqa: F401  (re-exported for the public runner.py API)
    resolve_env_cfg,
    resolve_log_dir,
    run_play_loop,  # noqa: F401  (re-exported for the public runner.py API)
    save_run_params,  # noqa: F401  (re-exported for the public runner.py API)
)
from genelab.rl.backends import (
    AgentKind,
    PlayContext,
    ProfileArgs,
    TrainContext,
    default_backend,
    select_backend,
)

__all__ = [
    "AgentKind",
    "build_bridges",
    "build_env",
    "close_bridges",
    "make_random_policy",
    "make_zero_policy",
    "play_task",
    "resolve_env_cfg",
    "resolve_log_dir",
    "run_play_loop",
    "save_run_params",
    "train_task",
]


def _profile_args(
    prof: bool | None,
    prof_out: Path | None,
    prof_wait: int | None,
    prof_warmup: int | None,
    prof_active: int | None,
    prof_repeat: int | None,
    prof_record_shapes: bool | None,
    prof_with_stack: bool | None,
) -> ProfileArgs:
    return ProfileArgs(
        prof=prof,
        prof_out=prof_out,
        prof_wait=prof_wait,
        prof_warmup=prof_warmup,
        prof_active=prof_active,
        prof_repeat=prof_repeat,
        prof_record_shapes=prof_record_shapes,
        prof_with_stack=prof_with_stack,
    )


def train_task(
    task_id: str,
    agent_cfg: Any,
    *,
    env_cfg: Any = None,
    num_envs: int | None = None,
    max_iterations: int | None = None,
    seed: int | None = None,
    log_root: Path | None = None,
    log_dir: Path | None = None,
    resume_from: Path | None = None,
    eval_callback: Any = None,
    prof: bool | None = None,
    prof_out: Path | None = None,
    prof_wait: int | None = None,
    prof_warmup: int | None = None,
    prof_active: int | None = None,
    prof_repeat: int | None = None,
    prof_record_shapes: bool | None = None,
    prof_with_stack: bool | None = None,
) -> Path:
    """Train ``task_id``. Returns the log directory.

    The backend is selected from ``type(agent_cfg)`` (``RslRlOnPolicyRunnerCfg`` →
    RSL-RL, ``SkrlAgentCfg`` → skrl). ``max_iterations`` is interpreted by the
    backend — RSL-RL learning iterations for RSL-RL, training timesteps for skrl.

    ``env_cfg`` defaults to the registered task's train env config. The CLI passes
    its own already-overridden copy so command-line env overrides (``--gpu``,
    ``--dt``, ``--a.env.*``) reach training — ``TASKS.get`` returns a fresh task
    each call, so re-resolving here would discard them.

    ``log_dir`` (final, pre-resolved) takes precedence over ``log_root`` (parent
    under which a ``<experiment>/<timestamp>`` directory is created). The torchrun
    relaunch path pre-resolves the log dir so every rank lands in the same folder.

    ``prof*`` keyword arguments override the matching ``GENELAB_PROFILE_*`` env vars;
    see ``genelab.rl.profiler.maybe_profile`` for the semantics.
    """
    ensure_project_cache()
    if env_cfg is None:
        env_cfg = resolve_env_cfg(task_id, play=False)
    if num_envs is not None:
        env_cfg.simulation.num_envs = int(num_envs)
    if seed is not None:
        env_cfg.seed = int(seed)

    profile = _profile_args(
        prof,
        prof_out,
        prof_wait,
        prof_warmup,
        prof_active,
        prof_repeat,
        prof_record_shapes,
        prof_with_stack,
    )
    backend = select_backend(agent_cfg)

    eval_enabled = bool(getattr(eval_callback, "enabled", False))
    if not eval_enabled or max_iterations is None:
        env = build_env(env_cfg)
        ctx = TrainContext(
            task_id=task_id,
            env=env,
            env_cfg=env_cfg,
            agent_cfg=agent_cfg,
            max_iterations=max_iterations,
            seed=seed,
            log_dir=log_dir,
            log_root=log_root,
            resume_from=resume_from,
            profile=profile,
            eval_callback=eval_callback,
        )
        return backend.train(ctx)

    # Eval-callback path: drive ``backend.train`` in chunks of
    # ``eval_callback.eval_every_iters`` so we can stop, eval the latest checkpoint,
    # and promote ``best_model`` between chunks. Each chunk rebuilds the Genesis env
    # because backends close it in their own ``finally``; tune ``--eval-every`` to
    # amortize that cost.
    from genelab.rl.eval_callback import run_with_eval_callback

    if log_dir is None:
        # Pre-resolve the log dir up-front so every chunk lands in the same place
        # (otherwise each chunk gets a fresh timestamped directory).
        experiment_name = getattr(agent_cfg, "experiment_name", None) or getattr(
            getattr(agent_cfg, "experiment", None), "experiment_name", "exp1"
        )
        run_name = getattr(agent_cfg, "run_name", None) or getattr(
            getattr(agent_cfg, "experiment", None), "run_name", ""
        )
        root = log_root or (Path("logs") / backend.name)
        log_dir = resolve_log_dir(root, experiment_name, run_name)

    train_num_envs = int(env_cfg.simulation.num_envs)

    def _train_chunk(chunk_iters: int, resume_from_path: Any) -> None:
        chunk_env = build_env(env_cfg)
        chunk_ctx = TrainContext(
            task_id=task_id,
            env=chunk_env,
            env_cfg=env_cfg,
            agent_cfg=agent_cfg,
            max_iterations=chunk_iters,
            seed=seed,
            log_dir=log_dir,
            log_root=log_root,
            resume_from=resume_from_path,
            profile=profile,
            eval_callback=None,  # avoid re-entry inside chunk
        )
        backend.train(chunk_ctx)

    return run_with_eval_callback(
        task_id=task_id,
        train_chunk=_train_chunk,
        eval_cfg=eval_callback,
        total_iterations=int(max_iterations),
        log_dir=log_dir,
        backend_name=backend.name,
        train_num_envs=train_num_envs,
    )


def play_task(
    task_id: str,
    *,
    env_cfg: Any = None,
    checkpoint: Path | None = None,
    num_envs: int | None = None,
    agent: AgentKind | None = None,
    agent_cfg: Any | None = None,
    deterministic: bool = True,
    max_steps: int | None = None,
    prof: bool | None = None,
    prof_out: Path | None = None,
    prof_wait: int | None = None,
    prof_warmup: int | None = None,
    prof_active: int | None = None,
    prof_repeat: int | None = None,
    prof_record_shapes: bool | None = None,
    prof_with_stack: bool | None = None,
) -> None:
    """Replay a policy. ``agent`` selects between ``"zero"``, ``"random"``, and ``"trained"``.

    When ``agent`` is ``None``, defaults to ``"trained"`` if ``checkpoint`` is set,
    else ``"zero"``. ``"trained"`` routes to the backend owning the task's agent
    config; ``"zero"`` / ``"random"`` are backend-agnostic.

    ``env_cfg`` defaults to the registered task's play env config. The CLI passes
    its own already-overridden copy so command-line env overrides (``--vis``,
    ``--gpu``, ...) reach playback — ``TASKS.get`` returns a fresh task each call,
    so re-resolving here would discard them.

    ``prof*`` keyword arguments override the matching ``GENELAB_PROFILE_*`` env vars;
    see ``genelab.rl.profiler.maybe_profile`` for the semantics.
    """
    ensure_project_cache()
    kind: AgentKind = (
        agent if agent is not None else ("trained" if checkpoint is not None else "zero")
    )
    if kind == "trained" and checkpoint is None:
        raise SystemExit("agent='trained' requires a --checkpoint path")
    if env_cfg is None:
        env_cfg = resolve_env_cfg(task_id, play=True)
    if num_envs is not None:
        env_cfg.simulation.num_envs = int(num_envs)
    # Scripted (zero / random) playback is a bounded smoke rollout: cap it at
    # ``simulation.steps`` (what the ``--steps`` CLI flag sets) so e.g.
    # ``play TASK --agent zero --steps 5`` stops after 5 steps instead of looping
    # forever headless. Trained playback stays unbounded (runs until the viewer is
    # closed or an explicit ``max_steps`` is given).
    if max_steps is None and kind in ("zero", "random"):
        max_steps = int(env_cfg.simulation.steps)
    env = build_env(env_cfg)
    bridges = build_bridges(env_cfg)
    for bridge in bridges:
        bridge.on_build(env)

    resolved_agent_cfg = agent_cfg
    if resolved_agent_cfg is None:
        task = TASKS.get(task_id)
        resolved_agent_cfg = getattr(getattr(task, "cfg", None), "agent", None)

    backend = (
        select_backend(resolved_agent_cfg) if resolved_agent_cfg is not None else default_backend()
    )
    ctx = PlayContext(
        task_id=task_id,
        env=env,
        env_cfg=env_cfg,
        agent_cfg=resolved_agent_cfg,
        checkpoint=checkpoint,
        kind=kind,
        deterministic=deterministic,
        max_steps=max_steps,
        bridges=bridges,
        profile=_profile_args(
            prof,
            prof_out,
            prof_wait,
            prof_warmup,
            prof_active,
            prof_repeat,
            prof_record_shapes,
            prof_with_stack,
        ),
    )
    backend.play(ctx)
