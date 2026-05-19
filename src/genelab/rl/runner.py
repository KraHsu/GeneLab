"""Train / play entry points.

``train_task`` / ``play_task`` are backend-agnostic dispatchers: they resolve the
env config, build the ``ManagerBasedRlEnv`` (and bridges, for play), then hand a
:class:`~genelab.rl.backends.base.TrainContext` / ``PlayContext`` to the backend
that owns the task's agent config (see :mod:`genelab.rl.backends`). The RSL-RL and
skrl library-specific code lives in ``genelab.rl.backends.rsl_rl`` / ``.skrl``.

This module also keeps the helpers shared by every backend: the bridge lifecycle,
the rollout loop, log-dir resolution, run-param dumps, and the zero/random policies.
"""

import dataclasses
import datetime as dt
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from genelab.bridges.base import Bridge
from genelab.cache import ensure_project_cache
from genelab.registry import TASKS
from genelab.rl.backends import (
    AgentKind,
    PlayContext,
    ProfileArgs,
    TrainContext,
    default_backend,
    select_backend,
)

if TYPE_CHECKING:
    from genelab.envs.manager_based_rl_env import ManagerBasedRlEnv

__all__ = ["AgentKind", "play_task", "resolve_log_dir", "train_task"]

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


def _resolve_env_cfg(task_id: str, play: bool) -> Any:
    """Pull the train or play env cfg off a registered task."""
    task = TASKS.get(task_id)
    task_cfg = getattr(task, "cfg", None)
    if task_cfg is None:
        raise ValueError(f"task {task_id!r} has no .cfg attribute")
    env_cfg = task_cfg.play_env if play and task_cfg.play_env is not None else task_cfg.env
    if env_cfg is None:
        raise ValueError(f"task {task_id!r} has no env config")
    return env_cfg


def _build_env(env_cfg: Any) -> "ManagerBasedRlEnv":
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
    num_envs: int | None = None,
    max_iterations: int | None = None,
    seed: int | None = None,
    log_root: Path | None = None,
    log_dir: Path | None = None,
    resume_from: Path | None = None,
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

    ``log_dir`` (final, pre-resolved) takes precedence over ``log_root`` (parent
    under which a ``<experiment>/<timestamp>`` directory is created). The torchrun
    relaunch path pre-resolves the log dir so every rank lands in the same folder.

    ``prof*`` keyword arguments override the matching ``GENELAB_PROFILE_*`` env vars;
    see ``genelab.rl.profiler.maybe_profile`` for the semantics.
    """
    ensure_project_cache()
    env_cfg = _resolve_env_cfg(task_id, play=False)
    if num_envs is not None:
        env_cfg.simulation.num_envs = int(num_envs)
    if seed is not None:
        env_cfg.seed = int(seed)

    env = _build_env(env_cfg)
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
    return select_backend(agent_cfg).train(ctx)


def play_task(
    task_id: str,
    *,
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

    ``prof*`` keyword arguments override the matching ``GENELAB_PROFILE_*`` env vars;
    see ``genelab.rl.profiler.maybe_profile`` for the semantics.
    """
    ensure_project_cache()
    kind: AgentKind = (
        agent if agent is not None else ("trained" if checkpoint is not None else "zero")
    )
    if kind == "trained" and checkpoint is None:
        raise SystemExit("agent='trained' requires a --checkpoint path")
    env_cfg = _resolve_env_cfg(task_id, play=True)
    if num_envs is not None:
        env_cfg.simulation.num_envs = int(num_envs)
    env = _build_env(env_cfg)
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
