"""Train / play helpers wiring ``ManagerBasedRlEnv`` to RSL-RL's ``OnPolicyRunner``."""

import dataclasses
import datetime as dt
import json
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from genelab.cache import ensure_project_cache
from genelab.registry import TASKS
from genelab.rl.config import RslRlOnPolicyRunnerCfg
from genelab.rl.distributed import is_main_process
from genelab.rl.rsl_rl_wrapper import RslRlVecEnvWrapper

if TYPE_CHECKING:
    from genelab.envs.manager_based_rl_env import ManagerBasedRlEnv

AgentKind = Literal["zero", "random", "trained"]


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


_MLP_MODEL_KEYS = {
    "class_name",
    "hidden_dims",
    "activation",
    "obs_normalization",
    "distribution_cfg",
}


def _prune_model_cfg(model: dict[str, Any]) -> dict[str, Any]:
    """Drop keys that rsl_rl's ``MLPModel`` does not accept."""
    if model.get("class_name", "MLPModel") == "MLPModel":
        return {k: v for k, v in model.items() if k in _MLP_MODEL_KEYS}
    return model


def _runner_cfg_to_dict(cfg: RslRlOnPolicyRunnerCfg) -> dict[str, Any]:
    """Convert an RslRlOnPolicyRunnerCfg dataclass into the dict shape RSL-RL expects."""
    data = asdict(cfg)
    data["actor"] = _prune_model_cfg(data["actor"])
    data["critic"] = _prune_model_cfg(data["critic"])
    return data


def resolve_log_dir(log_root: Path, experiment_name: str, run_name: str) -> Path:
    timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    suffix = f"_{run_name}" if run_name else ""
    return log_root / experiment_name / f"{timestamp}{suffix}"


def _save_run_params(log_dir: Path, env_cfg: Any, agent_cfg: RslRlOnPolicyRunnerCfg) -> None:
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


def train_task(
    task_id: str,
    agent_cfg: RslRlOnPolicyRunnerCfg,
    *,
    num_envs: int | None = None,
    max_iterations: int | None = None,
    seed: int | None = None,
    log_root: Path | None = None,
    resume_from: Path | None = None,
) -> Path:
    """Train ``task_id`` with PPO. Returns the log directory."""
    ensure_project_cache()
    env_cfg = _resolve_env_cfg(task_id, play=False)
    if num_envs is not None:
        env_cfg.scene.num_envs = int(num_envs)
    if seed is not None:
        env_cfg.seed = int(seed)
        agent_cfg.seed = int(seed)
    if max_iterations is not None:
        agent_cfg.max_iterations = int(max_iterations)

    env = _build_env(env_cfg)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    from rsl_rl.runners import OnPolicyRunner

    log_root = log_root or Path("logs") / "rsl_rl"
    log_dir = resolve_log_dir(log_root, agent_cfg.experiment_name, agent_cfg.run_name)
    if is_main_process():
        _save_run_params(log_dir, env_cfg, agent_cfg)

    runner = OnPolicyRunner(
        cast(Any, wrapped),
        _runner_cfg_to_dict(agent_cfg),
        log_dir=str(log_dir),
        device=str(wrapped.device),
    )
    if resume_from is not None:
        runner.load(str(resume_from))
    runner.learn(num_learning_iterations=agent_cfg.max_iterations)
    env.close()
    return log_dir


def play_task(
    task_id: str,
    *,
    checkpoint: Path | None = None,
    num_envs: int | None = None,
    agent: AgentKind | None = None,
    agent_cfg: RslRlOnPolicyRunnerCfg | None = None,
    deterministic: bool = True,
    max_steps: int | None = None,
) -> None:
    """Replay a policy. ``agent`` selects between ``"zero"``, ``"random"``, and ``"trained"``.

    When ``agent`` is ``None``, defaults to ``"trained"`` if ``checkpoint`` is set, else ``"zero"``.
    """
    ensure_project_cache()
    kind: AgentKind = agent if agent is not None else ("trained" if checkpoint is not None else "zero")
    if kind == "trained" and checkpoint is None:
        raise SystemExit("agent='trained' requires a --checkpoint path")
    env_cfg = _resolve_env_cfg(task_id, play=True)
    if num_envs is not None:
        env_cfg.scene.num_envs = int(num_envs)
    env = _build_env(env_cfg)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=None)

    import torch

    action_shape = (env.num_envs, wrapped.num_actions)
    device = env.device

    policy: Any
    if kind == "zero":
        def _zero_policy(_obs: Any) -> "torch.Tensor":
            return torch.zeros(action_shape, device=device)

        policy = _zero_policy
    elif kind == "random":
        def _random_policy(_obs: Any) -> "torch.Tensor":
            return 2.0 * torch.rand(action_shape, device=device) - 1.0

        policy = _random_policy
    else:
        assert kind == "trained"
        assert checkpoint is not None
        resolved_agent_cfg = agent_cfg
        if resolved_agent_cfg is None:
            task = TASKS.get(task_id)
            candidate = getattr(task.cfg, "agent", None)
            if not isinstance(candidate, RslRlOnPolicyRunnerCfg):
                raise ValueError(
                    f"task {task_id!r} did not register an agent cfg; pass agent_cfg explicitly"
                )
            resolved_agent_cfg = candidate
        from rsl_rl.runners import OnPolicyRunner

        runner = OnPolicyRunner(
            cast(Any, wrapped),
            _runner_cfg_to_dict(resolved_agent_cfg),
            log_dir=None,
            device=str(wrapped.device),
        )
        runner.load(str(checkpoint))
        policy = runner.get_inference_policy(device=str(wrapped.device))

    obs, _ = wrapped.reset()
    step = 0
    try:
        while True:
            with torch.inference_mode():
                actions = policy(obs)
            obs, _, _, _ = wrapped.step(actions)
            step += 1
            if max_steps is not None and step >= max_steps:
                break
    finally:
        env.close()
