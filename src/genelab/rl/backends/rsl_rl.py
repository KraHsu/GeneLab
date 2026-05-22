"""RSL-RL backend — the default. Drives ``rsl_rl.runners.OnPolicyRunner`` (PPO)."""

from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

from genelab.rl.backends import register_backend
from genelab.rl.backends.base import InferenceSetup, PlayContext, TrainContext
from genelab.rl.config import RslRlOnPolicyRunnerCfg
from genelab.rl.distributed import is_main_process, shutdown_process_group
from genelab.rl.profiler import maybe_profile
from genelab.rl.vecenvs.rsl_rl import RslRlVecEnvWrapper
from genelab.rl._helpers import (
    close_bridges,
    make_random_policy,
    make_zero_policy,
    resolve_log_dir,
    run_play_loop,
    save_run_params,
)

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


class RslRlBackend:
    """Trains / replays tasks through RSL-RL's on-policy PPO runner."""

    name = "rsl_rl"
    cfg_type = RslRlOnPolicyRunnerCfg

    def train(self, ctx: TrainContext) -> Path:
        agent_cfg = cast(RslRlOnPolicyRunnerCfg, ctx.agent_cfg)
        if ctx.seed is not None:
            agent_cfg.seed = int(ctx.seed)
        if ctx.max_iterations is not None:
            agent_cfg.max_iterations = int(ctx.max_iterations)

        env = ctx.env
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        from rsl_rl.runners import OnPolicyRunner

        log_dir = ctx.log_dir
        if log_dir is None:
            log_root = ctx.log_root or Path("logs") / "rsl_rl"
            log_dir = resolve_log_dir(log_root, agent_cfg.experiment_name, agent_cfg.run_name)
        if is_main_process():
            save_run_params(log_dir, ctx.env_cfg, agent_cfg)

        runner = OnPolicyRunner(
            cast(Any, wrapped),
            _runner_cfg_to_dict(agent_cfg),
            log_dir=str(log_dir),
            device=str(wrapped.device),
        )
        if ctx.resume_from is not None:
            runner.load(str(ctx.resume_from))
        try:
            with maybe_profile(**ctx.profile.as_maybe_profile_kwargs()) as prof_step:
                if prof_step is not None:
                    # RSL-RL exposes no per-iteration hook; the wrapper's ``step`` is the
                    # closest fire-once-per-rollout-step seam.
                    original_step = wrapped.step

                    def _step_with_profiler(*args: Any, **kwargs: Any) -> Any:
                        result = original_step(*args, **kwargs)
                        prof_step()
                        return result

                    wrapped.step = _step_with_profiler  # type: ignore[method-assign]
                runner.learn(num_learning_iterations=agent_cfg.max_iterations)
        finally:
            env.close()
            # rsl_rl's OnPolicyRunner inits the NCCL process group but never tears it
            # down; without this every rank prints a destroy_process_group leak warning.
            shutdown_process_group()
        return log_dir

    def make_inference_setup(self, ctx: PlayContext) -> InferenceSetup:
        if ctx.checkpoint is None:
            raise SystemExit("RslRlBackend.make_inference_setup requires ctx.checkpoint")
        agent_cfg = ctx.agent_cfg
        if not isinstance(agent_cfg, RslRlOnPolicyRunnerCfg):
            raise ValueError(
                f"task {ctx.task_id!r} did not register an RSL-RL agent cfg; "
                "pass agent_cfg explicitly"
            )
        wrapped = RslRlVecEnvWrapper(ctx.env, clip_actions=None)

        from rsl_rl.runners import OnPolicyRunner

        runner = OnPolicyRunner(
            cast(Any, wrapped),
            _runner_cfg_to_dict(agent_cfg),
            log_dir=None,
            device=str(wrapped.device),
        )
        runner.load(str(ctx.checkpoint))
        policy = runner.get_inference_policy(device=str(wrapped.device))
        actor_module, actor_input_dim = _extract_rsl_rl_actor(runner, wrapped.num_obs)
        return InferenceSetup(
            wrapper=wrapped,
            adapter=wrapped,
            policy=policy,
            actor_module=actor_module,
            actor_input_dim=actor_input_dim,
        )

    def play(self, ctx: PlayContext) -> None:
        env = ctx.env
        device = env.device

        policy: Any
        if ctx.kind == "trained":
            setup = self.make_inference_setup(ctx)
            wrapped = setup.wrapper
            policy = setup.policy
        else:
            wrapped = RslRlVecEnvWrapper(env, clip_actions=None)
            if ctx.kind == "zero":
                policy = make_zero_policy(env.num_envs, wrapped.num_actions, device)
            else:
                assert ctx.kind == "random"
                policy = make_random_policy(env.num_envs, wrapped.num_actions, device)

        try:
            with maybe_profile(**ctx.profile.as_maybe_profile_kwargs()) as prof_step:
                run_play_loop(
                    env,
                    wrapped,
                    policy,
                    ctx.bridges,
                    max_steps=ctx.max_steps,
                    prof_step=prof_step,
                )
        finally:
            close_bridges(ctx.bridges, env)
            env.close()


def _extract_rsl_rl_actor(runner: Any, num_obs: int) -> tuple[Any, int]:
    """Return ``(nn.Module, in_dim)`` extracting the deterministic actor from a loaded runner.

    Prefers ``runner.alg.actor_critic.actor`` if it is a pure ``nn.Module`` (the common
    MLP case). Otherwise wraps ``actor_critic.act_inference`` in a small ``nn.Module``
    adapter so TorchScript / ONNX still see a single ``forward(obs) -> actions`` shape.
    """
    import torch.nn as nn

    ac = getattr(getattr(runner, "alg", None), "actor_critic", None)
    if ac is None:
        return None, num_obs  # type: ignore[return-value]
    actor = getattr(ac, "actor", None)
    if isinstance(actor, nn.Module):
        for m in actor.modules():
            if isinstance(m, nn.Linear):
                return actor, m.in_features
        return actor, num_obs

    class _RslRlActor(nn.Module):
        def __init__(self, actor_critic: Any) -> None:
            super().__init__()
            self.actor_critic = actor_critic

        def forward(self, obs: Any) -> Any:
            return self.actor_critic.act_inference(obs)

    return _RslRlActor(ac), num_obs


register_backend(RslRlBackend())
