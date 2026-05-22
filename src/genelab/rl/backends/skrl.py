"""skrl backend — multi-algorithm (PPO, A2C, SAC, TD3, DDPG).

Every skrl import is local to a method so importing this module (which the backend
registry always does) never requires skrl to be installed; only actually training
or replaying a ``SkrlAgentCfg`` task does.
"""

import copy
from pathlib import Path
from typing import Any, cast

from genelab.rl.backends import register_backend
from genelab.rl.backends.base import InferenceSetup, PlayContext, TrainContext
from genelab.utils.distributed import is_main_process, shutdown_process_group
from genelab.rl.profiler import maybe_profile
from genelab.rl._helpers import (
    close_bridges,
    make_random_policy,
    make_zero_policy,
    resolve_log_dir,
    run_play_loop,
    save_run_params,
)
from genelab.rl.skrl_config import SkrlAgentCfg


def _agent_class(algorithm: str) -> tuple[Any, dict[str, Any]]:
    """Return ``(AgentClass, DEFAULT_CONFIG)`` for ``algorithm`` (skrl import is here)."""
    if algorithm == "PPO":
        from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG

        return PPO, PPO_DEFAULT_CONFIG
    if algorithm == "A2C":
        from skrl.agents.torch.a2c import A2C, A2C_DEFAULT_CONFIG

        return A2C, A2C_DEFAULT_CONFIG
    if algorithm == "SAC":
        from skrl.agents.torch.sac import SAC, SAC_DEFAULT_CONFIG

        return SAC, SAC_DEFAULT_CONFIG
    if algorithm == "TD3":
        from skrl.agents.torch.td3 import TD3, TD3_DEFAULT_CONFIG

        return TD3, TD3_DEFAULT_CONFIG
    if algorithm == "DDPG":
        from skrl.agents.torch.ddpg import DDPG, DDPG_DEFAULT_CONFIG

        return DDPG, DDPG_DEFAULT_CONFIG
    raise ValueError(f"unsupported skrl algorithm {algorithm!r}")


def _build_skrl_cfg(
    agent_cfg: SkrlAgentCfg, default_cfg: dict[str, Any], log_dir: Path | None
) -> dict[str, Any]:
    """Patch the algorithm's skrl ``DEFAULT_CONFIG`` from ``SkrlAgentCfg``.

    Only keys present in the algorithm's default config are written, so one
    ``SkrlAgentCfg`` drives every algorithm. ``extra_cfg`` is merged last.
    ``log_dir`` is ``None`` for replay (logging disabled).
    """
    cfg = copy.deepcopy(default_cfg)

    def _set(key: str, value: Any) -> None:
        if key in cfg:
            cfg[key] = value

    _set("discount_factor", agent_cfg.discount_factor)
    if agent_cfg.is_on_policy:
        _set("rollouts", agent_cfg.rollouts)
        _set("learning_epochs", agent_cfg.learning_epochs)
        _set("mini_batches", agent_cfg.mini_batches)
        _set("learning_rate", agent_cfg.learning_rate)
        _set("lambda", agent_cfg.gae_lambda)
        _set("ratio_clip", agent_cfg.ratio_clip)
        _set("entropy_loss_scale", agent_cfg.entropy_loss_scale)
        _set("grad_norm_clip", agent_cfg.grad_norm_clip)
    else:
        _set("batch_size", agent_cfg.batch_size)
        _set("polyak", agent_cfg.polyak)
        _set("learning_starts", agent_cfg.learning_starts)
        _set("actor_learning_rate", agent_cfg.learning_rate)
        _set("critic_learning_rate", agent_cfg.learning_rate)
        _set("grad_norm_clip", agent_cfg.grad_norm_clip)

    exp = agent_cfg.experiment
    experiment = dict(cfg.get("experiment", {}))
    if log_dir is not None:
        experiment["directory"] = str(log_dir.parent)
        experiment["experiment_name"] = log_dir.name
        experiment["write_interval"] = exp.write_interval
        experiment["checkpoint_interval"] = exp.checkpoint_interval
        experiment["wandb"] = exp.logger == "wandb"
        experiment["wandb_kwargs"] = {
            "project": exp.wandb_project,
            "tags": list(exp.wandb_tags),
        }
    else:
        # Replay: no TensorBoard writer, no checkpoint dumps.
        experiment["write_interval"] = 0
        experiment["checkpoint_interval"] = 0
        experiment["wandb"] = False
    cfg["experiment"] = experiment

    cfg.update(copy.deepcopy(agent_cfg.extra_cfg))
    return cfg


def _build_agent(agent_cfg: SkrlAgentCfg, wrapper: Any, device: Any, log_dir: Path | None) -> Any:
    """Build a skrl agent (models + memory + patched cfg) for ``agent_cfg``."""
    from skrl.memories.torch import RandomMemory

    from genelab.rl.skrl_models import build_models

    if agent_cfg.is_on_policy:
        memory_size = agent_cfg.rollouts
    else:
        memory_size = max(1, agent_cfg.replay_buffer_size // max(1, wrapper.num_envs))
    memory = RandomMemory(memory_size=memory_size, num_envs=wrapper.num_envs, device=device)

    models = build_models(agent_cfg.algorithm, wrapper, agent_cfg.models, device)
    agent_cls, default_cfg = _agent_class(agent_cfg.algorithm)
    cfg = _build_skrl_cfg(agent_cfg, default_cfg, log_dir)
    return agent_cls(
        models=models,
        memory=memory,
        cfg=cfg,
        observation_space=wrapper.observation_space,
        action_space=wrapper.action_space,
        device=device,
    )


def _make_skrl_policy(agent: Any, *, deterministic: bool) -> Any:
    """Wrap a loaded skrl agent as a ``policy(obs) -> actions`` callable for play."""

    def _policy(obs: Any) -> Any:
        actions, _, outputs = agent.act(obs, timestep=0, timesteps=1)
        if deterministic and isinstance(outputs, dict):
            mean = outputs.get("mean_actions")
            if mean is not None:
                return mean
        return actions

    return _policy


class _PlayAdapter:
    """Presents a ``GenelabSkrlWrapper`` to ``runner._run_play_loop`` as a 4-tuple env.

    skrl's wrapper ``step`` returns a 5-tuple ``(obs, reward, terminated, truncated,
    info)``; the shared play loop expects ``(obs, _, _, _)`` while the eval loop needs
    ``(obs, reward, dones, info)`` so it can read ``info["is_success"]``.
    """

    def __init__(self, wrapper: Any, *, with_info: bool = False) -> None:
        self._wrapper = wrapper
        self._with_info = with_info

    def reset(self) -> Any:
        obs, info = self._wrapper.reset()
        return (obs, info) if self._with_info else (obs, info)

    def step(self, actions: Any) -> tuple[Any, ...]:
        obs, reward, terminated, truncated, info = self._wrapper.step(actions)
        if self._with_info:
            dones = (terminated.bool() | truncated.bool()).view(-1)
            return obs, reward.view(-1), dones, info
        return obs, reward, terminated, truncated


class SkrlBackend:
    """Trains / replays tasks through skrl (PPO, A2C, SAC, TD3, DDPG)."""

    name = "skrl"
    cfg_type = SkrlAgentCfg

    def train(self, ctx: TrainContext) -> Path:
        agent_cfg = ctx.agent_cfg
        assert isinstance(agent_cfg, SkrlAgentCfg)
        if ctx.seed is not None:
            agent_cfg.seed = int(ctx.seed)
        # skrl trains in timesteps; --max-iterations maps onto the timestep budget.
        timesteps = (
            int(ctx.max_iterations) if ctx.max_iterations is not None else agent_cfg.timesteps
        )

        from skrl.trainers.torch import SequentialTrainer
        from skrl.utils import set_seed

        from genelab.rl.vecenvs.skrl import GenelabSkrlWrapper

        set_seed(agent_cfg.seed)
        env = ctx.env
        wrapper = GenelabSkrlWrapper(
            env, obs_group=agent_cfg.obs_group, state_group=agent_cfg.state_group
        )
        device = wrapper.device

        log_dir = ctx.log_dir
        if log_dir is None:
            log_root = ctx.log_root or Path("logs") / "skrl"
            log_dir = resolve_log_dir(
                log_root, agent_cfg.experiment.experiment_name, agent_cfg.experiment.run_name
            )
        if is_main_process():
            save_run_params(log_dir, ctx.env_cfg, agent_cfg)

        agent = _build_agent(agent_cfg, wrapper, device, log_dir)
        if ctx.resume_from is not None:
            agent.load(str(ctx.resume_from))

        trainer = SequentialTrainer(
            # GenelabSkrlWrapper subclasses skrl's Wrapper only at runtime (guarded
            # import), so the static type does not advertise the relationship.
            env=cast(Any, wrapper),
            agents=agent,
            cfg={"timesteps": timesteps, "headless": True, "close_environment_at_exit": False},
        )
        try:
            with maybe_profile(**ctx.profile.as_maybe_profile_kwargs()) as prof_step:
                if prof_step is not None:
                    original_step = wrapper.step

                    def _step_with_profiler(*args: Any, **kwargs: Any) -> Any:
                        result = original_step(*args, **kwargs)
                        prof_step()
                        return result

                    wrapper.step = _step_with_profiler  # type: ignore[method-assign]
                trainer.train()
        finally:
            env.close()
            shutdown_process_group()
        return log_dir

    def make_inference_setup(self, ctx: PlayContext) -> InferenceSetup:
        from genelab.rl.vecenvs.skrl import GenelabSkrlWrapper

        if ctx.checkpoint is None:
            raise SystemExit("SkrlBackend.make_inference_setup requires ctx.checkpoint")
        agent_cfg = ctx.agent_cfg
        if not isinstance(agent_cfg, SkrlAgentCfg):
            raise ValueError(
                f"task {ctx.task_id!r} did not register a skrl agent cfg; pass agent_cfg explicitly"
            )
        wrapper = GenelabSkrlWrapper(
            ctx.env, obs_group=agent_cfg.obs_group, state_group=agent_cfg.state_group
        )
        device = wrapper.device
        agent = _build_agent(agent_cfg, wrapper, device, log_dir=None)
        agent.init(trainer_cfg={"timesteps": 1, "headless": True})
        agent.load(str(ctx.checkpoint))
        agent.set_running_mode("eval")
        policy = _make_skrl_policy(agent, deterministic=ctx.deterministic)
        actor_module, actor_input_dim = _extract_skrl_actor(
            agent, int(wrapper.observation_space.shape[-1])
        )
        adapter = _PlayAdapter(wrapper, with_info=True)
        return InferenceSetup(
            wrapper=wrapper,
            adapter=adapter,
            policy=policy,
            actor_module=actor_module,
            actor_input_dim=actor_input_dim,
        )

    def play(self, ctx: PlayContext) -> None:
        from genelab.rl.vecenvs.skrl import GenelabSkrlWrapper

        env = ctx.env
        agent_cfg = ctx.agent_cfg

        if ctx.kind == "trained":
            setup = self.make_inference_setup(ctx)
            wrapper = setup.wrapper
            adapter = _PlayAdapter(wrapper, with_info=False)
            policy = setup.policy
        else:
            obs_group = agent_cfg.obs_group if isinstance(agent_cfg, SkrlAgentCfg) else "policy"
            state_group = agent_cfg.state_group if isinstance(agent_cfg, SkrlAgentCfg) else "critic"
            wrapper = GenelabSkrlWrapper(env, obs_group=obs_group, state_group=state_group)
            device = wrapper.device
            num_actions = env.action_manager.total_action_dim
            if ctx.kind == "zero":
                policy = make_zero_policy(env.num_envs, num_actions, device)
            else:
                assert ctx.kind == "random"
                policy = make_random_policy(env.num_envs, num_actions, device)
            adapter = _PlayAdapter(wrapper, with_info=False)

        try:
            with maybe_profile(**ctx.profile.as_maybe_profile_kwargs()) as prof_step:
                run_play_loop(
                    env,
                    adapter,
                    policy,
                    ctx.bridges,
                    max_steps=ctx.max_steps,
                    prof_step=prof_step,
                )
        finally:
            close_bridges(ctx.bridges, env)
            env.close()


def _extract_skrl_actor(agent: Any, num_obs: int) -> tuple[Any, int]:
    """Return ``(nn.Module, in_dim)`` extracting the deterministic policy network from a loaded skrl agent.

    Wraps ``agent.policy.act`` in a small ``nn.Module`` that always returns the
    deterministic mean (``info["mean_actions"]`` for stochastic policies) so export
    sees a single ``forward(obs) -> actions`` shape.
    """
    import torch.nn as nn

    policy = getattr(agent, "policy", None)
    if policy is None:
        return None, num_obs  # type: ignore[return-value]

    class _SkrlActor(nn.Module):
        def __init__(self, policy_model: Any) -> None:
            super().__init__()
            self.policy_model = policy_model

        def forward(self, obs: Any) -> Any:
            actions, _, info = self.policy_model.act({"states": obs}, role="policy")
            if isinstance(info, dict):
                mean = info.get("mean_actions")
                if mean is not None:
                    return mean
            return actions

    return _SkrlActor(policy), num_obs


register_backend(SkrlBackend())
