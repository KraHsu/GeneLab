"""Stable-Baselines3 backend — PPO, A2C, SAC, TD3, DDPG (+ HER for the off-policy three).

Every ``stable_baselines3`` import is local to a function so importing this module
(which the backend registry always does) never requires SB3 to be installed; only
actually training or replaying an ``Sb3AgentCfg`` task does.

SB3 builds its own networks from a ``policy`` string + ``policy_kwargs``, so unlike
the skrl backend there is no hand-written model factory.
"""

import copy
from pathlib import Path
from typing import Any, cast

from genelab.rl.backends import register_backend
from genelab.rl.backends.base import InferenceSetup, PlayContext, TrainContext
from genelab.rl.distributed import is_main_process, shutdown_process_group
from genelab.rl.profiler import maybe_profile
from genelab.rl.runner import (
    close_bridges,
    make_random_policy,
    make_zero_policy,
    resolve_log_dir,
    run_play_loop,
    save_run_params,
)
from genelab.rl.sb3_config import Sb3AgentCfg

_ACTIVATIONS = ("elu", "relu", "tanh", "gelu", "selu", "leaky_relu")


def _algo_class(algorithm: str) -> Any:
    """Return the SB3 algorithm class for ``algorithm`` (SB3 import is here)."""
    if algorithm == "PPO":
        from stable_baselines3 import PPO

        return PPO
    if algorithm == "A2C":
        from stable_baselines3 import A2C

        return A2C
    if algorithm == "SAC":
        from stable_baselines3 import SAC

        return SAC
    if algorithm == "TD3":
        from stable_baselines3 import TD3

        return TD3
    if algorithm == "DDPG":
        from stable_baselines3 import DDPG

        return DDPG
    raise ValueError(f"unsupported SB3 algorithm {algorithm!r}")


def _activation(name: str) -> type:
    """Resolve an activation name to its ``torch.nn`` module class."""
    import torch.nn as nn

    table: dict[str, type] = {
        "elu": nn.ELU,
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "gelu": nn.GELU,
        "selu": nn.SELU,
        "leaky_relu": nn.LeakyReLU,
    }
    cls = table.get(name.lower())
    if cls is None:
        raise ValueError(f"unsupported activation {name!r}; known: {list(_ACTIVATIONS)}")
    return cls


def _model_kwargs(
    agent_cfg: Sb3AgentCfg, vec_env: Any, device: str, tb_log: str | None
) -> dict[str, Any]:
    """Build the keyword arguments for the SB3 algorithm constructor.

    Only the keys the selected algorithm accepts are included (PPO / A2C / the
    off-policy three have disjoint hyper-parameter surfaces). ``extra_kwargs`` is
    merged last as an escape hatch.
    """
    her_or_dict = agent_cfg.her.enabled
    kwargs: dict[str, Any] = {
        "policy": "MultiInputPolicy" if her_or_dict else "MlpPolicy",
        "env": vec_env,
        "learning_rate": agent_cfg.learning_rate,
        "gamma": agent_cfg.discount_factor,
        "seed": agent_cfg.seed,
        "verbose": 1,
        "device": device,
        "tensorboard_log": tb_log,
        "policy_kwargs": {
            "net_arch": list(agent_cfg.policy.net_arch),
            "activation_fn": _activation(agent_cfg.policy.activation),
        },
    }
    if agent_cfg.algorithm == "PPO":
        kwargs.update(
            n_steps=agent_cfg.n_steps,
            batch_size=agent_cfg.batch_size,
            n_epochs=agent_cfg.n_epochs,
            gae_lambda=agent_cfg.gae_lambda,
            clip_range=agent_cfg.clip_range,
            ent_coef=agent_cfg.ent_coef,
        )
    elif agent_cfg.algorithm == "A2C":
        kwargs.update(
            n_steps=agent_cfg.n_steps,
            gae_lambda=agent_cfg.gae_lambda,
            ent_coef=agent_cfg.ent_coef,
        )
    else:  # SAC / TD3 / DDPG
        kwargs.update(
            buffer_size=agent_cfg.buffer_size,
            batch_size=agent_cfg.batch_size,
            tau=agent_cfg.tau,
            learning_starts=agent_cfg.learning_starts,
            train_freq=agent_cfg.train_freq,
        )
        if agent_cfg.her.enabled:
            from stable_baselines3 import HerReplayBuffer

            kwargs["replay_buffer_class"] = HerReplayBuffer
            kwargs["replay_buffer_kwargs"] = {
                "n_sampled_goal": agent_cfg.her.n_sampled_goal,
                "goal_selection_strategy": agent_cfg.her.goal_selection_strategy,
            }
    kwargs.update(copy.deepcopy(agent_cfg.extra_kwargs))
    return kwargs


def _make_vec_env(env: Any, agent_cfg: Sb3AgentCfg) -> Any:
    """Wrap ``env`` as a SB3 ``VecEnv`` (goal-conditioned when HER is enabled)."""
    from genelab.rl.sb3_wrapper import GenelabSb3VecEnv

    her_cfg = agent_cfg.her if agent_cfg.her.enabled else None
    return GenelabSb3VecEnv(env, obs_group=agent_cfg.obs_group, her_cfg=her_cfg)


class _PlayAdapter:
    """Presents a ``GenelabSb3VecEnv`` to ``runner.run_play_loop`` as a 4-tuple env.

    ``run_play_loop`` expects ``reset() -> (obs, info)`` and ``step() -> (obs, ...)``;
    SB3's ``VecEnv.reset`` returns only ``obs`` and ``step`` returns a 4-tuple. When
    ``with_info=True`` (eval path), ``step`` returns ``(obs, reward_tensor,
    dones_tensor, consolidated_info)`` — ``info`` is collapsed from SB3's per-env
    list of dicts into a single dict, with ``is_success`` aggregated as a bool
    tensor of shape ``(num_envs,)`` when every entry exposes it.
    """

    def __init__(self, vec_env: Any, *, with_info: bool = False) -> None:
        self._vec_env = vec_env
        self._with_info = with_info

    def reset(self) -> tuple[Any, dict[str, Any]]:
        return self._vec_env.reset(), {}

    def step(self, actions: Any) -> tuple[Any, ...]:
        obs, rewards, dones, infos = self._vec_env.step(actions)
        if not self._with_info:
            return obs, rewards, dones, infos
        import numpy as np
        import torch

        device = self._vec_env.device
        reward_t = torch.as_tensor(np.asarray(rewards), dtype=torch.float32, device=device).view(-1)
        dones_t = torch.as_tensor(np.asarray(dones), dtype=torch.bool, device=device).view(-1)
        info: dict[str, Any] = {}
        if isinstance(infos, list) and infos and all("is_success" in i for i in infos):
            info["is_success"] = torch.as_tensor(
                np.asarray([bool(i["is_success"]) for i in infos]),
                dtype=torch.bool,
                device=device,
            )
        return obs, reward_t, dones_t, info


class Sb3Backend:
    """Trains / replays tasks through Stable-Baselines3 (PPO, A2C, SAC, TD3, DDPG)."""

    name = "sb3"
    cfg_type = Sb3AgentCfg

    def train(self, ctx: TrainContext) -> Path:
        agent_cfg = cast(Sb3AgentCfg, ctx.agent_cfg)
        assert isinstance(agent_cfg, Sb3AgentCfg)
        if ctx.seed is not None:
            agent_cfg.seed = int(ctx.seed)
        # SB3 trains in env timesteps; --max-iterations maps onto the timestep budget.
        total_timesteps = (
            int(ctx.max_iterations) if ctx.max_iterations is not None else agent_cfg.total_timesteps
        )

        from stable_baselines3.common.callbacks import CheckpointCallback

        env = ctx.env
        vec_env = _make_vec_env(env, agent_cfg)
        device = str(vec_env.device)

        log_dir = ctx.log_dir
        if log_dir is None:
            log_root = ctx.log_root or Path("logs") / "sb3"
            log_dir = resolve_log_dir(
                log_root, agent_cfg.experiment.experiment_name, agent_cfg.experiment.run_name
            )
        if is_main_process():
            save_run_params(log_dir, ctx.env_cfg, agent_cfg)

        tb_log = str(log_dir) if agent_cfg.experiment.logger == "tensorboard" else None
        algo_cls = _algo_class(agent_cfg.algorithm)
        if ctx.resume_from is not None:
            model = algo_cls.load(str(ctx.resume_from), env=vec_env, device=device)
        else:
            model = algo_cls(**_model_kwargs(agent_cfg, vec_env, device, tb_log))

        callback: Any = None
        if agent_cfg.experiment.checkpoint_interval > 0:
            save_freq = max(agent_cfg.experiment.checkpoint_interval // vec_env.num_envs, 1)
            callback = CheckpointCallback(
                save_freq=save_freq, save_path=str(log_dir), name_prefix="model"
            )

        try:
            with maybe_profile(**ctx.profile.as_maybe_profile_kwargs()) as prof_step:
                if prof_step is not None:
                    original_step = vec_env.step

                    def _step_with_profiler(*args: Any, **kwargs: Any) -> Any:
                        result = original_step(*args, **kwargs)
                        prof_step()
                        return result

                    vec_env.step = _step_with_profiler  # type: ignore[method-assign]
                model.learn(
                    total_timesteps=total_timesteps,
                    callback=callback,
                    reset_num_timesteps=ctx.resume_from is None,
                )
            if is_main_process():
                model.save(str(log_dir / "model"))
        finally:
            env.close()
            shutdown_process_group()
        return log_dir

    def make_inference_setup(self, ctx: PlayContext) -> InferenceSetup:
        if ctx.checkpoint is None:
            raise SystemExit("Sb3Backend.make_inference_setup requires ctx.checkpoint")
        agent_cfg = ctx.agent_cfg
        if not isinstance(agent_cfg, Sb3AgentCfg):
            raise ValueError(
                f"task {ctx.task_id!r} did not register an SB3 agent cfg; pass agent_cfg explicitly"
            )
        vec_env = _make_vec_env(ctx.env, agent_cfg)
        device = str(vec_env.device)
        model = _algo_class(agent_cfg.algorithm).load(
            str(ctx.checkpoint), env=vec_env, device=device
        )
        deterministic = ctx.deterministic

        def _policy(obs: Any) -> Any:
            actions, _ = model.predict(obs, deterministic=deterministic)
            return actions

        actor_module, actor_input_dim = _extract_sb3_actor(model)
        adapter = _PlayAdapter(vec_env, with_info=True)
        return InferenceSetup(
            wrapper=vec_env,
            adapter=adapter,
            policy=_policy,
            actor_module=actor_module,
            actor_input_dim=actor_input_dim,
        )

    def play(self, ctx: PlayContext) -> None:
        env = ctx.env
        agent_cfg = ctx.agent_cfg

        if ctx.kind == "trained":
            setup = self.make_inference_setup(ctx)
            vec_env = setup.wrapper
            adapter = _PlayAdapter(vec_env, with_info=False)
            policy = setup.policy
        else:
            cfg = agent_cfg if isinstance(agent_cfg, Sb3AgentCfg) else Sb3AgentCfg()
            vec_env = _make_vec_env(env, cfg)
            num_actions = env.action_manager.total_action_dim
            if ctx.kind == "zero":
                policy = make_zero_policy(env.num_envs, num_actions, vec_env.device)
            else:
                assert ctx.kind == "random"
                policy = make_random_policy(env.num_envs, num_actions, vec_env.device)
            adapter = _PlayAdapter(vec_env, with_info=False)

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


def _extract_sb3_actor(model: Any) -> tuple[Any, int]:
    """Return ``(nn.Module, in_dim)`` for the deterministic actor of a loaded SB3 model.

    SB3's ``policy._predict(obs, deterministic=True)`` is the canonical deterministic
    forward for both on-policy (PPO/A2C) and off-policy (SAC/TD3/DDPG) algorithms.
    Wraps it in a small ``nn.Module`` so TorchScript / ONNX see a single
    ``forward(obs) -> actions`` shape.
    """
    import torch.nn as nn

    policy = getattr(model, "policy", None)
    if policy is None:
        return None, 0  # type: ignore[return-value]

    obs_space = getattr(model, "observation_space", None)
    in_dim = 0
    if obs_space is not None and hasattr(obs_space, "shape") and obs_space.shape:
        in_dim = int(obs_space.shape[-1])

    class _Sb3Actor(nn.Module):
        def __init__(self, sb3_policy: Any) -> None:
            super().__init__()
            self.sb3_policy = sb3_policy

        def forward(self, obs: Any) -> Any:
            return self.sb3_policy._predict(obs, deterministic=True)

    return _Sb3Actor(policy), in_dim


register_backend(Sb3Backend())
