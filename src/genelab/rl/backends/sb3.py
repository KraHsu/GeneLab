"""Stable-Baselines3 backend — PPO, A2C, SAC, TD3, DDPG (+ HER for the off-policy three).

Every ``stable_baselines3`` import is local to a function so importing this module
(which the backend registry always does) never requires SB3 to be installed; only
actually training or replaying an ``Sb3AgentCfg`` task does.

SB3 builds its own networks from a ``policy`` string + ``policy_kwargs``, so unlike
the skrl backend there is no hand-written model factory.
"""

import copy
import logging
from pathlib import Path
from typing import Any, cast

from genelab.rl.backends import register_backend
from genelab.rl.backends.base import PlayContext, TrainContext
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

_logger = logging.getLogger(__name__)


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
            # HerReplayBuffer works in whole episodes, and buffer_size is split
            # per env as buffer_size // num_envs. Two floors, both scaling with
            # num_envs (which the static cfg cannot know):
            #   * learning_starts >= one episode per env, so the first episode
            #     completes before training samples;
            #   * buffer_size >= several episodes per env. A near-one-episode
            #     slice fills and wraps immediately, and HER's wrap bookkeeping
            #     clears the just-finished episode before training can sample it
            #     -- so the buffer must comfortably hold multiple episodes.
            # warning, not info: configured values are being overridden and the
            # CLI does not surface info-level logs.
            max_ep_len = int(getattr(vec_env, "max_episode_length", 0) or 0)
            if max_ep_len:
                unit = vec_env.num_envs * (max_ep_len + 1)
                floors = {"learning_starts": unit, "buffer_size": 8 * unit}
                for key, floor in floors.items():
                    if kwargs[key] < floor:
                        _logger.warning(
                            "HER: raising %s %d -> %d (num_envs=%d, "
                            "episode_length=%d); HER stores whole episodes -- "
                            "prefer a modest --num-envs.",
                            key,
                            kwargs[key],
                            floor,
                            vec_env.num_envs,
                            max_ep_len,
                        )
                        kwargs[key] = floor
    kwargs.update(copy.deepcopy(agent_cfg.extra_kwargs))
    return kwargs


def _make_vec_env(env: Any, agent_cfg: Sb3AgentCfg) -> Any:
    """Wrap ``env`` as a SB3 ``VecEnv`` (goal-conditioned when HER is enabled)."""
    from genelab.rl.sb3_wrapper import GenelabSb3VecEnv

    her_cfg = agent_cfg.her if agent_cfg.her.enabled else None
    return GenelabSb3VecEnv(env, obs_group=agent_cfg.obs_group, her_cfg=her_cfg)


def _resolve_demo_path(agent_cfg: Sb3AgentCfg) -> str | None:
    """Pick up ``demo_path`` from the cfg field or fall back to the env var."""
    import os

    if agent_cfg.demo_path:
        return agent_cfg.demo_path
    env_path = os.environ.get("GENELAB_SB3_DEMO_PATH")
    return env_path if env_path else None


def _maybe_prefill_demos(model: Any, agent_cfg: Sb3AgentCfg, vec_env: Any) -> None:
    """Replay an ``.npz`` of offline transitions into ``model.replay_buffer``.

    The file layout matches what ``genelab_franka_pick_and_place.collect_demos``
    writes: time-major arrays shaped ``(T, num_envs, ...)`` for ``obs_*``,
    ``next_obs_*``, ``actions``, ``rewards``, ``dones``, ``truncateds``. The
    function walks the time axis and calls ``replay_buffer.add(...)`` once per
    step with the per-env batch; ``HerReplayBuffer`` then tracks episode
    boundaries from the ``done`` flag and stores transitions whole-episode."""
    import numpy as np

    path = _resolve_demo_path(agent_cfg)
    if path is None:
        return
    if agent_cfg.is_on_policy:
        _logger.warning(
            "demo_path=%r ignored: only off-policy algorithms (SAC/TD3/DDPG) have a "
            "replay buffer to pre-fill.",
            path,
        )
        return
    if not hasattr(model, "replay_buffer") or model.replay_buffer is None:
        _logger.warning("demo_path=%r ignored: model has no replay_buffer.", path)
        return

    data = np.load(path)
    actions = data["actions"]
    rewards = data["rewards"]
    dones = data["dones"]
    truncateds = data["truncateds"] if "truncateds" in data.files else np.zeros_like(dones)
    n_steps, n_envs_demo = rewards.shape
    if n_envs_demo != vec_env.num_envs:
        raise ValueError(
            f"demo file num_envs={n_envs_demo} does not match training num_envs="
            f"{vec_env.num_envs}; collect demos with --num-envs={vec_env.num_envs}."
        )

    her_enabled = agent_cfg.her.enabled
    if her_enabled:
        obs_keys = ("observation", "achieved_goal", "desired_goal")
        obs = {k: data[f"obs_{k}"] for k in obs_keys}
        next_obs = {k: data[f"next_obs_{k}"] for k in obs_keys}
    else:
        obs = {"_": data["obs_observation"]}
        next_obs = {"_": data["next_obs_observation"]}

    _logger.info(
        "pre-filling replay buffer with %d steps x %d envs = %d transitions from %s",
        n_steps,
        n_envs_demo,
        n_steps * n_envs_demo,
        path,
    )

    for t in range(n_steps):
        obs_t = {k: v[t] for k, v in obs.items()} if her_enabled else obs["_"][t]
        next_obs_t = {k: v[t] for k, v in next_obs.items()} if her_enabled else next_obs["_"][t]
        # SB3 expects info dicts to carry ``TimeLimit.truncated`` so the replay
        # buffer can distinguish termination from truncation when bootstrapping.
        # Episode-end metadata (``episode`` / ``is_success``) isn't read by add,
        # so the minimal dict is enough.
        infos = [
            {"TimeLimit.truncated": bool(truncateds[t, i] and not dones[t, i])}
            for i in range(n_envs_demo)
        ]
        model.replay_buffer.add(
            obs_t,
            next_obs_t,
            actions[t],
            rewards[t],
            dones[t],
            infos,
        )


class _PlayAdapter:
    """Presents a ``GenelabSb3VecEnv`` to ``runner.run_play_loop`` as a 4-tuple env.

    ``run_play_loop`` expects ``reset() -> (obs, info)`` and ``step() -> (obs, ...)``;
    SB3's ``VecEnv.reset`` returns only ``obs`` and ``step`` returns a 4-tuple.
    """

    def __init__(self, vec_env: Any) -> None:
        self._vec_env = vec_env

    def reset(self) -> tuple[Any, dict[str, Any]]:
        return self._vec_env.reset(), {}

    def step(self, actions: Any) -> tuple[Any, Any, Any, Any]:
        return self._vec_env.step(actions)


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

        if ctx.resume_from is None:
            _maybe_prefill_demos(model, agent_cfg, vec_env)

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

    def play(self, ctx: PlayContext) -> None:
        env = ctx.env
        agent_cfg = ctx.agent_cfg
        cfg = agent_cfg if isinstance(agent_cfg, Sb3AgentCfg) else Sb3AgentCfg()
        vec_env = _make_vec_env(env, cfg)
        device = str(vec_env.device)
        num_actions = env.action_manager.total_action_dim

        policy: Any
        if ctx.kind == "zero":
            policy = make_zero_policy(env.num_envs, num_actions, vec_env.device)
        elif ctx.kind == "random":
            policy = make_random_policy(env.num_envs, num_actions, vec_env.device)
        else:
            assert ctx.kind == "trained"
            if ctx.checkpoint is None:
                raise SystemExit("agent='trained' requires a --checkpoint path")
            if not isinstance(agent_cfg, Sb3AgentCfg):
                raise ValueError(
                    f"task {ctx.task_id!r} did not register an SB3 agent cfg; "
                    "pass agent_cfg explicitly"
                )
            model = _algo_class(agent_cfg.algorithm).load(
                str(ctx.checkpoint), env=vec_env, device=device
            )
            deterministic = ctx.deterministic

            def _policy(obs: Any) -> Any:
                actions, _ = model.predict(obs, deterministic=deterministic)
                return actions

            policy = _policy

        adapter = _PlayAdapter(vec_env)
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


register_backend(Sb3Backend())
