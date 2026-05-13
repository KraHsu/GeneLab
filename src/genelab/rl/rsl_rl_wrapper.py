"""Adapter from ``ManagerBasedRlEnv`` to the RSL-RL VecEnv interface."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from genelab.envs.manager_based_rl_env import ManagerBasedRlEnv


def _maybe_tensordict(obs: dict[str, torch.Tensor], num_envs: int) -> Any:
    """Convert obs dict to TensorDict if tensordict is installed; otherwise return plain dict."""
    try:
        from tensordict import TensorDict
    except ImportError:
        return obs
    return TensorDict(obs, batch_size=[num_envs])


class RslRlVecEnvWrapper:
    """Wraps a ``ManagerBasedRlEnv`` so it can be driven by ``rsl_rl.runners.OnPolicyRunner``.

    Subclass of ``rsl_rl.env.VecEnv`` when that import succeeds; otherwise falls back to a plain
    object exposing the same attribute set (so unit tests can poke at this class without RSL-RL).
    """

    def __init__(
        self,
        env: "ManagerBasedRlEnv",
        clip_actions: float | None = None,
    ) -> None:
        self.env = env
        self.clip_actions = clip_actions
        self.num_envs = env.num_envs
        self.device = torch.device(env.device)
        self.max_episode_length = env.max_episode_length
        self.num_actions = env.action_manager.total_action_dim
        # Pre-warm an observation pass so ``num_obs`` / ``num_privileged_obs`` are populated.
        obs_dict = env.observation_manager.compute()
        self.num_obs = int(obs_dict.get("policy", torch.zeros(env.num_envs, 0)).shape[-1])
        self.num_privileged_obs = int(
            obs_dict.get("critic", torch.zeros(env.num_envs, 0)).shape[-1]
        )

    @property
    def cfg(self) -> Any:
        return self.env.cfg

    @property
    def unwrapped(self) -> "ManagerBasedRlEnv":
        return self.env

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self.env.episode_length_buf

    @classmethod
    def class_name(cls) -> str:
        return cls.__name__

    def seed(self, seed: int = -1) -> int:
        return seed

    def get_observations(self) -> Any:
        obs_dict = self.env.observation_manager.compute()
        return _maybe_tensordict(obs_dict, self.num_envs)

    def reset(self) -> tuple[Any, dict[str, Any]]:
        obs_dict, extras = self.env.reset()
        return _maybe_tensordict(obs_dict, self.num_envs), extras

    def step(self, actions: torch.Tensor) -> tuple[Any, torch.Tensor, torch.Tensor, dict[str, Any]]:
        if self.clip_actions is not None:
            actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        obs_dict, rew, terminated, truncated, extras = self.env.step(actions)
        dones = (terminated | truncated).to(dtype=torch.long)
        extras = dict(extras)
        extras["time_outs"] = truncated
        return _maybe_tensordict(obs_dict, self.num_envs), rew, dones, extras

    def close(self) -> None:
        self.env.close()


def _attach_rsl_rl_base() -> None:
    """Make ``RslRlVecEnvWrapper`` a subclass of ``rsl_rl.env.VecEnv`` if RSL-RL is installed."""
    try:
        from rsl_rl.env import VecEnv as _RslVecEnv
    except ImportError:
        return
    global RslRlVecEnvWrapper

    class _RslRlVecEnvWrapper(RslRlVecEnvWrapper, _RslVecEnv):  # type: ignore[misc]
        pass

    _RslRlVecEnvWrapper.__name__ = "RslRlVecEnvWrapper"
    _RslRlVecEnvWrapper.__qualname__ = "RslRlVecEnvWrapper"
    RslRlVecEnvWrapper = _RslRlVecEnvWrapper  # type: ignore[misc]


_attach_rsl_rl_base()
