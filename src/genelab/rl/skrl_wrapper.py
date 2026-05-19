"""Adapter from ``ManagerBasedRlEnv`` to the skrl vectorized-env interface.

skrl agents consume a *flat* observation tensor ``(num_envs, obs_dim)`` — not the
group dict that ``ManagerBasedRlEnv`` produces — so the wrapper selects one
observation group (``obs_group``, default ``"policy"``) as the agent's observation.
skrl also expects reward / terminated / truncated shaped ``(num_envs, 1)``.

Like ``rsl_rl_wrapper``, this becomes a subclass of skrl's ``Wrapper`` base when
skrl is installed, and degrades to a plain object otherwise so zero/random play
and unit tests work without skrl. ``gymnasium`` is required either way.
"""

from typing import TYPE_CHECKING, Any

import gymnasium
import numpy as np
import torch

if TYPE_CHECKING:
    from genelab.envs.manager_based_rl_env import ManagerBasedRlEnv


def _box(dim: int) -> gymnasium.spaces.Box:
    """An unbounded ``Box`` of width ``dim`` (skrl reads ``.shape`` to size networks)."""
    return gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(dim,), dtype=np.float32)


class GenelabSkrlWrapper:
    """Wraps a ``ManagerBasedRlEnv`` so skrl agents / trainers can drive it."""

    def __init__(
        self,
        env: "ManagerBasedRlEnv",
        *,
        obs_group: str = "policy",
        state_group: str | None = "critic",
    ) -> None:
        self._env = env
        self._unwrapped = env
        self._device = torch.device(env.device)
        self._obs_group = obs_group
        self._state_group = state_group

        obs_dict = env.observation_manager.compute()
        if obs_group not in obs_dict:
            raise KeyError(
                f"observation group {obs_group!r} not produced by the env; "
                f"available groups: {sorted(obs_dict)}"
            )
        self._observation_space = _box(int(obs_dict[obs_group].shape[-1]))
        self._action_space = gymnasium.spaces.Box(
            low=-1.0, high=1.0,
            shape=(env.action_manager.total_action_dim,), dtype=np.float32,
        )
        self._state_space: gymnasium.spaces.Box | None = None
        if state_group is not None and state_group in obs_dict:
            self._state_space = _box(int(obs_dict[state_group].shape[-1]))

    # -- skrl Wrapper surface -------------------------------------------------
    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def num_envs(self) -> int:
        return int(self._env.num_envs)

    @property
    def num_agents(self) -> int:
        return 1

    @property
    def observation_space(self) -> gymnasium.spaces.Box:
        return self._observation_space

    @property
    def action_space(self) -> gymnasium.spaces.Box:
        return self._action_space

    @property
    def state_space(self) -> gymnasium.spaces.Box | None:
        return self._state_space

    @property
    def unwrapped(self) -> "ManagerBasedRlEnv":
        return self._env

    def reset(self) -> tuple[torch.Tensor, dict[str, Any]]:
        obs_dict, extras = self._env.reset()
        return obs_dict[self._obs_group], dict(extras)

    def step(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        obs_dict, rew, terminated, truncated, extras = self._env.step(actions)
        # skrl expects reward / done flags shaped (num_envs, 1).
        reward = rew.view(-1, 1).to(dtype=torch.float32)
        terminated = terminated.view(-1, 1)
        truncated = truncated.view(-1, 1)
        return obs_dict[self._obs_group], reward, terminated, truncated, dict(extras)

    def state(self) -> torch.Tensor | None:
        if self._state_group is None:
            return None
        return self._env.observation_manager.compute().get(self._state_group)

    def render(self, *args: Any, **kwargs: Any) -> None:
        return None

    def close(self) -> None:
        self._env.close()


def _attach_skrl_base() -> None:
    """Make ``GenelabSkrlWrapper`` subclass skrl's ``Wrapper`` when skrl is installed."""
    try:
        from skrl.envs.wrappers.torch import Wrapper as _SkrlWrapper
    except ImportError:
        return
    global GenelabSkrlWrapper

    class _GenelabSkrlWrapper(GenelabSkrlWrapper, _SkrlWrapper):  # type: ignore[misc]
        pass

    _GenelabSkrlWrapper.__name__ = "GenelabSkrlWrapper"
    _GenelabSkrlWrapper.__qualname__ = "GenelabSkrlWrapper"
    GenelabSkrlWrapper = _GenelabSkrlWrapper  # type: ignore[misc]


_attach_skrl_base()
