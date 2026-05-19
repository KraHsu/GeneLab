"""Adapter from ``ManagerBasedRlEnv`` to the Stable-Baselines3 ``VecEnv`` interface.

SB3 trains through a ``stable_baselines3.common.vec_env.VecEnv`` whose ``step`` /
``reset`` exchange *numpy* arrays, while ``ManagerBasedRlEnv`` is a GPU-resident
torch vectorized env — so this wrapper copies observations / rewards / dones to
CPU numpy every step (a known cost of pairing SB3 with a GPU-vectorized env;
mirrors Isaac Lab's ``Sb3VecEnvWrapper``).

Two observation modes:

* flat — one observation group (``obs_group``) becomes a ``Box`` observation.
* goal-conditioned (HER) — three groups become a ``Dict`` observation with
  ``observation`` / ``achieved_goal`` / ``desired_goal`` keys, and
  ``env_method("compute_reward", ...)`` routes to the task's relabelling reward.

Like ``skrl_wrapper``, this becomes a subclass of SB3's ``VecEnv`` when SB3 is
installed and degrades to a plain object otherwise, so zero/random play and unit
tests work without SB3. ``gymnasium`` is required either way.
"""

from typing import TYPE_CHECKING, Any

import gymnasium
import numpy as np
import torch

if TYPE_CHECKING:
    from genelab.envs.manager_based_rl_env import ManagerBasedRlEnv
    from genelab.rl.sb3_config import Sb3HerCfg


def _box(dim: int) -> gymnasium.spaces.Box:
    """An unbounded ``Box`` of width ``dim`` (SB3 reads ``.shape`` to size networks)."""
    return gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(dim,), dtype=np.float32)


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Detach a torch tensor to a contiguous CPU float32 numpy array."""
    return np.ascontiguousarray(tensor.detach().to("cpu", dtype=torch.float32).numpy())


class GenelabSb3VecEnv:
    """Wraps a ``ManagerBasedRlEnv`` so SB3 algorithms / trainers can drive it."""

    def __init__(
        self,
        env: "ManagerBasedRlEnv",
        *,
        obs_group: str = "policy",
        her_cfg: "Sb3HerCfg | None" = None,
    ) -> None:
        self._env = env
        self._device = torch.device(env.device)
        self._her = her_cfg if her_cfg is not None and her_cfg.enabled else None
        self._obs_group = self._her.obs_group if self._her is not None else obs_group
        self._actions: torch.Tensor | None = None

        obs_dict = env.observation_manager.compute()
        if self._obs_group not in obs_dict:
            raise KeyError(
                f"observation group {self._obs_group!r} not produced by the env; "
                f"available groups: {sorted(obs_dict)}"
            )

        action_space: gymnasium.spaces.Space[Any] = gymnasium.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(env.action_manager.total_action_dim,),
            dtype=np.float32,
        )
        observation_space: gymnasium.spaces.Space[Any]
        if self._her is not None:
            for group in (self._her.achieved_goal_group, self._her.desired_goal_group):
                if group not in obs_dict:
                    raise KeyError(
                        f"HER goal group {group!r} not produced by the env; "
                        f"available groups: {sorted(obs_dict)}"
                    )
            observation_space = gymnasium.spaces.Dict(
                {
                    "observation": _box(int(obs_dict[self._obs_group].shape[-1])),
                    "achieved_goal": _box(int(obs_dict[self._her.achieved_goal_group].shape[-1])),
                    "desired_goal": _box(int(obs_dict[self._her.desired_goal_group].shape[-1])),
                }
            )
        else:
            observation_space = _box(int(obs_dict[self._obs_group].shape[-1]))

        # Attributes SB3's ``VecEnv`` machinery expects (set here so the wrapper
        # works whether or not it is subclassed onto ``VecEnv``).
        num_envs = int(env.num_envs)
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space
        self.render_mode: str | None = None
        self.reset_infos: list[dict[str, Any]] = [{} for _ in range(num_envs)]
        self._seeds: list[int | None] = [None for _ in range(num_envs)]
        self._options: list[dict[str, Any]] = [{} for _ in range(num_envs)]

    # -- observation assembly -------------------------------------------------
    def _build_obs(self, obs_dict: dict[str, torch.Tensor]) -> Any:
        """Map the env's group dict to SB3's numpy observation (flat or goal Dict)."""
        if self._her is None:
            return _to_numpy(obs_dict[self._obs_group])
        return {
            "observation": _to_numpy(obs_dict[self._obs_group]),
            "achieved_goal": _to_numpy(obs_dict[self._her.achieved_goal_group]),
            "desired_goal": _to_numpy(obs_dict[self._her.desired_goal_group]),
        }

    @staticmethod
    def _obs_at(obs: Any, idx: int) -> Any:
        """Slice env ``idx`` out of a flat array or a Dict observation."""
        if isinstance(obs, dict):
            return {key: value[idx] for key, value in obs.items()}
        return obs[idx]

    # -- SB3 VecEnv surface ---------------------------------------------------
    @property
    def device(self) -> torch.device:
        return self._device

    def reset(self) -> Any:
        obs_dict, _ = self._env.reset()
        self.reset_infos = [{} for _ in range(self.num_envs)]
        return self._build_obs(obs_dict)

    def step_async(self, actions: Any) -> None:
        if not isinstance(actions, torch.Tensor):
            actions = torch.as_tensor(np.asarray(actions), dtype=torch.float32)
        self._actions = actions.to(self._device, dtype=torch.float32)

    def step_wait(self) -> tuple[Any, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        assert self._actions is not None, "step_wait called before step_async"
        obs_dict, reward, terminated, truncated, _ = self._env.step(self._actions)

        obs = self._build_obs(obs_dict)
        rewards = _to_numpy(reward.view(-1))
        terminated_np = terminated.view(-1).detach().to("cpu").numpy().astype(bool)
        truncated_np = truncated.view(-1).detach().to("cpu").numpy().astype(bool)
        dones = terminated_np | truncated_np

        # SB3 auto-reset contract: on a done env, ``obs`` already holds the new
        # episode (the env auto-resets internally), and ``terminal_observation``
        # carries the finished episode's last obs. We can't recover the true
        # pre-reset obs after the env's internal auto-reset, so it is approximated
        # by the post-reset obs — the standard caveat for GPU-vectorized SB3 envs.
        infos: list[dict[str, Any]] = []
        for i in range(self.num_envs):
            info: dict[str, Any] = {}
            if dones[i]:
                info["TimeLimit.truncated"] = bool(truncated_np[i] and not terminated_np[i])
                info["terminal_observation"] = self._obs_at(obs, i)
            infos.append(info)
        return obs, rewards, dones, infos

    def step(self, actions: Any) -> tuple[Any, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        self.step_async(actions)
        return self.step_wait()

    def close(self) -> None:
        self._env.close()

    def render(self, *args: Any, **kwargs: Any) -> None:
        return None

    def get_images(self) -> list[Any]:
        return [None] * self.num_envs

    def seed(self, seed: int | None = None) -> list[int | None]:
        return [seed] * self.num_envs

    def get_attr(self, attr_name: str, indices: Any = None) -> list[Any]:
        value = getattr(self, attr_name, None)
        return [value] * len(self._indices(indices))

    def set_attr(self, attr_name: str, value: Any, indices: Any = None) -> None:
        setattr(self, attr_name, value)

    def env_method(
        self, method_name: str, *args: Any, indices: Any = None, **kwargs: Any
    ) -> list[Any]:
        """Dispatch ``env_method`` calls. HER's ``HerReplayBuffer`` calls
        ``compute_reward``; that routes to the task-supplied relabelling reward."""
        count = len(self._indices(indices))
        if method_name == "compute_reward":
            if self._her is None or self._her.compute_reward is None:
                raise RuntimeError(
                    "env_method('compute_reward') requires a HER config with a "
                    "compute_reward function"
                )
            result = self._her.compute_reward(*args, **kwargs)
            return [result] * count
        raise NotImplementedError(f"env_method({method_name!r}) is not supported")

    def env_is_wrapped(self, wrapper_class: Any, indices: Any = None) -> list[bool]:
        return [False] * len(self._indices(indices))

    def _indices(self, indices: Any) -> list[int]:
        if indices is None:
            return list(range(self.num_envs))
        if isinstance(indices, int):
            return [indices]
        return list(indices)


def _attach_sb3_base() -> None:
    """Make ``GenelabSb3VecEnv`` subclass SB3's ``VecEnv`` when SB3 is installed."""
    try:
        from stable_baselines3.common.vec_env import VecEnv as _Sb3VecEnv
    except ImportError:
        return
    global GenelabSb3VecEnv

    class _GenelabSb3VecEnv(GenelabSb3VecEnv, _Sb3VecEnv):  # type: ignore[misc,valid-type]
        pass

    _GenelabSb3VecEnv.__name__ = "GenelabSb3VecEnv"
    _GenelabSb3VecEnv.__qualname__ = "GenelabSb3VecEnv"
    GenelabSb3VecEnv = _GenelabSb3VecEnv  # type: ignore[misc]


_attach_sb3_base()
