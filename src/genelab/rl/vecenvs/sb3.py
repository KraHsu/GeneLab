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

Like ``vecenvs.skrl``, this becomes a subclass of SB3's ``VecEnv`` when SB3 is
installed and degrades to a plain object otherwise, so zero/random play and unit
tests work without SB3. ``gymnasium`` is required either way.
"""

import time
from typing import TYPE_CHECKING, Any

import gymnasium
import numpy as np
import torch

from genelab.rl.vecenvs._attach_base import attach_optional_base

if TYPE_CHECKING:
    from genelab.envs.manager_based_rl_env import ManagerBasedRlEnv
    from genelab.rl.backends.sb3.config import Sb3HerCfg


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
        # Episode bookkeeping for SB3's ep_info_buffer (drives rollout/ep_*
        # metrics + rollout/success_rate). Replaces an external VecMonitor —
        # GPU envs auto-reset internally so we record episode totals before
        # the reset happens.
        self._ep_rew_sum = np.zeros(num_envs, dtype=np.float64)
        self._ep_step = np.zeros(num_envs, dtype=np.int64)
        self._ep_success = np.zeros(num_envs, dtype=bool)
        self._ep_start_time = np.full(num_envs, time.time(), dtype=np.float64)

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

    @property
    def max_episode_length(self) -> int:
        """Per-episode step budget — used to size HER's ``learning_starts``."""
        return int(self._env.max_episode_length)

    def reset(self) -> Any:
        obs_dict, _ = self._env.reset()
        self.reset_infos = [{} for _ in range(self.num_envs)]
        self._ep_rew_sum[:] = 0.0
        self._ep_step[:] = 0
        self._ep_success[:] = False
        self._ep_start_time[:] = time.time()
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

        # Episode bookkeeping: track per-env running totals; on done, emit a
        # VecMonitor-shaped ``info["episode"]`` so SB3's ep_info_buffer picks
        # up rollout/ep_rew_mean / ep_len_mean. ``is_success`` is computed from
        # achieved-vs-desired goal distance against the HER threshold (when HER
        # is configured) so success_rate aggregates into the same buffer.
        self._ep_rew_sum += rewards
        self._ep_step += 1
        per_env_success = self._episode_success_flags(obs, terminated_np, truncated_np)
        self._ep_success |= per_env_success

        # SB3 auto-reset contract: on a done env, ``obs`` already holds the new
        # episode (the env auto-resets internally), and ``terminal_observation``
        # carries the finished episode's last obs. We can't recover the true
        # pre-reset obs after the env's internal auto-reset, so it is approximated
        # by the post-reset obs — the standard caveat for GPU-vectorized SB3 envs.
        now = time.time()
        infos: list[dict[str, Any]] = []
        for i in range(self.num_envs):
            info: dict[str, Any] = {}
            if dones[i]:
                info["TimeLimit.truncated"] = bool(truncated_np[i] and not terminated_np[i])
                info["terminal_observation"] = self._obs_at(obs, i)
                info["episode"] = {
                    "r": float(self._ep_rew_sum[i]),
                    "l": int(self._ep_step[i]),
                    "t": float(now - self._ep_start_time[i]),
                }
                info["is_success"] = bool(self._ep_success[i])
                self._ep_rew_sum[i] = 0.0
                self._ep_step[i] = 0
                self._ep_success[i] = False
                self._ep_start_time[i] = now
            infos.append(info)
        return obs, rewards, dones, infos

    def _episode_success_flags(
        self, obs: Any, terminated: np.ndarray, truncated: np.ndarray
    ) -> np.ndarray:
        """Per-env ``is_success`` for the current step. ``True`` while the
        achieved goal is within HER's distance threshold of the desired goal —
        the same condition the sparse reward + HER ``compute_reward`` use, so
        an episode's ``info["is_success"]`` (latched across the episode) lines
        up with what the policy is being graded on. Returns all-False when HER
        is not configured (no goals exposed)."""
        if self._her is None or not isinstance(obs, dict):
            return np.zeros(self.num_envs, dtype=bool)
        achieved = obs.get("achieved_goal")
        desired = obs.get("desired_goal")
        if achieved is None or desired is None:
            return np.zeros(self.num_envs, dtype=bool)
        distance = np.linalg.norm(np.asarray(achieved) - np.asarray(desired), axis=-1)
        threshold = self._her_distance_threshold()
        return distance < threshold

    def _her_distance_threshold(self) -> float:
        """Recover the HER distance threshold by probing ``compute_reward`` at
        identical achieved/desired goals — the sparse reward is ``0`` inside the
        threshold and ``-1`` outside, so the perturbation that flips the sign
        reveals the threshold. Cached after the first probe."""
        cached = getattr(self, "_her_threshold_cache", None)
        if cached is not None:
            return cached
        # Fall back to a reasonable default if probing fails or HER is not set.
        if self._her is None or self._her.compute_reward is None:
            self._her_threshold_cache = 0.05
            return self._her_threshold_cache
        try:
            zero = np.zeros((1, 3), dtype=np.float32)
            lo, hi = 0.0, 1.0
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                probe = np.array([[mid, 0.0, 0.0]], dtype=np.float32)
                r = float(np.asarray(self._her.compute_reward(zero, probe, None)).reshape(-1)[0])
                if r >= 0.0:
                    lo = mid
                else:
                    hi = mid
            self._her_threshold_cache = 0.5 * (lo + hi)
        except Exception:
            self._her_threshold_cache = 0.05
        return self._her_threshold_cache

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


attach_optional_base(
    base_module="stable_baselines3.common.vec_env",
    base_attr="VecEnv",
    wrapper_name="GenelabSb3VecEnv",
    caller_globals=globals(),
)
