"""Reward manager: weights and sums reward terms per environment per step."""

from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from genelab.managers._base import instantiate_class_term
from genelab.managers.manager_term_cfg import ManagerTermBaseCfg

if TYPE_CHECKING:
    from genelab.envs.manager_based_rl_env import ManagerBasedRlEnv


@dataclass
class RewardTermCfg(ManagerTermBaseCfg):
    """Reward-specific term config: adds ``weight``."""

    weight: float = 0.0


class RewardManager:
    """Aggregates weighted reward terms.

    When ``scale_by_dt`` is True (default) the per-step reward is multiplied by ``dt`` so
    cumulative episode reward is comparable across simulation frequencies.
    """

    def __init__(
        self,
        cfg: dict[str, RewardTermCfg],
        env: "ManagerBasedRlEnv",
        *,
        scale_by_dt: bool = True,
    ) -> None:
        self._env = env
        self._scale_by_dt = scale_by_dt
        self.cfg: dict[str, RewardTermCfg] = deepcopy(cfg)
        self._term_names: list[str] = []
        self._term_cfgs: list[RewardTermCfg] = []
        for name, term_cfg in self.cfg.items():
            instantiate_class_term(term_cfg, env)
            self._term_names.append(name)
            self._term_cfgs.append(term_cfg)
        self._reward_buf = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self._episode_sums: dict[str, torch.Tensor] = {
            name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for name in self._term_names
        }

    @property
    def num_envs(self) -> int:
        return self._env.num_envs

    @property
    def device(self) -> str:
        return self._env.device

    @property
    def active_terms(self) -> list[str]:
        return list(self._term_names)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> dict[str, float]:
        if env_ids is None:
            env_ids = slice(None)
        max_episode_length_s = getattr(self._env, "max_episode_length_s", 1.0) or 1.0
        names = list(self._episode_sums.keys())
        if not names:
            return {}
        # Stack per-term means into a single tensor and pull all values to host with one sync.
        # The per-term ``.item()`` loop was a per-reset CUDA sync per reward term (6+ per step
        # on busy reset paths in g1-velocity-flat).
        means = torch.stack([torch.mean(self._episode_sums[name][env_ids]) for name in names])
        means_list = (means / max_episode_length_s).tolist()
        for name in names:
            self._episode_sums[name][env_ids] = 0.0
        return {f"Episode_Reward/{name}": value for name, value in zip(names, means_list)}

    def compute(self, dt: float) -> torch.Tensor:
        self._reward_buf.zero_()
        scale = dt if self._scale_by_dt else 1.0
        for name, term_cfg in zip(self._term_names, self._term_cfgs, strict=True):
            if term_cfg.weight == 0.0:
                continue
            value = term_cfg.func(self._env, **term_cfg.params)
            value = value * term_cfg.weight * scale
            value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
            self._reward_buf += value
            self._episode_sums[name] += value
        return self._reward_buf

    def get_term_cfg(self, term_name: str) -> RewardTermCfg:
        return self._term_cfgs[self._term_names.index(term_name)]
