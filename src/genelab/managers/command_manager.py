"""Command manager: per-term command generators with periodic resampling."""

from __future__ import annotations

import abc
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from genelab.managers.manager_term_cfg import ManagerTermBaseCfg

if TYPE_CHECKING:
    from genelab.envs.manager_based_rl_env import ManagerBasedRlEnv


@dataclass
class CommandTermCfg(ManagerTermBaseCfg):
    class_type: type["CommandTerm"] | None = None
    resampling_time_range: tuple[float, float] = (5.0, 5.0)
    asset_name: str = "robot"
    debug_vis: bool = False


class CommandTerm(abc.ABC):
    def __init__(self, cfg: CommandTermCfg, env: "ManagerBasedRlEnv") -> None:
        self.cfg = cfg
        self._env = env
        self._time_left = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

    @property
    def num_envs(self) -> int:
        return self._env.num_envs

    @property
    def device(self) -> str:
        return self._env.device

    @property
    @abc.abstractmethod
    def command(self) -> torch.Tensor:
        ...

    @abc.abstractmethod
    def _resample_command(self, env_ids: torch.Tensor) -> None:
        ...

    def _update_command(self) -> None:
        """Optional hook called every step (e.g., heading-based velocity rewrite)."""

    def compute(self, dt: float) -> None:
        self._time_left -= dt
        resample_ids = (self._time_left <= 0.0).nonzero(as_tuple=False).flatten()
        if resample_ids.numel() > 0:
            self._resample(resample_ids)
        self._update_command()

    def _resample(self, env_ids: torch.Tensor) -> None:
        low, high = self.cfg.resampling_time_range
        self._time_left[env_ids] = torch.empty_like(self._time_left[env_ids]).uniform_(low, high)
        self._resample_command(env_ids)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        elif isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        if env_ids.numel() == 0:
            return
        self._resample(env_ids)


class CommandManager:
    def __init__(
        self,
        cfg: dict[str, CommandTermCfg],
        env: "ManagerBasedRlEnv",
    ) -> None:
        self._env = env
        self.cfg: dict[str, CommandTermCfg] = deepcopy(cfg)
        self._terms: dict[str, CommandTerm] = {}
        for name, term_cfg in self.cfg.items():
            if term_cfg.class_type is None:
                continue
            self._terms[name] = term_cfg.class_type(term_cfg, env)

    @property
    def num_envs(self) -> int:
        return self._env.num_envs

    @property
    def device(self) -> str:
        return self._env.device

    @property
    def active_terms(self) -> list[str]:
        return list(self._terms.keys())

    def get_command(self, name: str) -> torch.Tensor:
        return self._terms[name].command

    def compute(self, dt: float) -> None:
        for term in self._terms.values():
            term.compute(dt)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> dict[str, float]:
        for term in self._terms.values():
            term.reset(env_ids)
        return {}
