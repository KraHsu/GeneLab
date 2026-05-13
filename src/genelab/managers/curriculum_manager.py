"""Curriculum manager: per-term hooks called on reset to mutate env config."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from genelab.managers._base import instantiate_class_term
from genelab.managers.manager_term_cfg import ManagerTermBaseCfg

if TYPE_CHECKING:
    from genelab.envs.manager_based_rl_env import ManagerBasedRlEnv


@dataclass
class CurriculumTermCfg(ManagerTermBaseCfg):
    pass


class CurriculumManager:
    def __init__(
        self,
        cfg: dict[str, CurriculumTermCfg],
        env: "ManagerBasedRlEnv",
    ) -> None:
        self._env = env
        self.cfg: dict[str, CurriculumTermCfg] = deepcopy(cfg)
        self._term_names: list[str] = []
        self._term_cfgs: list[CurriculumTermCfg] = []
        for name, term_cfg in self.cfg.items():
            instantiate_class_term(term_cfg, env)
            self._term_names.append(name)
            self._term_cfgs.append(term_cfg)

    @property
    def active_terms(self) -> list[str]:
        return list(self._term_names)

    def compute(self, env_ids: torch.Tensor | slice | None = None) -> dict[str, float]:
        extras: dict[str, float] = {}
        for name, term_cfg in zip(self._term_names, self._term_cfgs, strict=True):
            value = term_cfg.func(self._env, env_ids, **term_cfg.params)
            if isinstance(value, torch.Tensor):
                extras[f"Curriculum/{name}"] = float(value.float().mean().item())
            elif value is not None:
                extras[f"Curriculum/{name}"] = float(value)
        return extras
