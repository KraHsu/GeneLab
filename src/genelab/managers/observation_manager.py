"""Observation manager: per-group concatenated observation tensors."""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from genelab.managers._base import instantiate_class_term
from genelab.managers.manager_term_cfg import ManagerTermBaseCfg

if TYPE_CHECKING:
    from genelab.envs.manager_based_rl_env import ManagerBasedRlEnv


@dataclass
class ObservationTermCfg(ManagerTermBaseCfg):
    """Observation term: optional scale + clip applied to the raw tensor."""

    scale: float | None = None
    clip: tuple[float, float] | None = None


@dataclass
class ObservationGroupCfg:
    terms: dict[str, ObservationTermCfg] = field(default_factory=dict)
    concatenate_terms: bool = True


class ObservationManager:
    """Computes a dict of group-name → flat tensor each control step."""

    def __init__(
        self,
        cfg: dict[str, ObservationGroupCfg],
        env: "ManagerBasedRlEnv",
    ) -> None:
        self._env = env
        self.cfg: dict[str, ObservationGroupCfg] = deepcopy(cfg)
        self._group_terms: dict[str, list[tuple[str, ObservationTermCfg]]] = {}
        self._group_dims: dict[str, list[int]] = {}
        for group_name, group_cfg in self.cfg.items():
            terms: list[tuple[str, ObservationTermCfg]] = []
            for term_name, term_cfg in group_cfg.terms.items():
                instantiate_class_term(term_cfg, env)
                terms.append((term_name, term_cfg))
            self._group_terms[group_name] = terms

    @property
    def num_envs(self) -> int:
        return self._env.num_envs

    @property
    def device(self) -> str:
        return self._env.device

    @property
    def active_terms(self) -> dict[str, list[str]]:
        return {g: [n for n, _ in terms] for g, terms in self._group_terms.items()}

    def compute(self) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for group_name, terms in self._group_terms.items():
            group_cfg = self.cfg[group_name]
            tensors: list[torch.Tensor] = []
            for _, term_cfg in terms:
                value = term_cfg.func(self._env, **term_cfg.params)
                if value.dim() == 1:
                    value = value.unsqueeze(-1)
                if term_cfg.scale is not None:
                    value = value * term_cfg.scale
                if term_cfg.clip is not None:
                    value = value.clamp(term_cfg.clip[0], term_cfg.clip[1])
                tensors.append(value)
            if not tensors:
                out[group_name] = torch.zeros(self.num_envs, 0, device=self.device)
                continue
            out[group_name] = (
                torch.cat(tensors, dim=-1)
                if group_cfg.concatenate_terms
                else torch.stack(tensors, dim=-1)
            )
        # Cache dimensions for downstream wrappers
        self._group_dims = {g: [int(t.shape[-1]) for t in (out[g],)] for g in out}
        return out

    def group_obs_dim(self, group_name: str) -> int:
        # Lazy compute if not already populated
        if group_name not in self._group_dims:
            self.compute()
        return int(sum(self._group_dims[group_name]))
