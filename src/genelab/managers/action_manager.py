"""Action manager: routes a flat action tensor to per-term controllers."""

import abc
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from genelab.managers.manager_term_cfg import ManagerTermBaseCfg

if TYPE_CHECKING:
    from genelab.envs.manager_based_rl_env import ManagerBasedRlEnv


@dataclass
class ActionTermCfg(ManagerTermBaseCfg):
    """Action term config. ``class_type`` is the ``ActionTerm`` subclass to instantiate."""

    class_type: type["ActionTerm"] | None = None
    asset_name: str = "robot"


class ActionTerm(abc.ABC):
    """Base class for action terms. Subclasses own a slice of the action vector."""

    def __init__(self, cfg: ActionTermCfg, env: "ManagerBasedRlEnv") -> None:
        self.cfg = cfg
        self._env = env

    @property
    def num_envs(self) -> int:
        return self._env.num_envs

    @property
    def device(self) -> str:
        return self._env.device

    @property
    @abc.abstractmethod
    def action_dim(self) -> int: ...

    @property
    @abc.abstractmethod
    def raw_actions(self) -> torch.Tensor: ...

    @abc.abstractmethod
    def process_actions(self, actions: torch.Tensor) -> None:
        """Cache the raw policy action for this term."""

    @abc.abstractmethod
    def apply_actions(self) -> None:
        """Push processed actions to the simulator (called every sim step)."""

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        del env_ids


class ActionManager:
    def __init__(
        self,
        cfg: dict[str, ActionTermCfg],
        env: "ManagerBasedRlEnv",
    ) -> None:
        self._env = env
        self.cfg: dict[str, ActionTermCfg] = deepcopy(cfg)
        self._terms: dict[str, ActionTerm] = {}
        for name, term_cfg in self.cfg.items():
            if term_cfg.class_type is None:
                continue
            self._terms[name] = term_cfg.class_type(term_cfg, env)
        self._action: torch.Tensor = torch.zeros(
            (self.num_envs, self.total_action_dim), dtype=torch.float, device=self.device
        )
        self._prev_action: torch.Tensor = torch.zeros_like(self._action)

    @property
    def num_envs(self) -> int:
        return self._env.num_envs

    @property
    def device(self) -> str:
        return self._env.device

    @property
    def active_terms(self) -> list[str]:
        return list(self._terms.keys())

    @property
    def total_action_dim(self) -> int:
        return sum(t.action_dim for t in self._terms.values())

    @property
    def action(self) -> torch.Tensor:
        return self._action

    @property
    def prev_action(self) -> torch.Tensor:
        return self._prev_action

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._action[env_ids] = 0.0
        self._prev_action[env_ids] = 0.0
        for term in self._terms.values():
            term.reset(env_ids)

    def process_action(self, action: torch.Tensor) -> None:
        """Cache and slice a fresh policy action across the per-term controllers."""
        self._prev_action[:] = self._action
        self._action[:] = action
        offset = 0
        for term in self._terms.values():
            dim = term.action_dim
            term.process_actions(action[:, offset : offset + dim])
            offset += dim

    def apply_action(self) -> None:
        for term in self._terms.values():
            term.apply_actions()
