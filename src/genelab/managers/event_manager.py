"""Event manager: startup / reset / interval-scheduled side effects."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch

from genelab.managers._base import instantiate_class_term
from genelab.managers.manager_term_cfg import ManagerTermBaseCfg

if TYPE_CHECKING:
    from genelab.envs.manager_based_rl_env import ManagerBasedRlEnv

EventMode = Literal["startup", "reset", "interval"]


@dataclass
class EventTermCfg(ManagerTermBaseCfg):
    mode: EventMode = "reset"
    interval_range_s: tuple[float, float] | None = None
    is_global_time: bool = False


class EventManager:
    def __init__(
        self,
        cfg: dict[str, EventTermCfg],
        env: "ManagerBasedRlEnv",
    ) -> None:
        self._env = env
        self.cfg: dict[str, EventTermCfg] = deepcopy(cfg)
        self._terms_by_mode: dict[EventMode, list[tuple[str, EventTermCfg]]] = {
            "startup": [],
            "reset": [],
            "interval": [],
        }
        for name, term_cfg in self.cfg.items():
            instantiate_class_term(term_cfg, env)
            self._terms_by_mode[term_cfg.mode].append((name, term_cfg))
        self._interval_time_left: dict[str, torch.Tensor] = {}
        for name, term_cfg in self._terms_by_mode["interval"]:
            assert term_cfg.interval_range_s is not None, (
                f"interval event '{name}' must define interval_range_s"
            )
            low, high = term_cfg.interval_range_s
            self._interval_time_left[name] = torch.empty(
                self.num_envs, dtype=torch.float, device=self.device
            ).uniform_(low, high)

    @property
    def num_envs(self) -> int:
        return self._env.num_envs

    @property
    def device(self) -> str:
        return self._env.device

    @property
    def active_terms(self) -> dict[str, list[str]]:
        return {mode: [n for n, _ in terms] for mode, terms in self._terms_by_mode.items()}

    def apply(
        self,
        mode: EventMode,
        env_ids: torch.Tensor | slice | None = None,
        dt: float = 0.0,
    ) -> None:
        if mode in ("startup", "reset"):
            for _, term_cfg in self._terms_by_mode[mode]:
                term_cfg.func(self._env, env_ids, **term_cfg.params)
            return
        # interval
        for name, term_cfg in self._terms_by_mode["interval"]:
            time_left = self._interval_time_left[name]
            time_left -= dt
            fire = (time_left <= 0.0).nonzero(as_tuple=False).flatten()
            if fire.numel() > 0:
                term_cfg.func(self._env, fire, **term_cfg.params)
                assert term_cfg.interval_range_s is not None
                low, high = term_cfg.interval_range_s
                time_left[fire] = torch.empty_like(time_left[fire]).uniform_(low, high)
