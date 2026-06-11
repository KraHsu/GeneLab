"""Action manager: routes a flat action tensor to per-term controllers."""

import abc
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from genelab.managers.manager_term_cfg import ManagerTermBaseCfg

if TYPE_CHECKING:
    from genelab.contracts import EnvContext


@dataclass
class ActionTermCfg(ManagerTermBaseCfg):
    """Action term config. ``class_type`` is the ``ActionTerm`` subclass to instantiate."""

    class_type: type["ActionTerm"] | None = None
    asset_name: str = "robot"


class ActionTerm(abc.ABC):
    """Base class for action terms. Subclasses own a slice of the action vector."""

    def __init__(self, cfg: ActionTermCfg, env: "EnvContext") -> None:
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
        env: "EnvContext",
        action_noise_std: float = 0.0,
        action_delay_steps: tuple[int, int] = (0, 0),
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
        # Second-order history slot so ``mean_action_acc`` and similar metrics can compute a
        # finite-difference acceleration ``a_t − 2·a_{t-1} + a_{t-2}``. Two extra (B, A)
        # buffers are cheap; Isaac Lab keeps the same pair on its ActionManager.
        self._prev_prev_action: torch.Tensor = torch.zeros_like(self._action)

        # --- training-only sim2real action perturbation (never exported) ---
        self._action_noise_std = float(action_noise_std)
        lo, hi = int(action_delay_steps[0]), int(action_delay_steps[1])
        self._max_delay = max(0, hi)
        self._delay_range = (max(0, lo), self._max_delay)
        if self._max_delay > 0:
            # Ring of the last ``max_delay + 1`` applied commands, newest at index -1, plus a
            # per-env latency (resampled each reset) selecting how stale a command to apply.
            self._delay_buf = torch.zeros(
                (self.num_envs, self._max_delay + 1, self.total_action_dim),
                dtype=torch.float,
                device=self.device,
            )
            self._delay = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

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

    @property
    def prev_prev_action(self) -> torch.Tensor:
        return self._prev_prev_action

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._action[env_ids] = 0.0
        self._prev_action[env_ids] = 0.0
        self._prev_prev_action[env_ids] = 0.0
        if self._max_delay > 0:
            # Clear the latency ring for the reset envs and resample their per-env delay so a
            # fresh episode never applies commands buffered from the previous one.
            self._delay_buf[env_ids] = 0.0
            lo, hi = self._delay_range
            if isinstance(env_ids, slice):
                n = self.num_envs
                self._delay[env_ids] = torch.randint(lo, hi + 1, (n,), device=self.device)
            else:
                n = int(env_ids.numel())
                self._delay[env_ids] = torch.randint(lo, hi + 1, (n,), device=self.device)
        for term in self._terms.values():
            term.reset(env_ids)

    def process_action(self, action: torch.Tensor) -> None:
        """Cache and slice a fresh policy action across the per-term controllers."""
        # Shift through the three slots so the new value lands in ``action`` while the
        # previous two stay one and two steps behind respectively. The cached ``_action`` is
        # the *raw* policy output — it feeds ``last_action`` obs and action-rate metrics, which
        # must reflect what the network emitted, not the perturbed wire command.
        self._prev_prev_action[:] = self._prev_action
        self._prev_action[:] = self._action
        self._action[:] = action
        applied = self._perturb_action(action)
        offset = 0
        for term in self._terms.values():
            dim = term.action_dim
            term.process_actions(applied[:, offset : offset + dim])
            offset += dim

    def _perturb_action(self, action: torch.Tensor) -> torch.Tensor:
        """Apply training-only latency + Gaussian noise; identity when both are disabled."""
        applied = action
        if self._max_delay > 0:
            # Roll the ring one step and drop the newest command in.
            self._delay_buf = torch.cat([self._delay_buf[:, 1:], action.unsqueeze(1)], dim=1)
            # Per-env pick: delay ``d`` -> the command from ``d`` steps ago = ``max_delay - d``.
            idx = (self._max_delay - self._delay).view(-1, 1, 1).expand(-1, 1, action.shape[-1])
            applied = self._delay_buf.gather(1, idx).squeeze(1)
        if self._action_noise_std > 0.0:
            applied = applied + torch.randn_like(applied) * self._action_noise_std
        return applied

    def apply_action(self) -> None:
        for term in self._terms.values():
            term.apply_actions()
