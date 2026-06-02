"""Joint-position-offset action with EMA smoothing + startup warmup.

Subclasses :class:`JointPositionAction` to reuse its joint resolution / scale / defaults.
The policy outputs raw actions in roughly ``[-1, 1]``; the target is
``default + action_scale * clamp(action)`` clamped to joint limits, EMA-smoothed against the
previous target, and held at the default pose for ``warmup_time_s`` after each reset.
"""

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from genelab.mdp.actions.joint_position import JointPositionAction, JointPositionActionCfg

if TYPE_CHECKING:
    from genelab.contracts import EnvContext


@dataclass
class JointPositionOffsetEMAActionCfg(JointPositionActionCfg):
    ema_alpha: float = 0.5
    warmup_time_s: float = 0.4
    # Per-step Gaussian action noise (std, in raw-action units) injected before the target is
    # formed. Forces a control policy that doesn't rely on its exact action being executed —
    # a key robustness lever against feedback overfit to one engine's contact response. 0 in eval.
    action_noise_std: float = 0.0

    def __post_init__(self) -> None:
        if self.class_type is None:
            self.class_type = JointPositionOffsetEMAAction


class JointPositionOffsetEMAAction(JointPositionAction):
    cfg: JointPositionOffsetEMAActionCfg  # type: ignore[assignment]

    def __init__(self, cfg: JointPositionOffsetEMAActionCfg, env: "EnvContext") -> None:
        super().__init__(cfg, env)
        limits = self._articulation.joint_pos_limits.index_select(0, self._joint_indices)
        self._lo = limits[:, 0]
        self._hi = limits[:, 1]
        self._prev_target = self._default.unsqueeze(0).repeat(self.num_envs, 1)
        step_dt = float(getattr(env, "step_dt", 0.05)) or 0.05
        self._warmup_steps = max(0, math.ceil(cfg.warmup_time_s / step_dt))

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw[:] = actions
        if self.cfg.action_noise_std > 0.0:
            actions = actions + torch.randn_like(actions) * self.cfg.action_noise_std
        bias = self._robot_state.encoder_bias.index_select(1, self._joint_indices)
        raw_target = (
            self._default.unsqueeze(0) + self._scale.unsqueeze(0) * actions.clamp(-1.0, 1.0) - bias
        )
        raw_target = torch.minimum(torch.maximum(raw_target, self._lo), self._hi)
        smoothed = self.cfg.ema_alpha * raw_target + (1.0 - self.cfg.ema_alpha) * self._prev_target
        in_warmup = (self._env.episode_length_buf < self._warmup_steps).unsqueeze(-1)
        target = torch.where(in_warmup, self._default.unsqueeze(0).expand_as(smoothed), smoothed)
        self._target = target
        self._prev_target = target.detach().clone()

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            self._prev_target[:] = self._default.unsqueeze(0)
        else:
            self._prev_target[env_ids] = self._default.unsqueeze(0)
