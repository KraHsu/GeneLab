"""Action post-processing for deploy (numpy port of JointPositionOffsetEMAAction).

The policy emits raw actions ~[-1, 1]; the joint target is
``default + action_scale * clamp(action)``, clamped to joint limits, EMA-smoothed
against the previous target, and held at the default pose for ``warmup_steps`` after
each reset. Training-only terms (encoder_bias, action noise) are dropped.
"""

from __future__ import annotations

import numpy as np


class ActionProcessor:
    """Turn a raw policy action into a smoothed, limit-clamped joint target."""

    def __init__(
        self,
        default_joint_pos: np.ndarray,
        action_scale: float = 0.5,
        ema_alpha: float = 0.5,
        warmup_steps: int = 8,
        joint_pos_limits: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> None:
        self._default = np.asarray(default_joint_pos, dtype=float)
        self.action_scale = action_scale
        self.ema_alpha = ema_alpha
        self.warmup_steps = warmup_steps
        if joint_pos_limits is None:
            self._lo = np.full_like(self._default, -np.inf)
            self._hi = np.full_like(self._default, np.inf)
        else:
            self._lo = np.asarray(joint_pos_limits[0], dtype=float)
            self._hi = np.asarray(joint_pos_limits[1], dtype=float)
        self._prev_target = self._default.copy()
        self._step = 0

    def reset(self) -> None:
        """Reset the EMA state and warmup counter (call on episode boundaries)."""
        self._prev_target = self._default.copy()
        self._step = 0

    def process(self, action: np.ndarray) -> np.ndarray:
        """Return the joint target for this control step (JOINT_NAMES_20 order)."""
        action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        raw_target = self._default + self.action_scale * action
        raw_target = np.clip(raw_target, self._lo, self._hi)
        smoothed = self.ema_alpha * raw_target + (1.0 - self.ema_alpha) * self._prev_target
        target = self._default.copy() if self._step < self.warmup_steps else smoothed
        self._prev_target = target.copy()
        self._step += 1
        return target
