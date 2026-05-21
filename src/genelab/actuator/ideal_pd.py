"""Ideal PD actuator — Python-side PD with hard effort clipping.

``tau = clip(kp * (q* - q) - kv * q_dot, -effort_limit, +effort_limit)``

Bind-time the simulator's internal PD gains are forced to zero so the controller is fully
external. Use this when the policy must respond to a configurable effort limit, or as a
prerequisite for :class:`DCMotorActuator`'s velocity-dependent saturation.
"""

from dataclasses import dataclass
from typing import Any, Literal

import torch

from genelab.actuator.actuator_base import ActuatorBase, ActuatorBaseCfg


@dataclass
class IdealPDActuatorCfg(ActuatorBaseCfg):
    """Configuration for :class:`IdealPDActuator`."""

    def __post_init__(self) -> None:
        if self.class_type is None:
            self.class_type = IdealPDActuator


class IdealPDActuator(ActuatorBase):
    """Joint group whose effort is computed in Python and pushed via ``control_dofs_force``."""

    channel: Literal["implicit_pd", "force"] = "force"

    def initialize(self, gs_handle: Any) -> None:
        """Zero the simulator-side PD gains, then publish the static actuator parameters."""
        super().initialize(gs_handle)
        zeros = torch.zeros_like(self._stiffness)
        self._write_pd_gains(gs_handle, kp_values=zeros, kv_values=zeros)

    def compute(
        self,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
        target_pos: torch.Tensor,
        target_vel: torch.Tensor | None = None,
    ) -> torch.Tensor:
        kp = self._stiffness.unsqueeze(0)
        kv = self._damping.unsqueeze(0)
        tau = kp * (target_pos - joint_pos) - kv * joint_vel
        if self._effort_limit is not None:
            lim = self._effort_limit.unsqueeze(0)
            tau = torch.clamp(tau, min=-lim, max=lim)
        return tau
