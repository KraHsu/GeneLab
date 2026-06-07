"""Implicit velocity actuator — Genesis solves velocity tracking internally.

The joint group is driven by ``control_dofs_velocity``: Genesis applies torque to track a
commanded joint *velocity* using the ``kv`` (damping) gain, with ``kp`` forced to 0 so there
is no position hold. Used for continuously-rotating joints — e.g. the Unitree Go2-W wheels —
where a position target is meaningless (it would fight the spin). :meth:`compute` returns
``None`` (the simulator solves the law); the velocity target reaches Genesis through
``Articulation.write_joint_velocity_targets_partial`` (driven by ``mdp.JointVelocityAction``).
"""

from dataclasses import dataclass
from typing import Any, Literal

import torch

from genelab.actuator.actuator_base import ActuatorBase, ActuatorBaseCfg


@dataclass
class ImplicitVelocityActuatorCfg(ActuatorBaseCfg):
    """Configuration for :class:`ImplicitVelocityActuator`.

    ``damping`` is the velocity-tracking gain (Genesis ``kv``); ``stiffness`` is ignored
    (forced to 0). ``effort_limit`` caps the torque Genesis may apply to hit the target.
    """

    def __post_init__(self) -> None:
        if self.class_type is None:
            self.class_type = ImplicitVelocityActuator


class ImplicitVelocityActuator(ActuatorBase):
    """Joint group whose velocity is tracked by Genesis via ``control_dofs_velocity``."""

    channel: Literal["implicit_pd", "force", "velocity"] = "velocity"

    def initialize(self, gs_handle: Any) -> None:
        """Write ``kp=0`` / ``kv=damping`` / ``force_range`` / ``armature`` / ``friction``."""
        super().initialize(gs_handle)
        self._write_pd_gains(
            gs_handle, kp_values=torch.zeros_like(self._damping), kv_values=self._damping
        )

    def compute(
        self,
        joint_pos: Any,
        joint_vel: Any,
        target_pos: Any,
        target_vel: Any = None,
    ) -> None:
        return None
