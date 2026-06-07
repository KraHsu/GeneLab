"""Implicit PD actuator — delegates the PD law to Genesis.

The simulator computes ``tau = kp * (q* - q) - kv * q_dot`` internally each substep, so the
articulation only has to push the position target via ``control_dofs_position``. Numerically
equivalent to the pre-M2 behaviour where ``Articulation`` called ``set_dofs_kp / kv``
directly.

Use this actuator when the underlying Genesis engine's stable implicit-Euler PD integration
is preferable to a Python-side explicit force computation (high stiffness, small dt).
"""

from dataclasses import dataclass
from typing import Any, Literal

from genelab.actuator.actuator_base import ActuatorBase, ActuatorBaseCfg


@dataclass
class ImplicitPDActuatorCfg(ActuatorBaseCfg):
    """Configuration for :class:`ImplicitPDActuator`."""

    def __post_init__(self) -> None:
        if self.class_type is None:
            self.class_type = ImplicitPDActuator


class ImplicitPDActuator(ActuatorBase):
    """Joint group whose PD is solved by Genesis itself.

    :meth:`compute` returns ``None`` to signal the articulation to use the position-target
    control channel instead of the force channel.
    """

    channel: Literal["implicit_pd", "force", "velocity"] = "implicit_pd"

    def initialize(self, gs_handle: Any) -> None:
        """Write ``kp`` / ``kv`` / ``force_range`` / ``armature`` / ``friction`` to Genesis."""
        super().initialize(gs_handle)
        self._write_pd_gains(gs_handle, kp_values=self._stiffness, kv_values=self._damping)

    def compute(
        self,
        joint_pos: Any,
        joint_vel: Any,
        target_pos: Any,
        target_vel: Any = None,
    ) -> None:
        return None
