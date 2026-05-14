"""Actuator abstractions for Genesis-backed robotics environments.

Joint-group electromechanical models composed onto :class:`genelab.entity.Articulation`.
Each actuator owns a regex-matched subset of an articulation's actuated joints and decides
how to drive them through Genesis: either via the simulator's implicit PD law
(:class:`ImplicitPDActuator`), or by computing torque in Python and pushing it through the
force channel (:class:`IdealPDActuator`, :class:`DCMotorActuator`).
"""

from genelab.actuator.actuator_base import ActuatorBase, ActuatorBaseCfg
from genelab.actuator.dc_motor import DCMotorActuator, DCMotorActuatorCfg
from genelab.actuator.ideal_pd import IdealPDActuator, IdealPDActuatorCfg
from genelab.actuator.implicit_pd import ImplicitPDActuator, ImplicitPDActuatorCfg

__all__ = [
    "ActuatorBase",
    "ActuatorBaseCfg",
    "DCMotorActuator",
    "DCMotorActuatorCfg",
    "IdealPDActuator",
    "IdealPDActuatorCfg",
    "ImplicitPDActuator",
    "ImplicitPDActuatorCfg",
]
