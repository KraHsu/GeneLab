"""Domain-randomization event functions.

Thin wrappers over the per-env Genesis mutation API
(``set_friction_ratio`` / ``set_mass_shift`` / ``set_COM_shift``) plus the
genelab-native ``encoder_bias`` that lives in :class:`RobotState` and gets
subtracted from the PD target on the action side of the loop. Each function follows the
:class:`~genelab.managers.event_manager.EventTermCfg` calling convention so
configs hook them in with ``mode="startup"`` (constant DR sampled once per
episode start) — Isaac Lab's standard pattern for sim2real perturbations.

The Genesis mutation calls are wrapped in ``getattr``/``try`` so the same
functions stay usable from the fake-env scaffolding the test suite uses;
calls without a real Genesis backend silently no-op rather than raising.
"""

from genelab.mdp.dr.actuator import randomize_actuator_deadzone
from genelab.mdp.dr.body import body_com_offset, body_mass_offset
from genelab.mdp.dr.geom import geom_friction
from genelab.mdp.dr.gravity import gravity_tilt
from genelab.mdp.dr.joint import (
    dof_armature,
    dof_frictionloss,
    encoder_bias,
    randomize_joint_stiffness_damping,
)

__all__ = [
    "body_com_offset",
    "body_mass_offset",
    "dof_armature",
    "dof_frictionloss",
    "encoder_bias",
    "geom_friction",
    "gravity_tilt",
    "randomize_actuator_deadzone",
    "randomize_joint_stiffness_damping",
]
