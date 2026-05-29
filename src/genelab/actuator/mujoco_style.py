"""Mujoco-style actuator cfg — convenience translator over :class:`IdealPDActuator`.

Mujoco's general actuator transfer function is

.. code-block:: text

    force = gear * (gain_prm[0] * ctrl
                    - bias_prm[0] - bias_prm[1] * qpos - bias_prm[2] * qvel)

For the most common MJCF actuator shape — ``dyntype=NONE``, ``gaintype=FIXED``,
``biastype=AFFINE``, ``gain_prm[0] = 1`` — this reduces to a PD controller:

.. code-block:: text

    kp = -gear * bias_prm[1]
    kv = -gear * bias_prm[2]

with the controller's ``ctrl`` interpreted as a position target.
:class:`MujocoStyleActuatorCfg` accepts those Mujoco parameters and translates
them into the underlying :class:`~genelab.actuator.IdealPDActuator` primitives
so MJCF-defined actuators can be brought into GeneLab without translating the
``(kp, kv, force_range)`` triple by hand.

Genesis 1.0 has no separate runtime "general actuator" class — the MJCF parser
in ``genesis/utils/mjcf.py`` already translates these same Mujoco attributes
into Genesis's ``set_dofs_kp`` / ``set_dofs_kv`` / ``set_dofs_force_range``
setters at load time. This cfg is the GeneLab-side mirror: it exposes the same
named knobs for tasks that configure actuators in Python rather than loading
them from an MJCF.
"""

from dataclasses import dataclass
from typing import Literal

from genelab.actuator.ideal_pd import IdealPDActuator, IdealPDActuatorCfg


@dataclass
class MujocoStyleActuatorCfg(IdealPDActuatorCfg):
    """Cfg for actuators specified in Mujoco's ``(gear, gain_prm, bias_prm)`` form.

    Only the canonical ``dyntype=none / gaintype=fixed / biastype=affine`` shape is
    supported — every other dyntype / gaintype / biastype combination raises a
    ``ValueError`` at construction time (Genesis itself logs a warning and falls
    back to the supported shape when an MJCF declares them, so rejecting at the
    Python cfg layer mirrors that behaviour up front).

    ``bias_prm[0]`` (the constant bias term) must be zero — ``IdealPDActuator``
    has no constant-offset force term, so a non-zero constant cannot be
    represented exactly. ``stiffness`` / ``damping`` are computed from ``gear``
    and ``bias_prm``; setting them directly on this cfg raises.
    """

    # Mujoco's transfer-function knobs.
    gear: float = 1.0
    bias_prm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # Discrete-choice fields kept on the cfg for documentation; the supported
    # combination is the only one allowed today. Drop the ``Literal`` constraint
    # if a future revision needs to expand the supported set.
    dyntype: Literal["none"] = "none"
    gaintype: Literal["fixed"] = "fixed"
    biastype: Literal["affine"] = "affine"

    def __post_init__(self) -> None:
        if self.dyntype != "none":
            raise ValueError(
                f"MujocoStyleActuatorCfg: dyntype={self.dyntype!r} not supported; "
                "only 'none' (no internal actuator dynamics) is implementable on "
                "the existing IdealPDActuator primitives."
            )
        if self.gaintype != "fixed":
            raise ValueError(
                f"MujocoStyleActuatorCfg: gaintype={self.gaintype!r} not supported; "
                "only 'fixed' (gain_prm[0]=1) is implementable on IdealPDActuator."
            )
        if self.biastype != "affine":
            raise ValueError(
                f"MujocoStyleActuatorCfg: biastype={self.biastype!r} not supported; "
                "only 'affine' (kp/kv terms via bias_prm[1] / bias_prm[2]) maps to PD."
            )
        bias_const, bias_kp_term, bias_kv_term = self.bias_prm
        if bias_const != 0.0:
            raise ValueError(
                f"MujocoStyleActuatorCfg: bias_prm[0]={bias_const} (constant bias) "
                "is not representable by IdealPDActuator — there is no constant-offset "
                "force term. Set bias_prm[0]=0."
            )
        if self.stiffness != 0.0 or self.damping != 0.0:
            raise ValueError(
                "MujocoStyleActuatorCfg: stiffness / damping are computed from "
                "gear * bias_prm; don't set them directly on this cfg."
            )
        # PD gains derived from the Mujoco bias parameters; the sign flip turns
        # the +bias_prm[1]*qpos / +bias_prm[2]*qvel terms (which Mujoco *subtracts*
        # from the control output) into the +kp*(target-qpos) / -kv*qvel terms
        # IdealPDActuator computes.
        self.stiffness = -self.gear * bias_kp_term
        self.damping = -self.gear * bias_kv_term
        if self.class_type is None:
            self.class_type = IdealPDActuator
