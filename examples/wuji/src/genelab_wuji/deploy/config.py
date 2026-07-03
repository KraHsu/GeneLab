"""Shared deploy constants: joint ordering, remap, and home pose.

Two DIFFERENT joint orders are in play and MUST be remapped between (this was the
real-hand 0%-success bug):

* **Encoder / hardware order** = ``JOINT_NAMES_20`` = ``REORIENT_JOINT_POS`` keys =
  ``wujihandpy``'s (5, 4) row-major flatten: **finger-major** (finger1_joint1..4,
  finger2_joint1..4, ...). This is what ``read_encoders`` / ``write_target`` speak.
* **Policy order** = ``POLICY_JOINT_NAMES``: **joint-major** (finger1..5_joint1,
  then finger1..5_joint2, ...). Genesis 1.0.x enumerated the articulation
  breadth-first — joint-major for this hand — and the shipped checkpoint was
  trained under that order, so its obs and action layouts are joint-major.

``DeployController`` remaps encoder->policy on read and policy->encoder on write via
``ENC_TO_POLICY`` / its inverse.

Genesis 1.2+ parses kinematic trees depth-first, so a *live* articulation now
enumerates finger-major (identical to the encoder order). ``POLICY_JOINT_NAMES``
tracks the trained artifact, not the live env; a policy retrained under Genesis
1.2 has finger-major layouts and needs ``ENC_TO_POLICY`` updated to the identity.
``tests/test_examples_wuji_deploy_joint_order.py`` pins the env's depth-first
enumeration so any future Genesis ordering change is caught there.
"""

from __future__ import annotations

import numpy as np

from genelab_wuji.reorient.constants import REORIENT_JOINT_POS

JOINT_NAMES_20: tuple[str, ...] = tuple(REORIENT_JOINT_POS)
"""The 20 hand joint names in encoder / hardware order (finger-major)."""

N_JOINTS: int = len(JOINT_NAMES_20)
"""Hand DOF count (20 = 5 fingers x 4 joints)."""

ENC_TO_POLICY: tuple[int, ...] = (
    0,
    4,
    8,
    12,
    16,
    1,
    5,
    9,
    13,
    17,
    2,
    6,
    10,
    14,
    18,
    3,
    7,
    11,
    15,
    19,
)
"""``policy_order[i] = encoder_order[ENC_TO_POLICY[i]]`` (encoder = ``JOINT_NAMES_20``)."""

POLICY_JOINT_NAMES: tuple[str, ...] = tuple(JOINT_NAMES_20[i] for i in ENC_TO_POLICY)
"""Hand joint names in policy / Genesis articulation order (joint-major)."""


def default_joint_pos() -> np.ndarray:
    """Home grasp keyframe as a ``(20,)`` array, in ``JOINT_NAMES_20`` (encoder) order."""
    return np.array([REORIENT_JOINT_POS[name] for name in JOINT_NAMES_20], dtype=float)


def default_joint_pos_policy() -> np.ndarray:
    """Home grasp keyframe ``(20,)`` in policy / articulation order (joint-major)."""
    return default_joint_pos()[list(ENC_TO_POLICY)]
