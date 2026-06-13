"""Shared deploy constants: joint ordering and home pose.

The single source of truth for joint order is the reorient task's home keyframe
``REORIENT_JOINT_POS`` (finger1_joint1..4, finger2..., finger5_joint4). That order
matches both the GeneLab policy obs/action layout and ``wujihandpy``'s (5, 4)
row-major flatten, so the deploy stack never has to remap joints.
"""

from __future__ import annotations

import numpy as np

from genelab_wuji.reorient.constants import REORIENT_JOINT_POS

JOINT_NAMES_20: tuple[str, ...] = tuple(REORIENT_JOINT_POS)
"""The 20 hand joint names in encoder / policy order."""

N_JOINTS: int = len(JOINT_NAMES_20)
"""Hand DOF count (20 = 5 fingers x 4 joints)."""


def default_joint_pos() -> np.ndarray:
    """Home grasp keyframe as a ``(20,)`` array, in ``JOINT_NAMES_20`` order."""
    return np.array([REORIENT_JOINT_POS[name] for name in JOINT_NAMES_20], dtype=float)
