"""Reproduce the real cube inside the Genesis sim from its tag-frame pose.

The vision pipeline reports the cube pose in the wrist-AprilTag frame (the frame
the policy observes). To visualize it in sim we need the tag's pose in sim-world
coordinates, then lift the cube through it. For the fixed-base hand the tag world
pose is constant and derived from the palm pose via the ``TAG_IN_PALM`` rigid
offset (mirrors ``genelab_wuji.reorient.mdp.observations._tag_pose``).

``cube_pose_in_tag_to_world`` (viewer) and ``cube_pose_world_to_tag`` (obs) are
exact inverses so what the policy sees and what the viewer draws agree.
"""

from __future__ import annotations

import numpy as np

from genelab_wuji.deploy.frame_transform import quat_apply, quat_conjugate, quat_mul
from genelab_wuji.reorient.constants import TAG_IN_PALM_POS, TAG_IN_PALM_QUAT_WXYZ


def tag_pose_in_world(
    palm_pos_w: np.ndarray, palm_quat_w: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """World pose of the wrist tag given the palm world pose."""
    tag_in_palm_pos = np.asarray(TAG_IN_PALM_POS, dtype=float)
    tag_in_palm_quat = np.asarray(TAG_IN_PALM_QUAT_WXYZ, dtype=float)
    tag_pos_w = palm_pos_w + quat_apply(palm_quat_w, tag_in_palm_pos)
    tag_quat_w = quat_mul(palm_quat_w, tag_in_palm_quat)
    return tag_pos_w, tag_quat_w


def cube_pose_in_tag_to_world(
    tag_pos_w: np.ndarray,
    tag_quat_w: np.ndarray,
    cube_pos_tag: np.ndarray,
    cube_quat_tag: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Lift a tag-frame cube pose into sim-world coordinates (for the viewer)."""
    cube_pos_w = tag_pos_w + quat_apply(tag_quat_w, cube_pos_tag)
    cube_quat_w = quat_mul(tag_quat_w, cube_quat_tag)
    return cube_pos_w, cube_quat_w


def cube_pose_world_to_tag(
    tag_pos_w: np.ndarray,
    tag_quat_w: np.ndarray,
    cube_pos_w: np.ndarray,
    cube_quat_w: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Express a world cube pose in the tag frame (inverse of the lift above)."""
    tag_quat_inv = quat_conjugate(tag_quat_w)
    cube_pos_tag = quat_apply(tag_quat_inv, cube_pos_w - tag_pos_w)
    cube_quat_tag = quat_mul(tag_quat_inv, cube_quat_w)
    return cube_pos_tag, cube_quat_tag
