"""Frame transforms for the deploy pipeline (numpy, wxyz quaternion convention).

Two concerns live here:

* **Quaternion math** (``quat_apply`` / ``quat_mul`` / ``quat_conjugate``) in the
  wxyz Hamilton convention used across the MuJoCo / mjlab / GeneLab stack.
* **real2sim lift** (``cube_cam_to_tag``): the vision pipeline reports the cube
  pose and the wrist-AprilTag pose both in the *camera* frame; ``cube_cam_to_tag``
  expresses the cube in the *tag* frame, which is the frame the policy observes.

All rotations are 3x3 matrices whose columns are the source-frame axes written in
the target frame (``R_a_b`` maps a vector in frame ``a`` into frame ``b``).
"""

from __future__ import annotations

import numpy as np


def quat_apply(quat_wxyz: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Rotate ``vec`` by the unit quaternion ``quat_wxyz`` (Hamilton, wxyz)."""
    w, x, y, z = quat_wxyz
    R = np.array([
        [1 - 2 * (y * y + z * z),     2 * (x * y - z * w),     2 * (x * z + y * w)],
        [    2 * (x * y + z * w), 1 - 2 * (x * x + z * z),     2 * (y * z - x * w)],
        [    2 * (x * z - y * w),     2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    return R @ vec


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two wxyz quaternions: ``q1 ∘ q2``."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_conjugate(quat_wxyz: np.ndarray) -> np.ndarray:
    """Conjugate (== inverse for a unit quat) of a wxyz quaternion."""
    w, x, y, z = quat_wxyz
    return np.array([w, -x, -y, -z])


def cube_cam_to_tag(
    R_tag_cam: np.ndarray,
    t_tag_cam: np.ndarray,
    R_cube_cam: np.ndarray,
    t_cube_cam: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Express the cube pose in the wrist-tag frame.

    Args:
        R_tag_cam: ``(3, 3)`` tag rotation in the camera frame.
        t_tag_cam: ``(3,)`` tag origin in the camera frame.
        R_cube_cam: ``(3, 3)`` cube rotation in the camera frame.
        t_cube_cam: ``(3,)`` cube center in the camera frame.

    Returns:
        ``(R_cube_tag, t_cube_tag)`` — the cube pose in the tag frame.
    """
    R_cam_tag = R_tag_cam.T
    t_cam_tag = -R_cam_tag @ t_tag_cam
    R_cube_tag = R_cam_tag @ R_cube_cam
    t_cube_tag = R_cam_tag @ t_cube_cam + t_cam_tag
    return R_cube_tag, t_cube_tag
