"""Reorient task constants: frames, initial poses, joint home keyframe.

``right_mjlab.xml`` places the palm at the origin with fingers along +Z. The wrist "tag"
(AprilTag) frame differs from the palm frame by a clean R_y(-90deg) rotation; we replicate
that here on the robot root quaternion so the sim "palm-up" pose matches the real geometry.
Values mirror the mjlab reference so obs / reward math transfers unchanged.
"""

import math

# Robot root pose — palm rotated -90deg about Y so the palm faces up.
REORIENT_ROBOT_ROOT_POS: tuple[float, float, float] = (0.0, 0.0, 0.5)
REORIENT_ROBOT_ROOT_ROT: tuple[float, float, float, float] = (0.70710678, 0.0, -0.70710678, 0.0)

# Cube initial pose (world), above the fingertip cage in the palm-up pose.
REORIENT_CUBE_INIT_POS: tuple[float, float, float] = (-0.0967, 0.0100, 0.5599)
REORIENT_CUBE_INIT_ROT: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
REORIENT_CUBE_HALF_EXTENT: float = 0.027  # 54 mm dex cube

# Actor observation history length (term-major), matching the mjlab reference.
REORIENT_HISTORY_LEN: int = 3

# Tag (wrist marker) <-> palm_link rigid transform.
#   tag_in_palm pos = (0.0262, 0, -0.0563), quat (wxyz) = R_y(+90deg)
TAG_IN_PALM_POS: tuple[float, float, float] = (0.0262, 0.0, -0.0563)
TAG_IN_PALM_QUAT_WXYZ: tuple[float, float, float, float] = (
    math.cos(math.radians(45.0)),
    0.0,
    math.sin(math.radians(45.0)),
    0.0,
)

# Hand joint home keyframe (active grasp pose), per-joint regex -> angle (rad).
REORIENT_JOINT_POS: dict[str, float] = {
    "right_finger1_joint1": 0.8,
    "right_finger1_joint2": -0.0215,
    "right_finger1_joint3": 0.285,
    "right_finger1_joint4": 0.582,
    "right_finger2_joint1": 0.441,
    "right_finger2_joint2": 0.163,
    "right_finger2_joint3": 0.822,
    "right_finger2_joint4": 0.494,
    "right_finger3_joint1": 0.448,
    "right_finger3_joint2": -0.0832,
    "right_finger3_joint3": 0.883,
    "right_finger3_joint4": 0.305,
    "right_finger4_joint1": 0.594,
    "right_finger4_joint2": -0.268,
    "right_finger4_joint3": 1.13,
    "right_finger4_joint4": 0.18,
    "right_finger5_joint1": 1.2,
    "right_finger5_joint2": -0.228,
    "right_finger5_joint3": 0.794,
    "right_finger5_joint4": 0.407,
}

# Goal-marker world position in play/debug-vis mode — beside the hand, clear of the grasp.
GOAL_MARKER_POS: tuple[float, float, float] = (0.10, 0.01, 0.56)

TIP_SITE_NAMES: tuple[str, ...] = tuple(f"right_finger{i}_tip" for i in range(1, 6))
TIP_BODY_NAMES: tuple[str, ...] = tuple(f"right_finger{i}_link4" for i in range(1, 6))
PALM_BODY_NAME: str = "right_palm_link"
