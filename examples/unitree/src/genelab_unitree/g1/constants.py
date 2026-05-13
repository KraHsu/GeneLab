"""Unitree G1 actuator + posture constants.

Reflected inertia, stiffness, damping, and action scales are derived from the same motor
specs mjlab uses. Each actuator group maps a tuple of joint name regexes to a stiffness /
damping / effort triple; the per-joint dictionaries fan those values out by regex so
GeneLab's manager-based env can index them by joint name.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


def _reflected_inertia(rotor_inertias: tuple[float, float, float], gears: tuple[float, float, float]) -> float:
    """Two-stage planetary reflected inertia (port of mjlab ``reflected_inertia_from_two_stage_planetary``)."""
    i0, i1, i2 = rotor_inertias
    g0, g1, g2 = gears
    return i0 * (g0 * g1 * g2) ** 2 + i1 * (g1 * g2) ** 2 + i2 * g2 ** 2


ARMATURE_5020 = _reflected_inertia((0.139e-4, 0.017e-4, 0.169e-4), (1.0, 1 + 46 / 18, 1 + 56 / 16))
ARMATURE_7520_14 = _reflected_inertia((0.489e-4, 0.098e-4, 0.533e-4), (1.0, 4.5, 1 + 48 / 22))
ARMATURE_7520_22 = _reflected_inertia((0.489e-4, 0.109e-4, 0.738e-4), (1.0, 4.5, 5.0))
ARMATURE_4010 = _reflected_inertia((0.068e-4, 0.0, 0.0), (1.0, 5.0, 5.0))

NATURAL_FREQ: Final = 10 * 2.0 * 3.1415926535
DAMPING_RATIO: Final = 2.0

STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ ** 2
STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ ** 2
STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ ** 2
STIFFNESS_4010 = ARMATURE_4010 * NATURAL_FREQ ** 2

DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ
DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ
DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ
DAMPING_4010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ


@dataclass(frozen=True)
class ActuatorGroup:
    target_names_expr: tuple[str, ...]
    stiffness: float
    damping: float
    effort_limit: float


G1_ACTUATOR_5020 = ActuatorGroup(
    target_names_expr=(
        ".*_elbow_joint",
        ".*_shoulder_pitch_joint",
        ".*_shoulder_roll_joint",
        ".*_shoulder_yaw_joint",
        ".*_wrist_roll_joint",
    ),
    stiffness=STIFFNESS_5020,
    damping=DAMPING_5020,
    effort_limit=25.0,
)
G1_ACTUATOR_7520_14 = ActuatorGroup(
    target_names_expr=(".*_hip_pitch_joint", ".*_hip_yaw_joint", "waist_yaw_joint"),
    stiffness=STIFFNESS_7520_14,
    damping=DAMPING_7520_14,
    effort_limit=88.0,
)
G1_ACTUATOR_7520_22 = ActuatorGroup(
    target_names_expr=(".*_hip_roll_joint", ".*_knee_joint"),
    stiffness=STIFFNESS_7520_22,
    damping=DAMPING_7520_22,
    effort_limit=139.0,
)
G1_ACTUATOR_4010 = ActuatorGroup(
    target_names_expr=(".*_wrist_pitch_joint", ".*_wrist_yaw_joint"),
    stiffness=STIFFNESS_4010,
    damping=DAMPING_4010,
    effort_limit=5.0,
)
# Waist pitch/roll and ankles use two 5020 actuators in parallel.
G1_ACTUATOR_WAIST = ActuatorGroup(
    target_names_expr=("waist_pitch_joint", "waist_roll_joint"),
    stiffness=STIFFNESS_5020 * 2,
    damping=DAMPING_5020 * 2,
    effort_limit=25.0 * 2,
)
G1_ACTUATOR_ANKLE = ActuatorGroup(
    target_names_expr=(".*_ankle_pitch_joint", ".*_ankle_roll_joint"),
    stiffness=STIFFNESS_5020 * 2,
    damping=DAMPING_5020 * 2,
    effort_limit=25.0 * 2,
)

G1_ACTUATORS: Final = (
    G1_ACTUATOR_5020,
    G1_ACTUATOR_7520_14,
    G1_ACTUATOR_7520_22,
    G1_ACTUATOR_4010,
    G1_ACTUATOR_WAIST,
    G1_ACTUATOR_ANKLE,
)


def _fan_out(value_per_group: dict[ActuatorGroup, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for group, value in value_per_group.items():
        for pat in group.target_names_expr:
            out[pat] = value
    return out


G1_JOINT_KP: Final[dict[str, float]] = _fan_out({a: a.stiffness for a in G1_ACTUATORS})
G1_JOINT_KV: Final[dict[str, float]] = _fan_out({a: a.damping for a in G1_ACTUATORS})

# 0.25 * effort / stiffness — matches mjlab's per-joint action scale derivation.
G1_ACTION_SCALE: Final[dict[str, float]] = _fan_out(
    {a: 0.25 * a.effort_limit / a.stiffness for a in G1_ACTUATORS}
)

# Knees-bent home pose (matches mjlab KNEES_BENT_KEYFRAME).
G1_DEFAULT_JOINT_POS: Final[dict[str, float]] = {
    ".*_hip_pitch_joint": -0.312,
    ".*_knee_joint": 0.669,
    ".*_ankle_pitch_joint": -0.363,
    ".*_elbow_joint": 0.6,
    "left_shoulder_roll_joint": 0.2,
    "left_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "right_shoulder_pitch_joint": 0.2,
}

G1_INIT_BASE_HEIGHT: Final = 0.76

# Foot links used by the feet_air_time reward.
G1_FOOT_LINKS: Final = ("left_ankle_roll_link", "right_ankle_roll_link")
