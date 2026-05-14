"""Unitree G1 actuator + posture constants.

Reflected inertia, stiffness, damping, effort and velocity limits are derived from the
same motor specs mjlab uses. The actuator groups are :class:`DCMotorActuatorCfg`
instances ready to plug into :class:`ArticulationCfg.actuators`.
"""

from typing import Final

from genelab.actuator import DCMotorActuatorCfg


def _reflected_inertia(
    rotor_inertias: tuple[float, float, float], gears: tuple[float, float, float]
) -> float:
    """Two-stage planetary reflected inertia (port of mjlab ``reflected_inertia_from_two_stage_planetary``)."""
    i0, i1, i2 = rotor_inertias
    g0, g1, g2 = gears
    return i0 * (g0 * g1 * g2) ** 2 + i1 * (g1 * g2) ** 2 + i2 * g2**2


ARMATURE_5020 = _reflected_inertia((0.139e-4, 0.017e-4, 0.169e-4), (1.0, 1 + 46 / 18, 1 + 56 / 16))
ARMATURE_7520_14 = _reflected_inertia((0.489e-4, 0.098e-4, 0.533e-4), (1.0, 4.5, 1 + 48 / 22))
ARMATURE_7520_22 = _reflected_inertia((0.489e-4, 0.109e-4, 0.738e-4), (1.0, 4.5, 5.0))
ARMATURE_4010 = _reflected_inertia((0.068e-4, 0.0, 0.0), (1.0, 5.0, 5.0))

NATURAL_FREQ: Final = 10 * 2.0 * 3.1415926535
DAMPING_RATIO: Final = 2.0

STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ**2
STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ**2
STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ**2
STIFFNESS_4010 = ARMATURE_4010 * NATURAL_FREQ**2

DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ
DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ
DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ
DAMPING_4010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ

# No-load speeds from Unitree G1 motor specs (rad/s at the output shaft, after gearing).
# These set the linear de-rating breakpoint of the DCMotor torque-speed curve.
VEL_LIMIT_5020: Final = 32.0
VEL_LIMIT_7520_14: Final = 25.0
VEL_LIMIT_7520_22: Final = 14.0
VEL_LIMIT_4010: Final = 37.0


def _action_scale(effort_limit: float, stiffness: float) -> float:
    """Matches mjlab's per-joint action-scale derivation: ``0.25 * effort / stiffness``."""
    return 0.25 * effort_limit / stiffness


G1_ACTUATOR_5020 = DCMotorActuatorCfg(
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
    velocity_limit=VEL_LIMIT_5020,
    armature=ARMATURE_5020,
    action_scale=_action_scale(25.0, STIFFNESS_5020),
)
G1_ACTUATOR_7520_14 = DCMotorActuatorCfg(
    target_names_expr=(".*_hip_pitch_joint", ".*_hip_yaw_joint", "waist_yaw_joint"),
    stiffness=STIFFNESS_7520_14,
    damping=DAMPING_7520_14,
    effort_limit=88.0,
    velocity_limit=VEL_LIMIT_7520_14,
    armature=ARMATURE_7520_14,
    action_scale=_action_scale(88.0, STIFFNESS_7520_14),
)
G1_ACTUATOR_7520_22 = DCMotorActuatorCfg(
    target_names_expr=(".*_hip_roll_joint", ".*_knee_joint"),
    stiffness=STIFFNESS_7520_22,
    damping=DAMPING_7520_22,
    effort_limit=139.0,
    velocity_limit=VEL_LIMIT_7520_22,
    armature=ARMATURE_7520_22,
    action_scale=_action_scale(139.0, STIFFNESS_7520_22),
)
G1_ACTUATOR_4010 = DCMotorActuatorCfg(
    target_names_expr=(".*_wrist_pitch_joint", ".*_wrist_yaw_joint"),
    stiffness=STIFFNESS_4010,
    damping=DAMPING_4010,
    effort_limit=5.0,
    velocity_limit=VEL_LIMIT_4010,
    armature=ARMATURE_4010,
    action_scale=_action_scale(5.0, STIFFNESS_4010),
)
# Waist pitch/roll and ankles use two 5020 actuators in parallel.
G1_ACTUATOR_WAIST = DCMotorActuatorCfg(
    target_names_expr=("waist_pitch_joint", "waist_roll_joint"),
    stiffness=STIFFNESS_5020 * 2,
    damping=DAMPING_5020 * 2,
    effort_limit=25.0 * 2,
    velocity_limit=VEL_LIMIT_5020,
    armature=ARMATURE_5020 * 2,
    action_scale=_action_scale(50.0, STIFFNESS_5020 * 2),
)
G1_ACTUATOR_ANKLE = DCMotorActuatorCfg(
    target_names_expr=(".*_ankle_pitch_joint", ".*_ankle_roll_joint"),
    stiffness=STIFFNESS_5020 * 2,
    damping=DAMPING_5020 * 2,
    effort_limit=25.0 * 2,
    velocity_limit=VEL_LIMIT_5020,
    armature=ARMATURE_5020 * 2,
    action_scale=_action_scale(50.0, STIFFNESS_5020 * 2),
)

G1_ACTUATORS_CFG: Final = {
    "5020": G1_ACTUATOR_5020,
    "7520_14": G1_ACTUATOR_7520_14,
    "7520_22": G1_ACTUATOR_7520_22,
    "4010": G1_ACTUATOR_4010,
    "waist": G1_ACTUATOR_WAIST,
    "ankle": G1_ACTUATOR_ANKLE,
}

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
