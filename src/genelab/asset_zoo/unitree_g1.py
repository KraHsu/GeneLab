"""Unitree G1 asset zoo entry — 29-DoF humanoid sourced from MuJoCo Menagerie.

Actuator parameters mirror ``examples/unitree/.../g1/constants.py``: six joint groups
matched to the four Unitree motor families (5020 / 7520_14 / 7520_22 / 4010) plus two
parallel-pair groups (waist pitch+roll, ankles). Stiffness / damping / armature are all
derived from two-stage planetary reflected inertias and a fixed natural frequency, so
all four scalars stay coupled if a downstream override changes one. Joint regexes and
the home pose match the values that the existing G1 training example trains against.

The MJCF blob lives inside a Menagerie ``.tar.gz`` so the 38 STL mesh dependencies
travel together; :func:`fetch_asset` extracts the archive and returns the path to
``unitree_g1/g1.xml``.
"""

from typing import Final

from genelab.actuator import DCMotorActuatorCfg
from genelab.entity import ArticulationCfg
from genelab.registry import register_robot
from genelab.utils.download import AssetSpec, fetch_asset


_MJCF: Final = AssetSpec(
    name="g1",
    url="https://raw.githubusercontent.com/KraHsu/genelab-assets/main/unitree_g1/unitree_g1.tar.gz",
    md5="914aed45f46af706ce448a74319fe2f4",
    filename="unitree_g1.tar.gz",
    archive_member="unitree_g1/g1.xml",
)


def _reflected_inertia(
    rotor_inertias: tuple[float, float, float], gears: tuple[float, float, float]
) -> float:
    i0, i1, i2 = rotor_inertias
    g0, g1, g2 = gears
    return i0 * (g0 * g1 * g2) ** 2 + i1 * (g1 * g2) ** 2 + i2 * g2**2


_ARMATURE_5020 = _reflected_inertia((0.139e-4, 0.017e-4, 0.169e-4), (1.0, 1 + 46 / 18, 1 + 56 / 16))
_ARMATURE_7520_14 = _reflected_inertia((0.489e-4, 0.098e-4, 0.533e-4), (1.0, 4.5, 1 + 48 / 22))
_ARMATURE_7520_22 = _reflected_inertia((0.489e-4, 0.109e-4, 0.738e-4), (1.0, 4.5, 5.0))
_ARMATURE_4010 = _reflected_inertia((0.068e-4, 0.0, 0.0), (1.0, 5.0, 5.0))

_NATURAL_FREQ: Final = 10 * 2.0 * 3.1415926535
_DAMPING_RATIO: Final = 2.0

_STIFF_5020 = _ARMATURE_5020 * _NATURAL_FREQ**2
_STIFF_7520_14 = _ARMATURE_7520_14 * _NATURAL_FREQ**2
_STIFF_7520_22 = _ARMATURE_7520_22 * _NATURAL_FREQ**2
_STIFF_4010 = _ARMATURE_4010 * _NATURAL_FREQ**2

_DAMP_5020 = 2.0 * _DAMPING_RATIO * _ARMATURE_5020 * _NATURAL_FREQ
_DAMP_7520_14 = 2.0 * _DAMPING_RATIO * _ARMATURE_7520_14 * _NATURAL_FREQ
_DAMP_7520_22 = 2.0 * _DAMPING_RATIO * _ARMATURE_7520_22 * _NATURAL_FREQ
_DAMP_4010 = 2.0 * _DAMPING_RATIO * _ARMATURE_4010 * _NATURAL_FREQ


def _action_scale(effort_limit: float, stiffness: float) -> float:
    return 0.25 * effort_limit / stiffness


def UnitreeG1Cfg() -> ArticulationCfg:
    """Return a fresh :class:`ArticulationCfg` for the Unitree G1 29-DoF humanoid.

    Six actuator groups partition the 29 actuated joints by motor family. The home
    pose matches the knees-bent stand keyframe used by the existing G1 training
    example so a switch from the in-tree asset path to this asset-zoo entry leaves
    the training curriculum invariant.
    """

    mjcf_path = fetch_asset(_MJCF)
    return ArticulationCfg(
        mjcf_path=str(mjcf_path),
        init_pos=(0.0, 0.0, 0.76),
        default_joint_pos={
            ".*_hip_pitch_joint": -0.312,
            ".*_knee_joint": 0.669,
            ".*_ankle_pitch_joint": -0.363,
            ".*_elbow_joint": 0.6,
            "left_shoulder_roll_joint": 0.2,
            "left_shoulder_pitch_joint": 0.2,
            "right_shoulder_roll_joint": -0.2,
            "right_shoulder_pitch_joint": 0.2,
        },
        actuators={
            "5020": DCMotorActuatorCfg(
                target_names_expr=(
                    ".*_elbow_joint",
                    ".*_shoulder_pitch_joint",
                    ".*_shoulder_roll_joint",
                    ".*_shoulder_yaw_joint",
                    ".*_wrist_roll_joint",
                ),
                stiffness=_STIFF_5020,
                damping=_DAMP_5020,
                effort_limit=25.0,
                velocity_limit=32.0,
                armature=_ARMATURE_5020,
                action_scale=_action_scale(25.0, _STIFF_5020),
            ),
            "7520_14": DCMotorActuatorCfg(
                target_names_expr=(
                    ".*_hip_pitch_joint",
                    ".*_hip_yaw_joint",
                    "waist_yaw_joint",
                ),
                stiffness=_STIFF_7520_14,
                damping=_DAMP_7520_14,
                effort_limit=88.0,
                velocity_limit=25.0,
                armature=_ARMATURE_7520_14,
                action_scale=_action_scale(88.0, _STIFF_7520_14),
            ),
            "7520_22": DCMotorActuatorCfg(
                target_names_expr=(".*_hip_roll_joint", ".*_knee_joint"),
                stiffness=_STIFF_7520_22,
                damping=_DAMP_7520_22,
                effort_limit=139.0,
                velocity_limit=14.0,
                armature=_ARMATURE_7520_22,
                action_scale=_action_scale(139.0, _STIFF_7520_22),
            ),
            "4010": DCMotorActuatorCfg(
                target_names_expr=(".*_wrist_pitch_joint", ".*_wrist_yaw_joint"),
                stiffness=_STIFF_4010,
                damping=_DAMP_4010,
                effort_limit=5.0,
                velocity_limit=37.0,
                armature=_ARMATURE_4010,
                action_scale=_action_scale(5.0, _STIFF_4010),
            ),
            "waist": DCMotorActuatorCfg(
                target_names_expr=("waist_pitch_joint", "waist_roll_joint"),
                stiffness=_STIFF_5020 * 2,
                damping=_DAMP_5020 * 2,
                effort_limit=50.0,
                velocity_limit=32.0,
                armature=_ARMATURE_5020 * 2,
                action_scale=_action_scale(50.0, _STIFF_5020 * 2),
            ),
            "ankle": DCMotorActuatorCfg(
                target_names_expr=(".*_ankle_pitch_joint", ".*_ankle_roll_joint"),
                stiffness=_STIFF_5020 * 2,
                damping=_DAMP_5020 * 2,
                effort_limit=50.0,
                velocity_limit=32.0,
                armature=_ARMATURE_5020 * 2,
                action_scale=_action_scale(50.0, _STIFF_5020 * 2),
            ),
        },
        foot_link_names=("left_ankle_roll_link", "right_ankle_roll_link"),
    )


register_robot(
    "g1",
    UnitreeG1Cfg,
    description="Unitree G1 29-DoF humanoid (Menagerie source); 6 DCMotor actuator groups.",
    cfg_type=ArticulationCfg,
    examples=[
        "genelab info g1",
        "genelab list robots",
    ],
)
