"""Franka Emika Panda asset zoo entry — 7-DoF arm + 2-finger parallel gripper.

Joints split into two actuator groups: a high-PD arm group covering ``panda_joint1..7``
and a stiffer finger group covering the parallel-jaw fingers. Gain values follow
Isaac Lab's ``FRANKA_PANDA_HIGH_PD_CFG`` so behaviour transfers when downstream code
mixes the two stacks. ``default_joint_pos`` matches the MuJoCo Menagerie home keyframe.
"""

from typing import Final

from genelab.actuator import ImplicitPDActuatorCfg
from genelab.entity import ArticulationCfg
from genelab.registry import register_robot
from genelab.utils.download import AssetSpec, fetch_asset

_MJCF: Final = AssetSpec(
    name="franka",
    url="https://raw.githubusercontent.com/KraHsu/genelab-assets/main/franka/franka.xml",
    md5="4c413c93f77e8e7bd5364c23a772c72d",
    filename="franka.xml",
)

_HOME_POSE: Final[dict[str, float]] = {
    "panda_joint1": 0.0,
    "panda_joint2": -0.785,
    "panda_joint3": 0.0,
    "panda_joint4": -2.356,
    "panda_joint5": 0.0,
    "panda_joint6": 1.571,
    "panda_joint7": 0.785,
    "panda_finger_joint.*": 0.04,
}


def FrankaPandaCfg() -> ArticulationCfg:
    """Return a fresh :class:`ArticulationCfg` for the Franka Panda arm + gripper.

    Two actuator groups: ``panda_arm`` covers the 7 revolute arm joints with the
    Isaac Lab high-PD gains (stiffness=400, damping=80); ``panda_hand`` covers the
    parallel-jaw fingers with stiffer gains for predictable grasping.
    """

    mjcf_path = fetch_asset(_MJCF)
    return ArticulationCfg(
        mjcf_path=str(mjcf_path),
        init_pos=(0.0, 0.0, 0.0),
        default_joint_pos=dict(_HOME_POSE),
        actuators={
            "panda_arm": ImplicitPDActuatorCfg(
                target_names_expr=(r"panda_joint[1-7]",),
                stiffness=400.0,
                damping=80.0,
                effort_limit=87.0,
                velocity_limit=2.175,
                action_scale=0.5,
            ),
            "panda_hand": ImplicitPDActuatorCfg(
                target_names_expr=(r"panda_finger_joint.*",),
                stiffness=1.0e4,
                damping=200.0,
                effort_limit=20.0,
                velocity_limit=0.2,
                action_scale=0.04,
            ),
        },
    )


register_robot(
    "franka",
    FrankaPandaCfg,
    description="Franka Emika Panda 7-DoF arm with parallel-jaw gripper; high-PD gains.",
    cfg_type=ArticulationCfg,
    examples=[
        "genelab info franka",
        "genelab list robots",
    ],
)
