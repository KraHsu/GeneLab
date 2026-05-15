"""Franka Emika Panda asset zoo entry — 7-DoF arm + 2-finger parallel gripper.

Sourced from MuJoCo Menagerie ``franka_emika_panda`` (Apache-2.0). Joint names follow
the upstream convention (``joint1..7`` plus ``finger_joint1..2``) and the default pose
matches the Menagerie ``home`` keyframe. Two actuator groups split arm and hand with
the Isaac Lab ``FRANKA_PANDA_HIGH_PD_CFG`` gains so cross-stack behaviour stays
comparable.
"""

from typing import Final

from genelab.actuator import ImplicitPDActuatorCfg
from genelab.entity import ArticulationCfg
from genelab.registry import register_robot
from genelab.utils.download import AssetSpec, fetch_asset

_MJCF: Final = AssetSpec(
    name="franka",
    url="https://raw.githubusercontent.com/KraHsu/genelab-assets/main/franka_emika_panda/franka_emika_panda.tar.gz",
    md5="6890871be5fbb852a13b6be2a766f270",
    filename="franka_emika_panda.tar.gz",
    archive_member="franka_emika_panda/panda.xml",
)

# Menagerie home keyframe: qpos = "0 0 0 -1.57079 0 1.57079 -0.7853 0.04 0.04".
_HOME_POSE: Final[dict[str, float]] = {
    "joint1": 0.0,
    "joint2": 0.0,
    "joint3": 0.0,
    "joint4": -1.57079,
    "joint5": 0.0,
    "joint6": 1.57079,
    "joint7": -0.7853,
    "finger_joint.*": 0.04,
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
                # Anchored regex: bare ``joint[1-7]`` also substring-matches
                # ``finger_joint1`` / ``finger_joint2`` through ``re.search`` and
                # collides with ``panda_hand``.
                target_names_expr=(r"^joint[1-7]$",),
                stiffness=400.0,
                damping=80.0,
                effort_limit=87.0,
                velocity_limit=2.175,
                action_scale=0.5,
            ),
            "panda_hand": ImplicitPDActuatorCfg(
                target_names_expr=(r"finger_joint.*",),
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
    description="Franka Emika Panda 7-DoF arm + parallel gripper (Menagerie source); high-PD gains.",
    cfg_type=ArticulationCfg,
    examples=[
        "genelab info franka",
        "genelab list robots",
    ],
)
