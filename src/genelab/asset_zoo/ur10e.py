"""Universal Robots UR10e asset zoo entry — 6-DoF industrial arm from MuJoCo Menagerie.

Fixed-base manipulation arm (no floating base / feet), complementing the Franka's
gripper arm with a bare 6-axis wrist. Three implicit-PD actuator groups follow the
Menagerie ``size4`` / ``size3`` / ``size2`` motor classes — effort limits are the
upstream ``forcerange`` (±330 / ±150 / ±56 N·m) and damping the upstream joint damping
(10 / 5 / 2). Stiffness is a moderate position-control default (300 / 150 / 80); tune per
task. Home pose is the Menagerie ``home`` keyframe (tucked-up ready stance). Joint names
follow the upstream convention (``shoulder_pan_joint`` … ``wrist_3_joint``).

The MJCF blob travels inside a Menagerie ``.tar.gz`` (meshes included); :func:`fetch_asset`
extracts it and returns the path to ``universal_robots_ur10e/ur10e.xml``.
"""

from typing import Final

from genelab.actuator import ImplicitPDActuatorCfg
from genelab.entity import ArticulationCfg
from genelab.registry import register_robot
from genelab.utils.download import AssetSpec, fetch_asset

_MJCF: Final = AssetSpec(
    name="ur10e",
    url="https://raw.githubusercontent.com/KraHsu/genelab-assets/main/universal_robots_ur10e/universal_robots_ur10e.tar.gz",
    md5="7a16fa5e7cfeb6dfc1f7631cd60261b5",
    filename="universal_robots_ur10e.tar.gz",
    archive_member="universal_robots_ur10e/ur10e.xml",
)

# Menagerie ``home`` keyframe (qpos order = joint declaration order).
_HOME_POSE: Final[dict[str, float]] = {
    "shoulder_pan_joint": -1.5708,
    "shoulder_lift_joint": -1.5708,
    "elbow_joint": 1.5708,
    "wrist_1_joint": -1.5708,
    "wrist_2_joint": -1.5708,
    "wrist_3_joint": 0.0,
}


def UR10eCfg() -> ArticulationCfg:
    """Return a fresh :class:`ArticulationCfg` for the UR10e 6-DoF arm.

    Three actuator groups partition the joints by Menagerie motor class: ``base`` (the two
    ±330 N·m shoulder motors), ``elbow`` (±150), ``wrists`` (the three ±56 N·m wrist motors).
    """
    mjcf_path = fetch_asset(_MJCF)
    return ArticulationCfg(
        mjcf_path=str(mjcf_path),
        init_pos=(0.0, 0.0, 0.0),
        default_joint_pos=dict(_HOME_POSE),
        actuators={
            "base": ImplicitPDActuatorCfg(
                target_names_expr=("shoulder_pan_joint", "shoulder_lift_joint"),
                stiffness=300.0,
                damping=10.0,
                effort_limit=330.0,
                armature=0.1,
            ),
            "elbow": ImplicitPDActuatorCfg(
                target_names_expr=("elbow_joint",),
                stiffness=150.0,
                damping=5.0,
                effort_limit=150.0,
                armature=0.1,
            ),
            "wrists": ImplicitPDActuatorCfg(
                target_names_expr=(r"wrist_[123]_joint",),
                stiffness=80.0,
                damping=2.0,
                effort_limit=56.0,
                armature=0.1,
            ),
        },
    )


register_robot(
    "ur10e",
    UR10eCfg,
    description="Universal Robots UR10e 6-DoF industrial arm (Menagerie source); 3 implicit-PD groups.",
    cfg_type=ArticulationCfg,
    examples=[
        "genelab info ur10e",
        "genelab list robots",
    ],
)
