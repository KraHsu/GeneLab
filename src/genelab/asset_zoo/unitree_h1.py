"""Unitree H1 asset zoo entry — 19-DoF humanoid sourced from MuJoCo Menagerie.

Five implicit-PD actuator groups partition the 19 actuated joints by the hardware
torque limit reported in the Menagerie MJCF ``ctrlrange`` (hips/torso ±200, knees
±300, ankles ±40, shoulder pitch/roll ±40, shoulder-yaw/elbow ±18). Stiffness /
damping mirror Isaac Lab's published ``H1_CFG`` (legs/torso 150–200 stiffness, ankles
20, arms 40; damping 5 / 4 / 10). Joint names carry **no ``_joint`` suffix** (Menagerie
convention: ``left_hip_yaw``, ``torso``, …) — the regexes match accordingly. Feet are
``left_ankle_link`` / ``right_ankle_link``.

The MJCF blob travels inside a Menagerie ``.tar.gz`` (meshes included); :func:`fetch_asset`
extracts it and returns the path to ``unitree_h1/h1.xml``.
"""

from typing import Final

from genelab.actuator import ImplicitPDActuatorCfg
from genelab.entity import ArticulationCfg
from genelab.registry import register_robot
from genelab.utils.download import AssetSpec, fetch_asset

_MJCF: Final = AssetSpec(
    name="h1",
    url="https://raw.githubusercontent.com/KraHsu/genelab-assets/main/unitree_h1/unitree_h1.tar.gz",
    md5="9b8cbe7e28e96d57a0a7427fed92679b",
    filename="unitree_h1.tar.gz",
    archive_member="unitree_h1/h1.xml",
)


def UnitreeH1Cfg() -> ArticulationCfg:
    """Return a fresh :class:`ArticulationCfg` for the Unitree H1 19-DoF humanoid.

    Home pose is the knees-bent stand (``hip_pitch=-0.4``, ``knee=0.8``, ``ankle=-0.4``)
    Isaac Lab's H1 locomotion task resets into; ``init_pos`` z matches the pelvis height.
    """
    mjcf_path = fetch_asset(_MJCF)
    return ArticulationCfg(
        mjcf_path=str(mjcf_path),
        init_pos=(0.0, 0.0, 1.05),
        default_joint_pos={
            ".*_hip_pitch": -0.4,
            ".*_knee": 0.8,
            ".*_ankle": -0.4,
        },
        actuators={
            # Hips + torso: ±200 N·m motors. Hip pitch / torso run stiffer than yaw / roll.
            "hips_torso": ImplicitPDActuatorCfg(
                target_names_expr=(".*_hip_yaw", ".*_hip_roll", ".*_hip_pitch", "torso"),
                stiffness={
                    ".*_hip_yaw": 150.0,
                    ".*_hip_roll": 150.0,
                    ".*_hip_pitch": 200.0,
                    "torso": 200.0,
                },
                damping=5.0,
                effort_limit=200.0,
                armature=0.1,
            ),
            "knees": ImplicitPDActuatorCfg(
                target_names_expr=(".*_knee",),
                stiffness=200.0,
                damping=5.0,
                effort_limit=300.0,
                armature=0.1,
            ),
            "ankles": ImplicitPDActuatorCfg(
                target_names_expr=(".*_ankle",),
                stiffness=20.0,
                damping=4.0,
                effort_limit=40.0,
                armature=0.1,
            ),
            "shoulders": ImplicitPDActuatorCfg(
                target_names_expr=(".*_shoulder_pitch", ".*_shoulder_roll"),
                stiffness=40.0,
                damping=10.0,
                effort_limit=40.0,
                armature=0.1,
            ),
            "forearms": ImplicitPDActuatorCfg(
                target_names_expr=(".*_shoulder_yaw", ".*_elbow"),
                stiffness=40.0,
                damping=10.0,
                effort_limit=18.0,
                armature=0.1,
            ),
        },
        foot_link_names=("left_ankle_link", "right_ankle_link"),
    )


register_robot(
    "h1",
    UnitreeH1Cfg,
    description="Unitree H1 19-DoF humanoid (Menagerie source); 5 implicit-PD actuator groups.",
    cfg_type=ArticulationCfg,
    examples=[
        "genelab info h1",
        "genelab list robots",
    ],
)
