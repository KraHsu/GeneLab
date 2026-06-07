"""Unitree Go1 asset zoo entry — 12-DoF quadruped sourced from MuJoCo Menagerie.

Three actuator groups split by joint role (hip / thigh / calf) with implicit-PD gains
aligned to Isaac Lab's published Go1 configuration (``stiffness=25, damping=0.5``). The
calf group carries the higher ``effort_limit=35.55`` Menagerie reports for its forcerange,
matching real-hardware peak torque; hip and thigh share the lower ``23.7``.

The Menagerie MJCF has no foot *bodies* — each foot is a geom (class "foot") on the
corresponding ``*_calf`` body — so ``foot_link_names`` points at the calf links, which are
the foot-bearing links Genesis exposes. Downstream feet-air-time / contact rewards anchor a
contact sensor there; reward terms that need the toe position (clearance / slip) should add
a foot-site link offset of ``(0, 0, -0.213)`` (the ``FR``/``FL``/``RR``/``RL`` site in the
calf frame).
"""

from typing import Final

from genelab.actuator import ImplicitPDActuatorCfg
from genelab.entity import ArticulationCfg
from genelab.registry import register_robot
from genelab.utils.download import AssetSpec, fetch_asset


_MJCF: Final = AssetSpec(
    name="go1",
    url="https://raw.githubusercontent.com/KraHsu/genelab-assets/main/unitree_go1/unitree_go1.tar.gz",
    md5="739195fe2a1ebc7d8b502061cec77fb8",
    filename="unitree_go1.tar.gz",
    archive_member="unitree_go1/go1.xml",
)

_LEG_REGEX: Final = r"(FR|FL|RR|RL)"


def UnitreeGo1Cfg() -> ArticulationCfg:
    """Return a fresh :class:`ArticulationCfg` for the Unitree Go1 quadruped.

    Home pose matches the Menagerie ``home`` keyframe (hip=0, thigh=0.9, calf=-1.8)
    repeated across the four legs.
    """

    mjcf_path = fetch_asset(_MJCF)
    return ArticulationCfg(
        mjcf_path=str(mjcf_path),
        init_pos=(0.0, 0.0, 0.42),
        default_joint_pos={
            rf"{_LEG_REGEX}_hip_joint": 0.0,
            rf"{_LEG_REGEX}_thigh_joint": 0.9,
            rf"{_LEG_REGEX}_calf_joint": -1.8,
        },
        actuators={
            "hip": ImplicitPDActuatorCfg(
                target_names_expr=(rf"{_LEG_REGEX}_hip_joint",),
                stiffness=25.0,
                damping=0.5,
                effort_limit=23.7,
                velocity_limit=30.1,
                action_scale=0.25,
            ),
            "thigh": ImplicitPDActuatorCfg(
                target_names_expr=(rf"{_LEG_REGEX}_thigh_joint",),
                stiffness=25.0,
                damping=0.5,
                effort_limit=23.7,
                velocity_limit=30.1,
                action_scale=0.25,
            ),
            "calf": ImplicitPDActuatorCfg(
                target_names_expr=(rf"{_LEG_REGEX}_calf_joint",),
                stiffness=25.0,
                damping=0.5,
                effort_limit=35.55,
                velocity_limit=30.1,
                action_scale=0.25,
            ),
        },
        foot_link_names=("FR_calf", "FL_calf", "RR_calf", "RL_calf"),
    )


register_robot(
    "go1",
    UnitreeGo1Cfg,
    description="Unitree Go1 12-DoF quadruped (Menagerie source); 3 ImplicitPD groups, Isaac Lab gains.",
    cfg_type=ArticulationCfg,
    examples=[
        "genelab info go1",
        "genelab list robots",
    ],
)
