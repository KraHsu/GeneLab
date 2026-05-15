"""ANYbotics Anymal C asset zoo entry — 12-DoF quadruped sourced from MuJoCo Menagerie.

Single actuator group covering all 12 leg joints with implicit-PD gains aligned to Isaac
Lab's ``ANYMAL_C_CFG`` (``stiffness=80, damping=2, effort=80``). The Menagerie MJCF
ships no ``<keyframe>`` element so the canonical stand pose (HAA=0, HFE=±0.4, KFE=∓0.8)
is declared here as ``default_joint_pos``.
"""

from typing import Final

from genelab.actuator import ImplicitPDActuatorCfg
from genelab.entity import ArticulationCfg
from genelab.registry import register_robot
from genelab.utils.download import AssetSpec, fetch_asset


_MJCF: Final = AssetSpec(
    name="anymal-c",
    url="https://raw.githubusercontent.com/KraHsu/genelab-assets/main/anybotics_anymal_c/anybotics_anymal_c.tar.gz",
    md5="7c3f421c24d4767be6d26f68cdbadae0",
    filename="anybotics_anymal_c.tar.gz",
    archive_member="anybotics_anymal_c/anymal_c.xml",
)


def AnymalCCfg() -> ArticulationCfg:
    """Return a fresh :class:`ArticulationCfg` for the ANYbotics Anymal C quadruped.

    Front legs (LF / RF) bend forward, hind legs (LH / RH) bend backward so the robot
    spawns in the canonical mid-stance pose used in published Anymal training recipes.
    """

    mjcf_path = fetch_asset(_MJCF)
    return ArticulationCfg(
        mjcf_path=str(mjcf_path),
        init_pos=(0.0, 0.0, 0.6),
        default_joint_pos={
            r"(LF|RF|LH|RH)_HAA": 0.0,
            r"(LF|RF)_HFE": 0.4,
            r"(LH|RH)_HFE": -0.4,
            r"(LF|RF)_KFE": -0.8,
            r"(LH|RH)_KFE": 0.8,
        },
        actuators={
            "legs": ImplicitPDActuatorCfg(
                target_names_expr=(r"(LF|RF|LH|RH)_(HAA|HFE|KFE)",),
                stiffness=80.0,
                damping=2.0,
                effort_limit=80.0,
                velocity_limit=7.5,
                action_scale=0.5,
            ),
        },
        foot_link_names=("LF_FOOT", "RF_FOOT", "LH_FOOT", "RH_FOOT"),
    )


register_robot(
    "anymal-c",
    AnymalCCfg,
    description="ANYbotics Anymal C 12-DoF quadruped (Menagerie source); single ImplicitPD group.",
    cfg_type=ArticulationCfg,
    examples=[
        "genelab info anymal-c",
        "genelab list robots",
    ],
)
