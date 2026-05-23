"""Wonik Allegro Hand asset zoo entry — 16-DoF dexterous hand from MuJoCo Menagerie.

The right-hand variant: four fingers (index ``ff``, middle ``mf``, ring ``rf``, thumb
``th``) each with four joints ``*j0``–``*j3``. A single implicit-PD actuator group covers
all 16 joints with the uniform low gains Isaac Lab's ``AllegroHand`` RL task uses
(stiffness 3.0, damping 0.1, ~0.5 N·m effort) — dexterous in-hand manipulation drives the
fingers with soft position control rather than per-joint motor tuning. Fixed base; no feet;
the open-hand rest pose (all joints 0) is the MJCF ``qpos0``.

The MJCF blob travels inside a Menagerie ``.tar.gz`` (meshes included); :func:`fetch_asset`
extracts it and returns the path to ``wonik_allegro/right_hand.xml``.
"""

from typing import Final

from genelab.actuator import ImplicitPDActuatorCfg
from genelab.entity import ArticulationCfg
from genelab.registry import register_robot
from genelab.utils.download import AssetSpec, fetch_asset

_MJCF: Final = AssetSpec(
    name="allegro",
    url="https://raw.githubusercontent.com/KraHsu/genelab-assets/main/wonik_allegro/wonik_allegro.tar.gz",
    md5="24521cbb8b4093b4922495885ce6671c",
    filename="wonik_allegro.tar.gz",
    archive_member="wonik_allegro/right_hand.xml",
)


def AllegroHandCfg() -> ArticulationCfg:
    """Return a fresh :class:`ArticulationCfg` for the 16-DoF Allegro right hand.

    One actuator group spans all 16 finger joints (``(ff|mf|rf|th)j[0-3]``) with the
    Isaac Lab AllegroHand gains. Rest pose is the open hand (all joints 0).
    """
    mjcf_path = fetch_asset(_MJCF)
    return ArticulationCfg(
        mjcf_path=str(mjcf_path),
        init_pos=(0.0, 0.0, 0.0),
        actuators={
            "fingers": ImplicitPDActuatorCfg(
                target_names_expr=(r"(ff|mf|rf|th)j[0-3]",),
                stiffness=3.0,
                damping=0.1,
                effort_limit=0.5,
            ),
        },
    )


register_robot(
    "allegro",
    AllegroHandCfg,
    description="Wonik Allegro 16-DoF dexterous hand (Menagerie source); single implicit-PD group.",
    cfg_type=ArticulationCfg,
    examples=[
        "genelab info allegro",
        "genelab list robots",
    ],
)
