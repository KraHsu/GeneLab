"""WUJI asset zoo entries — 20-DoF dexterous hand (left + right) + reorient assets.

Five fingers (``finger1``–``finger5``), each with four joints (``joint1``–``joint4``),
for 20 actuated DoF per hand. The canonical hardware description is mirrored from
[WUJI Technology](https://github.com/wuji-technology/wuji-description); the meshes travel
inside a ``.tar.gz`` shared with the bundled Wuji example, so :func:`fetch_asset` extracts
the archive and returns the path to ``wuji_hand/description/mjcf/<side>.xml``.

A single implicit-PD actuator group spans all 20 joints with uniform nominal gains —
soft position control sized to the hand's small fingers. Downstream manipulation tasks
(e.g. in-hand reorientation) override these with hardware-calibrated per-joint gains.

This module is the single source of truth for every WUJI ``AssetSpec`` — the base hand
plus the reorient-task meshes and viewer cube. They are declared at module level so the
asset zoo discovers them (``genelab asset list`` / ``asset download`` walk this package
for module-level ``AssetSpec`` instances); the bundled Wuji example imports them rather
than redeclaring its own.
"""

from typing import Final

from genelab.actuator import ImplicitPDActuatorCfg
from genelab.entity import ArticulationCfg
from genelab.registry import register_robot
from genelab.utils.download import AssetSpec, fetch_asset

_URL: Final = (
    "https://raw.githubusercontent.com/KraHsu/genelab-assets/main/wuji_hand/wuji_hand.tar.gz"
)
_MD5: Final = "46827dfc417773d469a75347a072e82e"

WUJI_HAND_SPEC: Final = AssetSpec(
    name="wuji_hand",
    url=_URL,
    md5=_MD5,
    filename="wuji_hand.tar.gz",
    archive_member="wuji_hand/description/mjcf/right.xml",
)
"""20-DoF dexterous hand description (MJCF + meshes), left + right."""

WUJI_HAND_REORIENT_SPEC: Final = AssetSpec(
    name="wuji_hand_reorient",
    url="https://raw.githubusercontent.com/KraHsu/genelab-assets/main/wuji_hand_reorient/wuji_hand_reorient.tar.gz",
    md5="68bed6d8f0fe4adc81ac8aa7f62cdfbe",
    filename="wuji_hand_reorient.tar.gz",
    archive_member="wuji_hand_reorient/meshes/right/right_palm_link.STL",
)
"""mjlab-tuned reorient right-hand meshes (~4 MB), paired with the in-tree ``right_mjlab.xml``."""

WUJI_CUBE_SPEC: Final = AssetSpec(
    name="wuji_cube",
    url="https://raw.githubusercontent.com/KraHsu/genelab-assets/main/wuji_cube/wuji_cube.tar.gz",
    md5="f77eff83a9ca8ade2966d5202b5d337d",
    filename="wuji_cube.tar.gz",
    archive_member="wuji_cube/dex_cube.obj",
)
"""54 mm UV-textured cube (held object + goal marker) for the reorient viewer."""


def _spec(side: str) -> AssetSpec:
    if side == "right":
        return WUJI_HAND_SPEC
    return AssetSpec(
        name="wuji_hand",
        url=_URL,
        md5=_MD5,
        filename="wuji_hand.tar.gz",
        archive_member=f"wuji_hand/description/mjcf/{side}.xml",
    )


def WujiHandCfg(side: str = "right") -> ArticulationCfg:
    """Return a fresh :class:`ArticulationCfg` for the 20-DoF WUJI hand.

    Args:
      side: ``"right"`` or ``"left"``. Selects the hand variant and prefixes the joint /
        link names accordingly.

    One actuator group spans all 20 finger joints (``<side>_finger[1-5]_joint[1-4]``) with
    uniform nominal gains. The rest pose keeps every joint at 0 except ``finger1_joint1``,
    whose lower limit is above 0.
    """
    if side not in ("right", "left"):
        raise ValueError(f"side must be 'right' or 'left', got {side!r}")
    mjcf_path = fetch_asset(_spec(side))
    return ArticulationCfg(
        mjcf_path=str(mjcf_path),
        init_pos=(0.0, 0.0, 0.0),
        default_joint_pos={f"{side}_finger1_joint1": 0.1},
        actuators={
            "fingers": ImplicitPDActuatorCfg(
                target_names_expr=(rf"{side}_finger[1-5]_joint[1-4]",),
                stiffness=0.5,
                damping=0.02,
                effort_limit=0.5,
            ),
        },
    )


for _side in ("right", "left"):
    register_robot(
        f"wuji_hand_{_side}",
        (lambda s: lambda: WujiHandCfg(s))(_side),
        description=(
            f"WUJI Hand 20-DoF dexterous {_side} hand (wuji-description source); "
            "single implicit-PD group with nominal gains."
        ),
        cfg_type=ArticulationCfg,
        examples=[
            f"genelab info wuji_hand_{_side}",
            "genelab list robots",
        ],
    )
