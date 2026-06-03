"""WUJI Hand asset zoo entry — 20-DoF dexterous hand (left + right).

Five fingers (``finger1``–``finger5``), each with four joints (``joint1``–``joint4``),
for 20 actuated DoF per hand. The canonical hardware description is mirrored from
[WUJI Technology](https://github.com/wuji-technology/wuji-description); the meshes travel
inside a ``.tar.gz`` shared with the bundled Wuji example, so :func:`fetch_asset` extracts
the archive and returns the path to ``wuji_hand/description/mjcf/<side>.xml``.

A single implicit-PD actuator group spans all 20 joints with uniform nominal gains —
soft position control sized to the hand's small fingers. Downstream manipulation tasks
(e.g. in-hand reorientation) override these with hardware-calibrated per-joint gains.
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


def _spec(side: str) -> AssetSpec:
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
