"""Reorient WUJI hand asset: in-tree mjlab MJCF + meshes fetched from the asset zoo.

The mjlab-tuned ``right_mjlab.xml`` (calibrated actuators, fingertip sites, ``_col``
collision geoms, soft-pad ``sol_params``) ships in-tree as text. Its ~4 MB mesh set lives
in genelab-assets. :func:`resolve_reorient_mjcf` downloads the meshes and writes a copy of
the MJCF whose ``meshdir`` points at the cached mesh directory, so Genesis resolves the
mesh files without bundling them in the source tree.
"""

import re
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

from genelab.actuator import ImplicitPDActuatorCfg
from genelab.asset_zoo.wuji_hand import (
    WUJI_CUBE_SPEC as _CUBE,
    WUJI_HAND_REORIENT_SPEC as _MESHES,
)
from genelab.entity import ArticulationCfg
from genelab.utils.download import fetch_asset

from genelab_wuji.reorient.constants import (
    REORIENT_JOINT_POS,
    REORIENT_ROBOT_ROOT_POS,
    REORIENT_ROBOT_ROOT_ROT,
)

# Reorient meshes (``_MESHES``) and the viewer cube (``_CUBE``) are declared in the central
# asset zoo (genelab.asset_zoo.wuji_hand) so ``genelab asset list``/``download`` discover
# them; imported here as the example's handles.
_PACKAGE_ROOT = Path(__file__).resolve().parent
_MJCF_TEMPLATE = _PACKAGE_ROOT / "mjcf" / "right_mjlab.xml"


def resolve_cube_mesh() -> str:
    """Download (once) and return the path to the 54 mm textured cube OBJ."""
    return str(fetch_asset(_CUBE))


def resolve_reorient_mjcf() -> Path:
    """Fetch the reorient meshes and return a Genesis-loadable MJCF path.

    Writes a copy of the in-tree ``right_mjlab.xml`` with ``meshdir`` rewritten to the
    cached mesh directory's absolute path, alongside the extracted meshes.
    """
    mesh_file = fetch_asset(_MESHES)
    mesh_dir = mesh_file.parent  # <cache>/.../wuji_hand_reorient/meshes/right
    xml = _MJCF_TEMPLATE.read_text()
    xml = re.sub(r'meshdir="[^"]*"', f'meshdir="{mesh_dir.as_posix()}"', xml, count=1)
    out = mesh_dir.parents[1] / "right_mjlab.xml"  # .../wuji_hand_reorient/right_mjlab.xml
    out.write_text(xml)
    return out


@lru_cache(maxsize=1)
def _calibrated_gains() -> tuple[dict[str, float], dict[str, float], float]:
    """Parse per-joint kp / kv and the max force range from the embedded MJCF actuators."""
    root = ET.fromstring(_MJCF_TEMPLATE.read_text())
    kp: dict[str, float] = {}
    kv: dict[str, float] = {}
    max_force = 0.0
    for pos in root.iter("position"):
        joint = pos.get("joint")
        assert joint is not None
        kp[joint] = float(pos.get("kp", "0"))
        kv[joint] = float(pos.get("kv", "0"))
        lo, hi = (float(v) for v in pos.get("forcerange", "0 0").split())
        max_force = max(max_force, abs(lo), abs(hi))
    return kp, kv, max_force


def wuji_hand_reorient_cfg() -> ArticulationCfg:
    """Return the fixed-base WUJI right-hand articulation for the reorient task.

    One implicit-PD group spans all 20 finger joints with the hardware-calibrated per-joint
    ``kp`` / ``kv`` read from the MJCF; the root is rotated palm-up.
    """
    kp, kv, max_force = _calibrated_gains()
    return ArticulationCfg(
        mjcf_path=str(resolve_reorient_mjcf()),
        init_pos=REORIENT_ROBOT_ROOT_POS,
        init_quat=REORIENT_ROBOT_ROOT_ROT,
        default_joint_pos=dict(REORIENT_JOINT_POS),
        actuators={
            "fingers": ImplicitPDActuatorCfg(
                target_names_expr=(r"right_finger[1-5]_joint[1-4]",),
                stiffness=kp,
                damping=kv,
                effort_limit=max_force,
            ),
        },
    )
