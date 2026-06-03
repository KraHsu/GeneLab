"""Wuji hand asset path and trajectory helpers.

The hand description (MJCF + meshes) is not bundled in the source tree; it is fetched on
demand from the GeneLab asset zoo (``genelab-assets``) and md5-verified into the local
cache. Only the small playback trajectory (``data/wave.npy``) ships with the package.
Pass ``desc_dir=None`` (the default) to use the downloaded description, or an explicit
path to override it.
"""

from pathlib import Path
import numpy as np
from numpy.typing import NDArray

from genelab.utils.download import AssetSpec, fetch_asset

type FloatArray = NDArray[np.float32]

REPO_ROOT = Path(__file__).resolve().parents[4]
PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_TRAJECTORY = PACKAGE_ROOT / "data" / "wave.npy"
SIDES = ("left", "right")

# Full left+right hand description (MJCF + ~5 MB of STL meshes) hosted as a .tar.gz in the
# genelab-assets repo, so the source tree stays lean. ``archive_member`` points at the
# right-hand MJCF; the description directory is its grandparent in the extracted tree.
WUJI_HAND_DESCRIPTION = AssetSpec(
    name="wuji_hand",
    url="https://raw.githubusercontent.com/KraHsu/genelab-assets/main/wuji_hand/wuji_hand.tar.gz",
    md5="46827dfc417773d469a75347a072e82e",
    filename="wuji_hand.tar.gz",
    archive_member="wuji_hand/description/mjcf/right.xml",
)


def fetch_description_dir() -> Path:
    """Download (once) and return the local Wuji hand ``description`` directory."""
    entry = fetch_asset(WUJI_HAND_DESCRIPTION)
    return entry.parent.parent


def wuji_joint_names(side: str) -> list[str]:
    """Return Wuji hand joint names in the MuJoCo trajectory order."""
    if side not in SIDES:
        raise ValueError(f"side must be one of {SIDES}, got {side!r}")
    return [
        f"{side}_finger{finger}_joint{joint}" for finger in range(1, 6) for joint in range(1, 5)
    ]


def resolve_description_dir(desc_dir: Path | str | None = None) -> Path:
    if desc_dir is None:
        return fetch_description_dir()
    path = Path(desc_dir).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.exists():
        raise FileNotFoundError(
            f"Wuji hand description directory not found: {path}. "
            "Omit env.robot.desc_dir to download the bundled asset, or pass a valid path."
        )
    return path


def candidate_mjcf_paths(desc_dir: Path | str, side: str) -> list[Path]:
    base = Path(desc_dir)
    return [
        base / "mjcf" / f"{side}.xml",
        base / "mjcf" / side / "wujihand.xml",
        base / "mjcf" / side / f"{side}.xml",
    ]


def resolve_mjcf_path(desc_dir: Path | str, side: str) -> Path:
    if side not in SIDES:
        raise ValueError(f"side must be one of {SIDES}, got {side!r}")
    resolved_dir = resolve_description_dir(desc_dir)
    checked = candidate_mjcf_paths(resolved_dir, side)
    for path in checked:
        if path.exists():
            return path
    checked_text = "\n".join(f"  - {path}" for path in checked)
    raise FileNotFoundError(f"No Wuji {side} MJCF asset found. Checked:\n{checked_text}")


def load_trajectory(path: Path | str = DEFAULT_TRAJECTORY) -> FloatArray:
    trajectory_path = Path(path).expanduser()
    if not trajectory_path.is_absolute():
        trajectory_path = REPO_ROOT / trajectory_path
    if not trajectory_path.exists():
        raise FileNotFoundError(
            f"Wuji trajectory not found: {trajectory_path}. "
            "Use the bundled trajectory or override env.robot.trajectory."
        )
    trajectory = np.load(trajectory_path)
    if trajectory.ndim != 2:
        raise ValueError(f"Expected a 2D trajectory array, got shape {trajectory.shape}")
    if trajectory.shape[1] < len(wuji_joint_names("right")):
        raise ValueError(
            f"Expected at least {len(wuji_joint_names('right'))} trajectory columns, got {trajectory.shape[1]}"
        )
    return trajectory.astype(np.float32, copy=False)
