"""Wuji hand asset path and trajectory helpers."""

from pathlib import Path
import numpy as np
from numpy.typing import NDArray

type FloatArray = NDArray[np.float32]

REPO_ROOT = Path(__file__).resolve().parents[4]
PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_DESC_DIR = PACKAGE_ROOT / "description"
DEFAULT_TRAJECTORY = PACKAGE_ROOT / "data" / "wave.npy"
SIDES = ("left", "right")


def wuji_joint_names(side: str) -> list[str]:
    """Return Wuji hand joint names in the MuJoCo trajectory order."""
    if side not in SIDES:
        raise ValueError(f"side must be one of {SIDES}, got {side!r}")
    return [
        f"{side}_finger{finger}_joint{joint}" for finger in range(1, 6) for joint in range(1, 5)
    ]


def resolve_description_dir(desc_dir: Path | str = DEFAULT_DESC_DIR) -> Path:
    path = Path(desc_dir).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.exists():
        raise FileNotFoundError(
            f"Wuji hand description directory not found: {path}. "
            "Use the bundled description assets or override env.robot.desc_dir."
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
