"""Tests for the bundled Wuji-hand example: bundled assets, joint-name ordering,
MJCF resolution, trajectory loading, and the joint-mapping / playback target math.

These exercise ``genelab_examples`` robot logic (not the CLI) — they were split
out of ``tests/test_cli.py`` to keep that module focused on the CLI surface.
"""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import pytest

from genelab.cli import main
from genelab_examples.wuji_hand.assets import (
    DEFAULT_DESC_DIR,
    DEFAULT_TRAJECTORY,
    candidate_mjcf_paths,
    load_trajectory,
    resolve_mjcf_path,
    wuji_joint_names,
)
from genelab_examples.wuji_hand.sim import build_joint_mapping, trajectory_target

type FloatArray = NDArray[np.floating]


class _FakeWujiJoint:
    def __init__(self, dof_idx: int) -> None:
        self.dofs_idx_local = [dof_idx]


class _FakeWujiEntity:
    n_dofs = 24

    def __init__(self, names: list[str]) -> None:
        self.names = {name: i + 4 for i, name in enumerate(names)}

    def get_joint(self, name: str) -> _FakeWujiJoint:
        if name not in self.names:
            raise KeyError(name)
        return _FakeWujiJoint(self.names[name])

    def get_dofs_limit(self, dof_indices: list[int]) -> tuple[FloatArray, FloatArray]:
        n_dofs = len(dof_indices)
        return np.full(n_dofs, -1.0, dtype=np.float32), np.full(n_dofs, 1.0, dtype=np.float32)

    def set_dofs_kp(self, kp: FloatArray, dofs_idx_local: list[int]) -> None:
        _ = (kp, dofs_idx_local)

    def set_dofs_kv(self, kv: FloatArray, dofs_idx_local: list[int]) -> None:
        _ = (kv, dofs_idx_local)

    def control_dofs_position(self, position: FloatArray, dof_indices: list[int]) -> None:
        _ = (position, dof_indices)

    def set_dofs_position(
        self, position: FloatArray, dof_indices: list[int], zero_velocity: bool
    ) -> None:
        _ = (position, dof_indices, zero_velocity)


def test_wuji_default_trajectory_is_bundled_with_package_assets() -> None:
    assert "genelab_examples/wuji_hand/data/wave.npy" in DEFAULT_TRAJECTORY.as_posix()
    assert DEFAULT_TRAJECTORY.exists()


def test_wuji_default_description_assets_are_bundled_with_package_assets() -> None:
    assert "genelab_examples/wuji_hand/description" in DEFAULT_DESC_DIR.as_posix()
    assert resolve_mjcf_path(DEFAULT_DESC_DIR, "left").exists()
    assert resolve_mjcf_path(DEFAULT_DESC_DIR, "right").exists()
    assert (DEFAULT_DESC_DIR / "meshes" / "left" / "left_palm_link.STL").exists()
    assert (DEFAULT_DESC_DIR / "meshes" / "right" / "right_palm_link.STL").exists()


def test_wuji_hand_is_registered_for_cli(capsys: pytest.CaptureFixture[str]) -> None:
    main(["--import", "genelab_examples.tasks", "list", "tasks"])

    assert "GeneLab-Wuji-Hand-Playback-v0" in capsys.readouterr().out


def test_wuji_joint_names_match_reference_order() -> None:
    names = wuji_joint_names("right")

    assert len(names) == 20
    assert names[:4] == [
        "right_finger1_joint1",
        "right_finger1_joint2",
        "right_finger1_joint3",
        "right_finger1_joint4",
    ]
    assert names[-1] == "right_finger5_joint4"


def test_wuji_mjcf_resolution_uses_flat_layout(tmp_path: Path) -> None:
    desc = tmp_path / "description"
    mjcf = desc / "mjcf" / "left.xml"
    mjcf.parent.mkdir(parents=True)
    mjcf.write_text("<mujoco />")

    assert candidate_mjcf_paths(desc, "left")[0] == mjcf
    assert resolve_mjcf_path(desc, "left") == mjcf


def test_wuji_trajectory_loader_requires_2d_array(tmp_path: Path) -> None:
    path = tmp_path / "bad.npy"
    np.save(path, np.zeros(20, dtype=np.float32))

    try:
        load_trajectory(path)
    except ValueError as exc:
        assert "2D trajectory" in str(exc)
    else:
        raise AssertionError("expected non-2D trajectory to fail")


def test_wuji_joint_mapping_skips_missing_joints() -> None:
    names = wuji_joint_names("right")
    entity = _FakeWujiEntity(names[:3] + names[4:])

    mapping = build_joint_mapping(entity, "right")

    assert mapping.mujoco_indices.tolist() == list(range(3)) + list(range(4, 20))
    assert mapping.dof_indices[:3] == [4, 5, 6]
    assert mapping.missing_joint_names == ["right_finger1_joint4"]


def test_wuji_trajectory_target_clips_to_dof_limits() -> None:
    mapping = build_joint_mapping(_FakeWujiEntity(wuji_joint_names("left")[:2]), "left")
    frame = np.zeros(20, dtype=np.float32)
    frame[:2] = [-1.0, 2.0]

    target = trajectory_target(
        frame,
        mapping,
        lower=np.array([-0.2, -0.5], dtype=np.float32),
        upper=np.array([0.7, 1.1], dtype=np.float32),
    )

    np.testing.assert_allclose(target, [-0.2, 1.1])
