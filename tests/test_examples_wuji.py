"""Tests for the bundled Wuji-hand example: bundled assets, joint-name ordering,
MJCF resolution, trajectory loading, and the joint-mapping / playback target math.

These exercise ``genelab_wuji`` robot logic (not the CLI) — they were split
out of ``tests/test_cli.py`` to keep that module focused on the CLI surface.
"""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import pytest

from genelab.cli import main
from genelab_wuji.wuji_hand.assets import (
    DEFAULT_TRAJECTORY,
    WUJI_HAND_DESCRIPTION,
    candidate_mjcf_paths,
    load_trajectory,
    resolve_mjcf_path,
    wuji_joint_names,
)
from genelab_wuji.wuji_hand.sim import build_joint_mapping, trajectory_target

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
    assert "genelab_wuji/wuji_hand/data/wave.npy" in DEFAULT_TRAJECTORY.as_posix()
    assert DEFAULT_TRAJECTORY.exists()


def test_wuji_env_cfg_inherits_device_default() -> None:
    """The Wuji playback scene uses the base ``ManagerBasedEnvCfg``; ``device`` lives on the
    base cfg (PR #153), so the demo cfg inherits it."""
    from genelab_wuji.wuji_hand.config import WujiEnvCfg

    assert WujiEnvCfg().device == "cuda"


def test_wuji_play_with_rl_flags_runs_scene_demo(monkeypatch: pytest.MonkeyPatch) -> None:
    """P8/P9 regression: ``genelab play GeneLab-Wuji-Hand-Playback-v0 --agent zero --steps 5``
    must run the fixed-trajectory scene demo (``WujiHandPlaybackEnv.play``), honoring
    ``--steps``, and must NOT route through the RL play helper — ``play_task`` → ``build_env``
    would crash building ``ManagerBasedRlEnv`` from the non-RL ``WujiEnvCfg``. Exercises the
    same dispatch path the CLI uses, with Genesis stubbed out so it runs headless."""
    from genelab.cli import main
    from genelab_wuji import envs

    captured: dict[str, object] = {}

    def _fake_play(self: envs.WujiHandPlaybackEnv) -> None:
        captured["steps"] = self.cfg.simulation.steps

    monkeypatch.setattr(envs.WujiHandPlaybackEnv, "play", _fake_play)

    # Patch the real module's attribute (not sys.modules) so the dispatcher's
    # `from genelab.rl import play_task` resolves to the sentinel while entry-point
    # extension loading can still `from genelab.rl import RslRlModelCfg`.
    import genelab.rl as genelab_rl

    def _no_play_task(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("non-RL Wuji demo must not route through play_task")

    monkeypatch.setattr(genelab_rl, "play_task", _no_play_task)

    main(
        [
            "--import",
            "genelab_wuji.tasks",
            "play",
            "GeneLab-Wuji-Hand-Playback-v0",
            "--agent",
            "zero",
            "--steps",
            "5",
        ]
    )

    assert captured["steps"] == 5


def test_wuji_description_spec_targets_asset_zoo() -> None:
    """The hand description is fetched from genelab-assets, not bundled in-tree."""
    assert WUJI_HAND_DESCRIPTION.url.startswith(
        "https://raw.githubusercontent.com/KraHsu/genelab-assets/"
    )
    assert len(WUJI_HAND_DESCRIPTION.md5) == 32
    assert WUJI_HAND_DESCRIPTION.archive_member.endswith("description/mjcf/right.xml")


def test_wuji_description_resolved_from_downloaded_asset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``desc_dir=None`` resolves the MJCFs from the extracted asset tree.

    ``fetch_asset`` is monkeypatched (like the asset-zoo tests) so the resolution logic is
    exercised without touching the network."""
    desc = tmp_path / "wuji_hand" / "description"
    (desc / "mjcf").mkdir(parents=True)
    (desc / "mjcf" / "left.xml").write_text("<mujoco/>")
    (desc / "mjcf" / "right.xml").write_text("<mujoco/>")
    member = desc / "mjcf" / "right.xml"
    monkeypatch.setattr("genelab_wuji.wuji_hand.assets.fetch_asset", lambda spec: member)

    assert resolve_mjcf_path(None, "left").name == "left.xml"
    assert resolve_mjcf_path(None, "right").name == "right.xml"


def test_wuji_hand_is_registered_for_cli(capsys: pytest.CaptureFixture[str]) -> None:
    main(["--import", "genelab_wuji.tasks", "list", "tasks"])

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
