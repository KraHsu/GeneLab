"""Reproduce the real cube in the Genesis sim world from its tag-frame pose.

The observer reports the cube pose in the wrist-tag frame; to draw it in sim we
lift it back to sim-world coordinates given the tag's world pose. These tests pin
that lift as the exact inverse of the obs-side ``cube_*_world_to_tag`` transform,
so what the policy sees and what the viewer draws agree.
"""

import numpy as np

from genelab_wuji.deploy.real2sim import (
    cube_pose_in_tag_to_world,
    cube_pose_world_to_tag,
    tag_pose_in_world,
)


def _quat_z(angle: float) -> np.ndarray:
    return np.array([np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)])


def test_cube_world_to_tag_and_back_round_trips() -> None:
    tag_pos_w = np.array([0.02, 0.0, 0.55])
    tag_quat_w = _quat_z(np.pi / 5)
    cube_pos_w = np.array([-0.05, 0.01, 0.56])
    cube_quat_w = _quat_z(-np.pi / 7)

    cube_pos_tag, cube_quat_tag = cube_pose_world_to_tag(
        tag_pos_w, tag_quat_w, cube_pos_w, cube_quat_w
    )
    back_pos, back_quat = cube_pose_in_tag_to_world(
        tag_pos_w, tag_quat_w, cube_pos_tag, cube_quat_tag
    )

    assert np.allclose(back_pos, cube_pos_w, atol=1e-9)
    # quaternion equality up to sign
    assert np.allclose(back_quat, cube_quat_w, atol=1e-9) or np.allclose(
        back_quat, -cube_quat_w, atol=1e-9
    )


def test_cube_at_tag_origin_lifts_to_tag_world_position() -> None:
    # A cube reported at the tag origin must render exactly at the tag's world pose.
    tag_pos_w = np.array([0.1, -0.2, 0.5])
    tag_quat_w = _quat_z(0.3)

    cube_pos_w, cube_quat_w = cube_pose_in_tag_to_world(
        tag_pos_w, tag_quat_w, np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])
    )
    assert np.allclose(cube_pos_w, tag_pos_w, atol=1e-9)
    assert np.allclose(cube_quat_w, tag_quat_w, atol=1e-9)


def test_tag_pose_in_world_applies_tag_in_palm_offset() -> None:
    # With the palm at the origin (identity), the tag world pose equals the
    # constant TAG_IN_PALM transform from the reorient constants.
    from genelab_wuji.reorient.constants import TAG_IN_PALM_POS, TAG_IN_PALM_QUAT_WXYZ

    tag_pos_w, tag_quat_w = tag_pose_in_world(np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))
    assert np.allclose(tag_pos_w, TAG_IN_PALM_POS, atol=1e-9)
    assert np.allclose(tag_quat_w, TAG_IN_PALM_QUAT_WXYZ, atol=1e-9)
