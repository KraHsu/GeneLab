"""real2sim coordinate transforms for the Wuji-hand deploy pipeline.

The vision pipeline detects the cube pose in the *camera* frame and the wrist
AprilTag pose in the *camera* frame. ``cube_cam_to_tag`` lifts the cube into the
wrist-tag frame — the exact frame the policy was trained on. These tests pin that
math (pure numpy, no simulator) so the real cube position reproduces correctly.
"""

import numpy as np

from genelab_wuji.deploy.frame_transform import cube_cam_to_tag


def _rand_rot(rng: np.random.Generator) -> np.ndarray:
    R, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    if np.linalg.det(R) < 0:  # keep it a proper rotation (det = +1)
        R[:, 0] = -R[:, 0]
    return R


def test_cube_sitting_on_tag_maps_to_tag_origin() -> None:
    # When the cube pose in camera frame equals the tag pose in camera frame, the
    # cube sits exactly at the tag origin: tag-frame position ~0, rotation ~identity.
    rng = np.random.default_rng(0)
    R_tag_cam, _ = np.linalg.qr(rng.standard_normal((3, 3)))  # random rotation
    t_tag_cam = rng.standard_normal(3)

    R_cube_tag, t_cube_tag = cube_cam_to_tag(R_tag_cam, t_tag_cam, R_tag_cam, t_tag_cam)

    assert np.allclose(t_cube_tag, np.zeros(3), atol=1e-9)
    assert np.allclose(R_cube_tag, np.eye(3), atol=1e-9)


def test_known_cube_pose_in_tag_is_recovered_through_camera() -> None:
    # Define a known cube pose in the tag frame, push it out to the camera frame,
    # then lift it back: cube_cam_to_tag must recover the original tag-frame pose.
    rng = np.random.default_rng(7)
    R_tag_cam, t_tag_cam = _rand_rot(rng), rng.standard_normal(3)
    R_cube_tag_true, t_cube_tag_true = _rand_rot(rng), np.array([0.01, -0.02, 0.03])

    # Compose the cube pose into the camera frame.
    R_cube_cam = R_tag_cam @ R_cube_tag_true
    t_cube_cam = R_tag_cam @ t_cube_tag_true + t_tag_cam

    R_cube_tag, t_cube_tag = cube_cam_to_tag(R_tag_cam, t_tag_cam, R_cube_cam, t_cube_cam)

    assert np.allclose(t_cube_tag, t_cube_tag_true, atol=1e-9)
    assert np.allclose(R_cube_tag, R_cube_tag_true, atol=1e-9)
