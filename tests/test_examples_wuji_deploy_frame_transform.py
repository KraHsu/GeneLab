"""wxyz quaternion helpers used by the deploy pipeline (pure numpy)."""

import numpy as np

from genelab_wuji.deploy.frame_transform import (
    quat_apply,
    quat_conjugate,
    quat_mul,
)


def _quat_z(angle: float) -> np.ndarray:
    return np.array([np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)])


def test_quat_apply_rotates_x_to_y_about_z() -> None:
    q = _quat_z(np.pi / 2)  # +90deg about z
    out = quat_apply(q, np.array([1.0, 0.0, 0.0]))
    assert np.allclose(out, [0.0, 1.0, 0.0], atol=1e-9)


def test_quat_mul_composes_rotations() -> None:
    q45 = _quat_z(np.pi / 4)
    q90 = _quat_z(np.pi / 2)
    composed = quat_mul(q45, q45)  # 45 + 45 = 90
    out = quat_apply(composed, np.array([1.0, 0.0, 0.0]))
    expected = quat_apply(q90, np.array([1.0, 0.0, 0.0]))
    assert np.allclose(out, expected, atol=1e-9)


def test_quat_conjugate_inverts_rotation() -> None:
    q = _quat_z(np.pi / 3)
    identity = quat_mul(q, quat_conjugate(q))
    assert np.allclose(identity, [1.0, 0.0, 0.0, 0.0], atol=1e-9)
