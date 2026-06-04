"""Quaternion + transform helpers (wxyz convention).

Ported from ``utils.lab_api.math``; trimmed to the surface used by GeneLab's MDP.
All quaternions use the ``(w, x, y, z)`` convention.
"""

import torch


def normalize(x: torch.Tensor, eps: float = 1.0e-9) -> torch.Tensor:
    """Normalize ``x`` along the last dim to unit length."""
    return x / x.norm(p=2, dim=-1).clamp(min=eps).unsqueeze(-1)


def matrix_from_quat(quaternions: torch.Tensor) -> torch.Tensor:
    """Rotation matrix from wxyz quaternions. Output shape: ``(..., 3, 3)``."""
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)
    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    """Conjugate of a wxyz quaternion: negate the imaginary part."""
    shape = q.shape
    q = q.reshape(-1, 4)
    return torch.cat((q[..., 0:1], -q[..., 1:]), dim=-1).view(shape)


def quat_inv(q: torch.Tensor, eps: float = 1.0e-9) -> torch.Tensor:
    """Inverse of a (unit-ish) wxyz quaternion."""
    return quat_conjugate(q) / q.pow(2).sum(dim=-1, keepdim=True).clamp(min=eps)


def quat_from_euler_xyz(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """XYZ-convention Euler -> wxyz quaternion."""
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    qw = cy * cr * cp + sy * sr * sp
    qx = cy * sr * cp - sy * cr * sp
    qy = cy * cr * sp + sy * sr * cp
    qz = sy * cr * cp - cy * sr * sp
    return torch.stack([qw, qx, qy, qz], dim=-1)


def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product of two wxyz quaternions. Shapes must match."""
    if q1.shape != q2.shape:
        raise ValueError(f"quaternion shape mismatch: {q1.shape} != {q2.shape}")
    shape = q1.shape
    q1 = q1.reshape(-1, 4)
    q2 = q2.reshape(-1, 4)
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)
    return torch.stack([w, x, y, z], dim=-1).view(shape)


def yaw_quat(quat: torch.Tensor) -> torch.Tensor:
    """Project a wxyz quaternion onto the z-axis (yaw-only rotation)."""
    shape = quat.shape
    q = quat.view(-1, 4)
    qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    yaw = torch.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    out = torch.zeros_like(q)
    out[:, 0] = torch.cos(yaw / 2)
    out[:, 3] = torch.sin(yaw / 2)
    return normalize(out).view(shape)


def quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Rotate ``vec`` by ``quat`` (wxyz)."""
    shape = vec.shape
    q = quat.reshape(-1, 4)
    v = vec.reshape(-1, 3)
    xyz = q[:, 1:]
    t = torch.cross(xyz, v, dim=-1) * 2
    return (v + q[:, 0:1] * t + torch.cross(xyz, t, dim=-1)).view(shape)


def quat_apply_inverse(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Rotate ``vec`` by the inverse of ``quat`` (wxyz)."""
    shape = vec.shape
    q = quat.reshape(-1, 4)
    v = vec.reshape(-1, 3)
    xyz = q[:, 1:]
    t = torch.cross(xyz, v, dim=-1) * 2
    return (v - q[:, 0:1] * t + torch.cross(xyz, t, dim=-1)).view(shape)


def axis_angle_from_quat(quat: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    """Axis-angle 3-vector (rotation axis scaled by angle in radians) from a wxyz
    quaternion. The sign is canonicalised so the scalar part is non-negative."""
    quat = quat * (1.0 - 2.0 * (quat[..., 0:1] < 0.0))
    mag = torch.linalg.norm(quat[..., 1:], dim=-1)
    half_angle = torch.atan2(mag, quat[..., 0])
    angle = 2.0 * half_angle
    sin_half_over_angle = torch.where(
        angle.abs() > eps, torch.sin(half_angle) / angle, 0.5 - angle * angle / 48
    )
    return quat[..., 1:4] / sin_half_over_angle.unsqueeze(-1)


def quat_error_magnitude(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Geodesic angle (radians) between two wxyz quaternions. Returns a scalar per row."""
    diff = quat_mul(q1, quat_conjugate(q2))
    return torch.norm(axis_angle_from_quat(diff), dim=-1)


def subtract_frame_transforms(
    t01: torch.Tensor,
    q01: torch.Tensor,
    t02: torch.Tensor | None = None,
    q02: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(t12, q12)`` — pose of frame 2 expressed in frame 1.

    ``T12 = inv(T01) @ T02``. Defaults: ``t02 = 0``, ``q02 = identity``.
    """
    q10 = quat_inv(q01)
    q12 = quat_mul(q10, q02) if q02 is not None else q10
    if t02 is not None:
        t12 = quat_apply(q10, t02 - t01)
    else:
        t12 = quat_apply(q10, -t01)
    return t12, q12


def sample_uniform(
    lower: torch.Tensor | float,
    upper: torch.Tensor | float,
    size: int | tuple[int, ...],
    device: str | torch.device,
) -> torch.Tensor:
    """Uniform sample in ``[lower, upper)`` with the given shape."""
    if isinstance(size, int):
        size = (size,)
    return torch.rand(*size, device=device) * (upper - lower) + lower


__all__ = [
    "axis_angle_from_quat",
    "matrix_from_quat",
    "normalize",
    "quat_apply",
    "quat_apply_inverse",
    "quat_conjugate",
    "quat_error_magnitude",
    "quat_from_euler_xyz",
    "quat_inv",
    "quat_mul",
    "sample_uniform",
    "subtract_frame_transforms",
    "yaw_quat",
]
