"""Internal torch helpers shared by entity wrappers."""

from typing import Any

import torch


def to_tensor(value: Any, device: str) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    return torch.as_tensor(value, device=device, dtype=torch.float)


def quat_rotate_inverse(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Rotate ``vec`` (world frame) into the body frame defined by ``quat`` (wxyz)."""
    w = quat[..., 0:1]
    xyz = quat[..., 1:4]
    t = 2.0 * torch.cross(xyz, vec, dim=-1)
    return vec - w * t + torch.cross(xyz, t, dim=-1)
