"""SO(3) helpers used across reorient terms (built on genelab.utils.math, wxyz)."""

import torch

from genelab.utils.math import matrix_from_quat


def matrix_to_rotation_6d(mat: torch.Tensor) -> torch.Tensor:
    """Flatten the first two rows of a rotation matrix to the Zhou et al. 6D representation.

    Args:
      mat: ``(..., 3, 3)`` rotation matrices.

    Returns:
      ``(..., 6)`` continuous rotation features (rows 0 and 1, row-major).
    """
    return mat[..., :2, :].reshape(*mat.shape[:-2], 6)


def quat_to_rotation_6d(quat: torch.Tensor) -> torch.Tensor:
    """6D rotation features from a wxyz quaternion ``(..., 4)`` -> ``(..., 6)``."""
    return matrix_to_rotation_6d(matrix_from_quat(quat))


def random_quat(n: int, device: torch.device | str) -> torch.Tensor:
    """Sample ``n`` quaternions uniformly on SO(3) (Shoemake; wxyz, canonicalized ``w >= 0``)."""
    u = torch.rand(n, 3, device=device)
    r1 = torch.sqrt(1.0 - u[:, 0])
    r2 = torch.sqrt(u[:, 0])
    t1 = 2.0 * torch.pi * u[:, 1]
    t2 = 2.0 * torch.pi * u[:, 2]
    quat = torch.stack(
        [r2 * torch.cos(t2), r1 * torch.sin(t1), r1 * torch.cos(t1), r2 * torch.sin(t2)],
        dim=-1,
    )
    return torch.where(quat[:, :1] >= 0, quat, -quat)
